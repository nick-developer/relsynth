#!/usr/bin/env bash
# A local stand-in for a Databricks SQL warehouse (tests only): Apache Spark's
# Thrift server with Delta Lake, behind thrift_proxy.py, which adapts the
# Databricks SQL connector's protocol (see its docstring).  Needs Java 17+ and
#   pip install "pyspark==3.5.3" "relsynth[databricks]"
# Then:
#   export RELSYNTH_TEST_DATABRICKS_URL="databricks://token:rs-emulator-token@localhost?http_path=cliservice&catalog=spark_catalog"
#   export RELSYNTH_TEST_DATABRICKS_CONNECT_ARGS='{"_connection_uri": "http://localhost:10002/cliservice", "catalog": null, "schema": null, "enable_telemetry": false}'
#   pytest -m databricks tests/integration
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${RS_DATABRICKS_WORK:-${TMPDIR:-/tmp}/relsynth-databricks}"
DELTA=3.2.1
mkdir -p "$WORK/jars"
for j in "delta-spark_2.12/$DELTA/delta-spark_2.12-$DELTA.jar" "delta-storage/$DELTA/delta-storage-$DELTA.jar"; do
  f="$WORK/jars/$(basename "$j")"
  [ -s "$f" ] || curl -fsSL -o "$f" "https://repo1.maven.org/maven2/io/delta/$j"
done
SPARK_HOME="$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')"
export SPARK_HOME
cd "$WORK"
# Delta on the driver's class path: with --packages the Thrift server's worker
# threads cannot load Delta's classes
nohup "$SPARK_HOME/bin/spark-submit" --class org.apache.spark.sql.hive.thriftserver.HiveThriftServer2 \
  --conf "spark.driver.extraClassPath=$WORK/jars/delta-spark_2.12-$DELTA.jar:$WORK/jars/delta-storage-$DELTA.jar" \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  --conf "spark.sql.warehouse.dir=$WORK/warehouse" --conf "spark.driver.memory=${RS_SPARK_MEMORY:-4g}" \
  --conf spark.sql.session.timeZone=UTC --conf spark.sql.shuffle.partitions=4 \
  --hiveconf hive.server2.transport.mode=http --hiveconf hive.server2.thrift.http.port=10001 \
  --hiveconf hive.server2.thrift.http.path=cliservice --hiveconf hive.server2.authentication=NONE \
  spark-internal > "$WORK/thrift.log" 2>&1 &
nohup python "$HERE/thrift_proxy.py" 10002 10001 > "$WORK/proxy.log" 2>&1 &
for _ in $(seq 1 120); do
  grep -q "HiveThriftServer2 started" "$WORK/thrift.log" 2>/dev/null && { echo "Databricks stand-in on http://localhost:10002/cliservice"; exit 0; }
  sleep 2
done
echo "the Spark Thrift server did not start; see $WORK/thrift.log" >&2
exit 1
