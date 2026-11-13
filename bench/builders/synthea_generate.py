# SPDX-License-Identifier: Apache-2.0
"""Generate a Synthea population (Apache-2.0, MITRE) with the pinned release jar (G7).

    python synthea_generate.py synthea-with-dependencies.jar PATIENTS OUTDIR

Runs `java -jar` (JDK 17 or newer) with a fixed seed, clinician seed and
reference date, CSV export only, for the state of Massachusetts, and leaves
OUTDIR/csv/<table>.csv for sdv_sqlite.py (read with the 1K sample's metadata).
The same jar, arguments and Java version give the same population.
"""
import shutil
import subprocess
import sys
from pathlib import Path

jar, n, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
if shutil.which("java") is None:
    raise SystemExit("java (JDK 17 or newer) is needed to generate a Synthea population")
out.mkdir(parents=True, exist_ok=True)
subprocess.run(["java", "-Xmx6g", "-jar", str(jar), "-p", str(n), "-s", "7", "-cs", "7", "-r", "20250101",  # noqa: S603, S607
                "--exporter.baseDirectory", str(out), "--exporter.csv.export", "true",
                "--exporter.fhir.export", "false", "--exporter.hospital.fhir.export", "false",
                "--exporter.practitioner.fhir.export", "false", "--generate.only_alive_patients", "false",
                "Massachusetts"], check=True, stdout=subprocess.DEVNULL)
csv = out / "csv"
if not any(csv.glob("patients.csv")):
    raise SystemExit(f"Synthea wrote no patients.csv under {csv}")
with open(csv / "patients.csv", encoding="utf-8") as fh:
    print(f"{sum(1 for _ in fh) - 1} patients")
