# SPDX-License-Identifier: Apache-2.0
"""Generate a Synthea population (Apache-2.0, MITRE) with the pinned release jar (G7).

    python synthea_generate.py synthea-with-dependencies.jar PATIENTS OUTDIR

Runs `java -jar` (JDK 17 or newer) with a fixed seed, clinician seed,
reference date and end date, CSV export only, for the state of Massachusetts, and leaves
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
# generated into a sibling work folder and moved into place only when Java succeeded and wrote its
# patients: a failed run leaves no partial csv/ that a retry would take for a whole population
out.parent.mkdir(parents=True, exist_ok=True)
work = out.with_name(out.name + ".part")
shutil.rmtree(work, ignore_errors=True)
work.mkdir()
# -r and -e: a fixed reference and end date (the end date otherwise defaults to the clock), so the same
# jar and seeds give the same histories on any day
subprocess.run(["java", "-Xmx6g", "-jar", str(jar), "-p", str(n), "-s", "7", "-cs", "7", "-r", "20250101",  # noqa: S603, S607
                "-e", "20250101",
                "--exporter.baseDirectory", str(work), "--exporter.csv.export", "true",
                "--exporter.fhir.export", "false", "--exporter.hospital.fhir.export", "false",
                "--exporter.practitioner.fhir.export", "false", "--generate.only_alive_patients", "false",
                "Massachusetts"], check=True, stdout=subprocess.DEVNULL)
patients = work / "csv" / "patients.csv"
if not patients.is_file():
    raise SystemExit(f"Synthea wrote no patients.csv under {work / 'csv'}")
with patients.open(encoding="utf-8") as fh:
    n_written = sum(1 for _ in fh) - 1
if n_written < 1:
    raise SystemExit("Synthea wrote no patients")
shutil.rmtree(out, ignore_errors=True)
work.rename(out)
print(f"{n_written} patients")
