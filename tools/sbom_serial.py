# SPDX-License-Identifier: Apache-2.0
"""Give a reproducible CycloneDX SBOM the serial number that attestations need.

    python tools/sbom_serial.py SBOM.cdx.json WHEEL

``cyclonedx-py --output-reproducible`` leaves out ``serialNumber`` (and the
timestamp) so that the same environment gives the same file.  GitHub's SBOM
attestation (actions/attest) accepts a CycloneDX document only when it has
``bomFormat``, ``specVersion`` and ``serialNumber``, so the release stopped at
that step.  The serial number written here is a UUID 5 of the wheel's SHA-256:
the same wheel always gets the same SBOM, and a different wheel a different one.
The script fails when any of the three fields is missing afterwards.
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

REQUIRED = ("bomFormat", "specVersion", "serialNumber")


def serial(wheel: Path) -> str:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, f'https://pypi.org/project/relsynth/#sha256={digest}')}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    sbom_path, wheel = Path(argv[0]), Path(argv[1])
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    sbom.setdefault("serialNumber", serial(wheel))
    missing = [k for k in REQUIRED if not sbom.get(k)]
    if sbom.get("bomFormat") != "CycloneDX" or missing:
        print(f"{sbom_path}: not an attestable CycloneDX SBOM (missing {missing or ['bomFormat=CycloneDX']})", file=sys.stderr)
        return 1
    # header fields first, as in the CycloneDX examples
    order = ["$schema", "bomFormat", "specVersion", "serialNumber", "version"]
    out = {k: sbom[k] for k in order if k in sbom} | {k: v for k, v in sbom.items() if k not in order}
    sbom_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{sbom_path}: CycloneDX {out['specVersion']}, {len(out.get('components', []))} components, {out['serialNumber']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
