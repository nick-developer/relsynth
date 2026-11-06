# SPDX-License-Identifier: Apache-2.0
"""Compare the licence register of text models with the Hugging Face hub.

    python tools/check_text_models.py [--strict]

For every hub model in src/relsynth/data/text_models.yaml, reads the hub's
`license` tag (https://huggingface.co/api/models/<id>) and compares it with the
register.  Exit code 1 when a licence differs (and, with --strict, when the hub
cannot be read).  Run by the CI job `text-lm`; no weights are downloaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# register licence -> hub license tag
HUB = {"Apache-2.0": "apache-2.0", "MIT": "mit", "Llama 3.2 Community License": "llama3.2",
       "Gemma Terms of Use": "gemma", "Qwen Research License": "other"}


def hub_licence(model_id: str) -> str | None:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{model_id}", timeout=60) as r:  # noqa: S310
        info = json.load(r)
    lic = (info.get("cardData") or {}).get("license")
    if not lic:
        lic = next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), None)
    return lic.lower() if isinstance(lic, str) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    reg = yaml.safe_load((ROOT / "src/relsynth/data/text_models.yaml").read_text())
    bad = 0
    for m in reg["models"]:
        if m["id"].startswith("spacy/"):
            continue
        want = HUB.get(m["licence"], m["licence"].lower())
        try:
            got = hub_licence(m["id"])
        except Exception as e:  # noqa: BLE001
            print(f"? {m['id']}: hub not readable ({e})")
            bad += a.strict
            continue
        ok = got == want
        print(f"{'ok' if ok else 'DIFFERS'} {m['id']}: register {m['licence']} ({want}), hub {got}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
