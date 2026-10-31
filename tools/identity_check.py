# SPDX-License-Identifier: Apache-2.0
"""Byte identity of the synthetic output between two source trees.

    python tools/identity_check.py --base ../main/src --head src [--scenarios S2_star S4_tree] [--seeds 1 2]

Writes every fake scenario (tests/fakedata/scenarios.py) and the quickstart
example, runs `relsynth run` on each with both trees (PYTHONPATH), and compares
the SHA-256 of every synthetic file.  Used by the CI job `identity` to show that
a change which only adds settings (with unchanged defaults) leaves every draw as
it was.  Exit code 1 lists the files that differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(folder: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.csv"))}


def run(src: str, spec: Path, out: Path, seed: int) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": str(Path(src).resolve()), "RELSYNTH_QUIET": "1"}
    env.pop("RELSYNTH_SETTINGS", None)
    cmd = [sys.executable, "-m", "relsynth", "run", str(spec), "--target", f"csv:{out}", "--seed", str(seed),
           "--json"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=spec.parent, check=False)  # noqa: S603
    return p.returncode, p.stderr[-800:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="src folder of the reference tree (e.g. main)")
    ap.add_argument("--head", default=str(ROOT / "src"), help="src folder of the tree under test")
    ap.add_argument("--scenarios", nargs="*", default=None, help="fake scenarios (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--no-quickstart", action="store_true")
    ap.add_argument("--json", help="write the comparison here")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from tests.fakedata.scenarios import SCENARIOS, write
    names = a.scenarios or list(SCENARIOS)
    tmp = Path(tempfile.mkdtemp(prefix="identity-"))
    cases = []
    for n in names:
        tables, spec, _ = SCENARIOS[n]()
        cases.append((n, write(n, tables, spec, out=tmp) / "spec.yaml"))
    if not a.no_quickstart:
        q = tmp / "quickstart"
        shutil.copytree(ROOT / "examples" / "quickstart", q)
        cases.append(("quickstart", q / "spec.yaml"))
    result, bad = {}, []
    for name, spec in cases:
        for seed in a.seeds:
            d = {}
            for side, src in (("base", a.base), ("head", a.head)):
                out = spec.parent / f"syn_{side}_{seed}"
                rc, err = run(src, spec, out, seed)
                # exit code 1 is the quality gate: the data was written
                d[side] = digest(out) if rc in (0, 1) and out.exists() else {"error": f"exit {rc}: {err}"}
            same = d["base"] == d["head"] and "error" not in d["base"]
            result[f"{name} seed {seed}"] = {"identical": same, "files": len(d["base"])}
            print(f"{name:<16} seed {seed}: {'identical' if same else 'DIFFERENT'} ({len(d['base'])} files)")
            if not same:
                diff = sorted(k for k in set(d["base"]) | set(d["head"]) if d["base"].get(k) != d["head"].get(k))
                bad.append(f"{name} seed {seed}: {diff}")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("\nnot identical:\n  " + "\n  ".join(bad))
        return 1
    print(f"\nall {len(result)} runs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
