# SPDX-License-Identifier: Apache-2.0
"""The scientific check suite (C00-C78): every proposition of
docs/proofs/layer2-proofs.pdf tested on fake data with known ground truth
(posterior recovery, calibration of the scorecard, identification, rules,
privacy, ...).  Slow (about an hour on 4 cores):

    pytest -m science                      # the whole suite, one test per check
    python legacy/checks/run_checks.py C05 # one check, from a checkout laid out as below

The suite runs once per session in a mirror of the layout it was written for
(relsynth/, fakedata/, checks/, docs/, examples/, realdata/), in a subprocess,
and each check becomes one test.  Checks on the real Chinook database need the
layer 2 data folder (RELSYNTH_LAYER2_ROOT, default /mnt/project-files/layer2) and
are skipped without it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.science, pytest.mark.slow]

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
LEGACY = PKG / "legacy"
LAYER2 = Path(os.environ.get("RELSYNTH_LAYER2_ROOT", "/mnt/project-files/layer2"))
NEEDS_LAYER2 = {"C76"}            # v09_eval reads realdata/chinook


def _ids() -> list[str]:
    src = (LEGACY / "checks" / "run_checks.py").read_text()
    return re.findall(r'@check\("(C\d+)"', src)


@pytest.fixture(scope="session")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("science") / "layer2"
    root.mkdir()
    (root / "relsynth").symlink_to(PKG / "src" / "relsynth")
    (root / "fakedata").symlink_to(LEGACY / "fakedata")
    shutil.copytree(LEGACY / "checks", root / "checks")
    for old in ("results.json", "results.md"):          # never read results of an earlier run
        (root / "checks" / old).unlink(missing_ok=True)
    shutil.copytree(PKG / "docs", root / "docs")
    shutil.copytree(PKG / "examples", root / "examples")
    (root.parent / "research").symlink_to(LEGACY / "research")
    if (LAYER2 / "realdata").is_dir():
        (root / "realdata").symlink_to(LAYER2 / "realdata")
    env = {**os.environ, "PYTHONPATH": str(root), "RELSYNTH_QUIET": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, str(root / "checks" / "run_checks.py")], cwd=root, env=env,
                   timeout=4 * 3600, check=False)
    out = root / "checks" / "results.json"
    if out.exists():
        return {r["id"]: r for r in json.loads(out.read_text())}
    return {}


@pytest.mark.parametrize("cid", _ids())
def test_check(results, cid):
    if cid in NEEDS_LAYER2 and not (LAYER2 / "realdata").is_dir():
        pytest.skip(f"{cid} needs the layer 2 real data (RELSYNTH_LAYER2_ROOT)")
    assert results, "the check suite wrote no results.json"
    r = results.get(cid)
    assert r is not None, f"{cid} did not run"
    failed = [i for i in r["items"] if not i["pass"]]
    assert r["pass"], f"{cid} {r['title']}: " + ("; ".join(f"{i['what']} = {i['value']} (target {i['target']})"
                                                         for i in failed) or (r.get("error") or "")[-800:])
