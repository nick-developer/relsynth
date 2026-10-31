# SPDX-License-Identifier: Apache-2.0
"""A run survives the death of its worker: kill -9 the worker process in the
middle of a job; its job process dies with it, the lease runs out, and a new
worker finishes the run (attempt 2) with one set of outputs."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from relsynth.service.store import Store

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="process-group and death-signal handling is Linux")


def _env(settings) -> dict:
    env = dict(os.environ)
    env.update({"RELSYNTH_SVC_DB_URL": settings.db_url, "RELSYNTH_SVC_DATA_DIR": settings.data_dir,
                "RELSYNTH_SVC_AUTH_MODE": "dev", "RELSYNTH_SVC_ALLOW_DEV_AUTH": "true",
                "RELSYNTH_SVC_MASTER_KEY_REF": settings.master_key_ref, "RELSYNTH_SVC_FILE_ROOTS": settings.file_roots,
                "RELSYNTH_SVC_LEASE_S": "4", "RELSYNTH_SVC_HEARTBEAT_S": "1", "RELSYNTH_SVC_LOG_JSON": "true"})
    return env


def test_run_survives_worker_kill(settings, store: Store, shop):
    import relsynth as rs
    spec = rs.introspect({"kind": "sqlite", "path": str(shop)})
    import yaml
    raw = dict(spec.raw)
    raw.pop("source")
    raw.pop("target", None)
    p = store.add_project("acme", "p", "", "user:erin")
    c = store.add_connection("acme", p["id"], "shop", {"kind": "sqlite", "path": str(shop)}, None, {}, "user:erin")
    s = store.add_spec("acme", p["id"], "shop", yaml.safe_dump(raw), [], c["id"], "user:erin")
    r = store.add_run("acme", p["id"], s["id"], c["id"], "run", {"scale": 4}, "internal", 3, "user:erin")

    cmd = [sys.executable, "-m", "relsynth.service.cli", "worker", "--poll", "0.2"]
    w1 = subprocess.Popen(cmd, env=_env(settings), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while store.run("acme", r["id"])["state"] != "running" and time.time() < deadline:
            time.sleep(0.1)
        assert store.run("acme", r["id"])["state"] == "running"
        time.sleep(1.5)                          # the job process is fitting now
        w1.send_signal(signal.SIGKILL)
        w1.wait(10)
    finally:
        if w1.poll() is None:
            w1.kill()
    time.sleep(0.5)
    out = subprocess.run(["pgrep", "-f", "-a", r"^\S*python\S* -m relsynth\.service\.runner"], capture_output=True, text=True)
    assert out.stdout.strip() == "", "the job process outlived its worker"
    assert store.run("acme", r["id"])["state"] == "running"     # still leased by the dead worker
    w2 = subprocess.run(cmd[:-2] + ["--once", "--poll", "0.2"], env=_env(settings), capture_output=True, text=True,
                        timeout=240)
    final = store.run("acme", r["id"])
    if final["state"] == "running":              # w2 started before the lease ran out; run it again
        time.sleep(4)
        subprocess.run(cmd[:-2] + ["--once"], env=_env(settings), capture_output=True, timeout=240)
        final = store.run("acme", r["id"])
    assert final["state"] == "succeeded", (final, w2.stderr[-2000:])
    assert final["attempts"] == 2
    names = [a["name"] for a in store.artifacts("acme", r["id"])]
    assert sorted(names) == sorted(set(names)) and "customer.csv" in names
    acts = [e["action"] for e in store.audit_log("acme")]
    assert acts.count("run.started") == 2 and acts.count("run.succeeded") == 1
