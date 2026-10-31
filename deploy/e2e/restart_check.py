# SPDX-License-Identifier: Apache-2.0
"""Kill the worker container that runs a job; another worker must finish it.

Run after e2e_flow.py (it reuses its tenant, project and spec)."""
from __future__ import annotations

import argparse
import subprocess  # nosec B404 (docker CLI with fixed arguments)
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))
from e2e_flow import Client, check, login  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--token-url", default="http://localhost:8081/realms/relsynth/protocol/openid-connect/token")
    ap.add_argument("--compose", required=True)
    a = ap.parse_args()
    erin = Client(a.api, login(a.token_url, "relsynth-e2e", "erin", "erin-Demo-Passw0rd"))
    T = "/api/v1/t/acme"
    _, runs = erin.call("GET", f"{T}/runs?state=succeeded")
    spec_id = next(r["spec_id"] for r in runs if r["kind"] == "run")
    _, run = erin.call("POST", f"{T}/runs", {"spec_id": spec_id, "scale": 30, "seed": 1})
    owner = None
    for _ in range(300):
        _, r = erin.call("GET", f"{T}/runs/{run['id']}")
        if r["state"] == "running":
            owner = r["lease_owner"]
            break
        time.sleep(0.5)
    check("big run is leased by a worker", owner is not None)
    time.sleep(3)
    host = owner.split("-")[0]
    ids = subprocess.run(["docker", "compose", "-f", a.compose, "ps", "-q", "worker"],  # noqa: S603, S607  # nosec B603 B607
                         capture_output=True, text=True, check=True).stdout.split()
    victim = next(i for i in ids if i.startswith(host))
    subprocess.run(["docker", "kill", victim], check=True, capture_output=True)  # noqa: S603, S607  # nosec B603 B607
    check("worker container killed mid-run", True)
    t0 = time.time()
    while time.time() - t0 < 900:
        _, r = erin.call("GET", f"{T}/runs/{run['id']}")
        if r["state"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(2)
    check("the other worker finished the run", r["state"] == "succeeded" and r["attempts"] == 2,
          f"{r['state']} attempts={r['attempts']} {r.get('error')}")
    subprocess.run(["docker", "compose", "-f", a.compose, "up", "-d", "worker"], check=True,  # noqa: S603, S607  # nosec B603 B607
                   capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
