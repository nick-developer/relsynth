# SPDX-License-Identifier: Apache-2.0
"""Kill the worker that runs a job; another worker must finish it.

Run after e2e_flow.py (it reuses its tenant, project and spec).  With
``--compose`` the worker container is killed with ``docker kill``; with
``--k8s-namespace`` the worker pod is deleted at once (``--grace-period=0
--force``), as a node loss would, and its Deployment starts a new one."""
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
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--compose", help="docker compose file of the stack")
    where.add_argument("--k8s-namespace", help="Kubernetes namespace of the Helm release")
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
    # a worker's lease owner is "<hostname>-<pid>-<hex>"; the hostname is the
    # container id (compose) or the pod name (Kubernetes)
    if a.compose:
        ids = subprocess.run(["docker", "compose", "-f", a.compose, "ps", "-q", "worker"],  # noqa: S603, S607  # nosec B603 B607
                             capture_output=True, text=True, check=True).stdout.split()
        victim = next((i for i in ids if i.startswith(owner.split("-")[0])), None)
        check("the leasing worker container is found", victim is not None, f"{owner} not in {ids}")
        subprocess.run(["docker", "kill", victim], check=True, capture_output=True)  # noqa: S603, S607  # nosec B603 B607
    else:
        pods = subprocess.run(["kubectl", "-n", a.k8s_namespace, "get", "pods",  # noqa: S603, S607  # nosec B603 B607
                               "-l", "app.kubernetes.io/component=worker", "-o", "name"],
                              capture_output=True, text=True, check=True).stdout.split()
        victim = next((p for p in pods if owner.startswith(p.split("/", 1)[1] + "-")), None)
        check("the leasing worker pod is found", victim is not None, f"{owner} not in {pods}")
        subprocess.run(["kubectl", "-n", a.k8s_namespace, "delete", victim,  # noqa: S603, S607  # nosec B603 B607
                        "--grace-period=0", "--force", "--wait=false"], check=True, capture_output=True)
    check("worker killed mid-run", True, victim)
    t0 = time.time()
    while time.time() - t0 < 900:
        _, r = erin.call("GET", f"{T}/runs/{run['id']}")
        if r["state"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(2)
    check("the other worker finished the run", r["state"] == "succeeded" and r["attempts"] == 2,
          f"{r['state']} attempts={r['attempts']} {r.get('error')}")
    if a.compose:
        subprocess.run(["docker", "compose", "-f", a.compose, "up", "-d", "worker"], check=True,  # noqa: S603, S607  # nosec B603 B607
                       capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
