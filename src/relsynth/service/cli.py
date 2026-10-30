# SPDX-License-Identifier: Apache-2.0
"""``relsynth-service``: run and administer the service.

    relsynth-service serve [--host 0.0.0.0 --port 8080]   API and web UI
    relsynth-service worker [--once]                     job worker (scale out with more processes or pods)
    relsynth-service migrate                             create or upgrade the metadata tables
    relsynth-service tenant add ID NAME                  create a tenant (bootstrap; later via the API)
    relsynth-service audit verify TENANT                 recompute a tenant's audit hash chain
    relsynth-service openapi [--out FILE]                write the OpenAPI document
    relsynth-service check                               validate the settings and reach the store

Settings come from RELSYNTH_SVC_* variables (relsynth.service.config).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="relsynth-service", description="relsynth service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="API and web UI")
    sv.add_argument("--host", default="0.0.0.0")  # noqa: S104  # nosec B104 (containers listen on every interface)
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--workers", type=int, default=1, help="HTTP worker processes")
    sv.add_argument("--proxy-headers", action="store_true", help="trust X-Forwarded-* from the ingress")
    wk = sub.add_parser("worker", help="job worker")
    wk.add_argument("--once", action="store_true", help="drain the queue, then exit")
    wk.add_argument("--poll", type=float, default=2.0)
    sub.add_parser("migrate", help="create or upgrade the metadata tables")
    tn = sub.add_parser("tenant", help="tenants")
    tn.add_argument("action", choices=["add", "list"])
    tn.add_argument("id", nargs="?")
    tn.add_argument("name", nargs="?")
    au = sub.add_parser("audit", help="audit log")
    au.add_argument("action", choices=["verify", "export"])
    au.add_argument("tenant")
    op = sub.add_parser("openapi", help="write the OpenAPI document")
    op.add_argument("--out")
    sub.add_parser("check", help="validate settings and reach the store")
    a = ap.parse_args(argv)

    from .config import Settings
    from .store import Store
    s = Settings.load()

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("relsynth.service.app:app", host=a.host, port=a.port, workers=a.workers,
                    proxy_headers=a.proxy_headers, forwarded_allow_ips="*" if a.proxy_headers else None,
                    server_header=False, log_config=None)
        return 0
    if a.cmd == "openapi":
        import os
        os.environ.setdefault("RELSYNTH_SVC_AUTH_MODE", "dev")
        os.environ.setdefault("RELSYNTH_SVC_ALLOW_DEV_AUTH", "true")
        os.environ.setdefault("RELSYNTH_SVC_DB_URL", "sqlite://")
        from .app import create_app
        doc = json.dumps(create_app(Settings.load()).openapi(), indent=1, sort_keys=True)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(doc + "\n")
        else:
            print(doc)
        return 0
    errs = s.check()
    if errs:
        print("settings: " + "; ".join(errs), file=sys.stderr)
        return 2
    st = Store(s.db_url).create()
    if a.cmd in ("migrate", "check"):
        st.ping()
        print("ok")
        return 0
    if a.cmd == "tenant":
        if a.action == "list":
            for t in st.list_tenants():
                print(t["id"], t["name"])
            return 0
        if not a.id:
            print("tenant add needs ID and NAME", file=sys.stderr)
            return 2
        st.ensure_tenant(a.id, a.name or a.id)
        st.append_audit(a.id, "cli:operator", "tenant.created", a.id)
        print(a.id)
        return 0
    if a.cmd == "audit":
        if a.action == "verify":
            r = st.verify_audit(a.tenant)
            print(json.dumps(r))
            return 0 if r["ok"] else 1
        for row in st.audit_log(a.tenant, limit=10**9):
            print(json.dumps(row, default=str))
        return 0
    if a.cmd == "worker":
        from .artifacts import Artifacts
        from .crypto import KeyWrapper
        from .telemetry import setup_logging, setup_tracing
        from .worker import Worker, drain_handler
        setup_logging(s.log_json)
        setup_tracing(s.otel, "relsynth-worker")
        w = Worker(s, st, Artifacts(s.data_dir, st, KeyWrapper(s.kms, s.master_key_ref, s.kms_key_id)))
        signal.signal(signal.SIGTERM, drain_handler(w))
        signal.signal(signal.SIGINT, drain_handler(w))
        n = w.serve(poll_s=a.poll, once=a.once)
        print(f"{n} job(s) processed", file=sys.stderr)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
