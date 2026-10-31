# SPDX-License-Identifier: Apache-2.0
"""The REST API of the relsynth service (OpenAPI at ``/api/v1/openapi.json``).

Every tenant resource lives under ``/api/v1/t/{tenant}/...``.  A caller without
a role in that tenant gets 404 for every id in it, the same answer as for an id
that does not exist, so ids cannot be probed across tenants.  Every change and
every refusal is written to the tenant's hash-chained audit log.
"""
from __future__ import annotations

import io
import json
import logging
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .._version import __version__
from . import release
from . import secrets as sec
from .artifacts import Artifacts
from .auth import PERMISSIONS, TENANT_RE, Authenticator, AuthError, Forbidden, Principal
from .config import Settings
from .crypto import KeyWrapper
from .policy import SourceRefused, check_source
from .store import Store
from .telemetry import METRICS, setup_logging, setup_tracing, span

log = logging.getLogger("relsynth.service.api")
UI_DIR = Path(__file__).parent / "ui"


# ------------------------------------------------------------------ bodies
class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=4000)


class ConnectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source: dict[str, Any] = Field(description="a relsynth source section without credentials")
    secret_ref: str | None = Field(None, max_length=500, description="env:/file:/vault:/aws-sm:/azure-kv:/gcp-sm:")
    secret_is_url: bool = Field(False, description="the secret is the whole URL (url_env), not only the password")
    labels: dict[str, str] = Field(default_factory=dict, description="ABAC attributes, e.g. classification")


class IntrospectIn(BaseModel):
    name: str = Field("draft", min_length=1, max_length=200)
    sample_rows: int = Field(20000, ge=100, le=5_000_000)
    include: list[str] | None = None
    exclude: list[str] | None = None


class SpecIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4_000_000, description="the spec YAML")
    connection_id: str | None = None


class RunIn(BaseModel):
    spec_id: str
    connection_id: str | None = Field(None, description="default: the spec's connection")
    scale: float | None = Field(None, gt=0, le=1000)
    seed: int | None = None
    datasets: int | None = Field(None, ge=1, le=20)
    format: Literal["csv", "parquet"] = "csv"
    quality: bool = True
    fail_under: float | None = Field(None, ge=0, le=1)
    context: Literal["internal", "external"] = "internal"


class ApprovalIn(BaseModel):
    decision: Literal["approve", "reject", "revoke"]
    reason: str = Field("", max_length=4000)


class TenantIn(BaseModel):
    id: str = Field(pattern=TENANT_RE.pattern)
    name: str = Field(min_length=1, max_length=200)


class ServiceAccountIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    roles: list[Literal["data_owner", "privacy_officer", "engineer", "consumer"]]


# ------------------------------------------------------------------ app
def create_app(settings: Settings | None = None, store: Store | None = None) -> FastAPI:
    s = settings or Settings.load()
    errs = s.check()
    if errs:
        raise SystemExit("refusing to start: " + "; ".join(errs))
    setup_logging(s.log_json)
    setup_tracing(s.otel)
    st = store or Store(s.db_url).create()
    wrapper = KeyWrapper(s.kms, s.master_key_ref, s.kms_key_id)
    arts = Artifacts(s.data_dir, st, wrapper)
    authn = Authenticator(s, st)

    app = FastAPI(title="relsynth service", version=__version__, openapi_url="/api/v1/openapi.json",
                  docs_url="/api/v1/docs", redoc_url=None,
                  description="Synthetic relational data as a service: projects, connections, specs, runs, "
                              "reports, approvals. See docs/service/README.md.")
    app.state.settings, app.state.store, app.state.artifacts = s, st, arts

    # middleware -----------------------------------------------------------
    @app.middleware("http")
    async def guard(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        if not _safe_id(rid):
            rid = uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > s.max_body_mb << 20:
            resp = JSONResponse({"detail": "request body too large"}, status_code=413)
        else:
            try:
                with span(f"{request.method} {request.url.path.split('/t/')[0]}"):
                    resp = await call_next(request)
            except Exception:  # noqa: BLE001  never leak a stack trace to the client
                log.exception("unhandled error", extra={"request_id": rid})
                resp = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        METRICS.inc("relsynth_http_requests_total", method=request.method, route=path, status=resp.status_code)
        METRICS.observe("relsynth_http_request_seconds", ms / 1000, route=path)
        resp.headers["X-Request-Id"] = rid
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = resp.headers.get("Cache-Control", "no-store")
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            f"connect-src 'self' {s.oidc_issuer or ''}; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        log.info("request", extra={"request_id": rid, "path": path, "status": resp.status_code, "ms": round(ms, 1)})
        return resp

    @app.exception_handler(AuthError)
    async def _authn(req, e: AuthError):
        log.warning("authentication failed", extra={"request_id": getattr(req.state, "request_id", None),
                                                    "outcome": str(e)[:200]})
        return JSONResponse({"detail": f"authentication failed: {e}"}, status_code=401,
                            headers={"WWW-Authenticate": 'Bearer realm="relsynth"'})

    @app.exception_handler(Forbidden)
    async def _forbid(_req, e: Forbidden):
        return JSONResponse({"detail": str(e)}, status_code=403)

    # dependencies -----------------------------------------------------------
    def principal(request: Request, authorization: str | None = Header(None),
                  x_relsynth_dev_user: str | None = Header(None)) -> Principal:
        p = authn.authenticate(authorization, x_relsynth_dev_user)
        request.state.principal = p
        return p

    def tenant_of(tenant: str, request: Request, p: Principal = Depends(principal)) -> str:
        if not TENANT_RE.match(tenant) or tenant not in p.roles:
            log.info("tenant refused", extra={"request_id": request.state.request_id, "actor": p.id,
                                               "tenant": tenant if TENANT_RE.match(tenant) else "?"})
            raise HTTPException(404, "not found")
        return tenant

    def need(p: Principal, tenant: str, perm: str, target: str = "") -> None:
        if not p.has(tenant, perm):
            st.append_audit(tenant, p.id, "access.denied", target, "denied", {"permission": perm})
            raise Forbidden(f"{perm} needs one of the roles {sorted(PERMISSIONS[perm])}")

    def audit(p: Principal, tenant: str, action: str, target: str = "", outcome: str = "ok", **detail) -> None:
        st.append_audit(tenant, p.id, action, target, outcome, detail)
        log.info(action, extra={"tenant": tenant, "actor": p.id, "action": action, "target": target,
                                "outcome": outcome})

    def get_or_404(row, what="not found"):
        if row is None:
            raise HTTPException(404, what)
        return row

    def connection_for(p: Principal, tenant: str, cid: str) -> dict:
        c = get_or_404(st.connection(tenant, cid))
        if s.abac and not p.cleared_for(c.get("labels")):
            st.append_audit(tenant, p.id, "access.denied", cid, "denied", {"abac": c["labels"].get("classification")})
            raise Forbidden("your clearance is below this connection's classification")
        return c

    # health and metadata ---------------------------------------------------
    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz():
        try:
            st.ping()
            arts.root.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"status": "unavailable", "detail": type(e).__name__}, status_code=503)
        return {"status": "ready", "version": __version__}

    @app.get("/metrics", include_in_schema=False, response_class=PlainTextResponse)
    def metrics():
        return METRICS.render()

    @app.get("/api/v1/ui-config", tags=["meta"])
    def ui_config():
        """What the browser needs to start single sign-on (public)."""
        return {"issuer": s.oidc_issuer, "client_id": s.ui_client_id, "auth_mode": s.auth_mode,
                "version": __version__}

    @app.get("/api/v1/me", tags=["meta"])
    def me(p: Principal = Depends(principal)):
        return {"subject": p.subject, "name": p.name, "kind": p.kind, "platform_admin": p.platform_admin,
                "tenants": {t: sorted(r) for t, r in p.roles.items()}}

    # tenants (platform administration: no access to tenant data) --------------
    @app.get("/api/v1/tenants", tags=["tenants"])
    def list_tenants(p: Principal = Depends(principal)):
        rows = st.list_tenants()
        return rows if p.platform_admin else [r for r in rows if r["id"] in p.roles]

    @app.post("/api/v1/tenants", tags=["tenants"], status_code=201)
    def create_tenant(body: TenantIn, p: Principal = Depends(principal)):
        if not p.platform_admin:
            raise Forbidden("creating tenants needs the platform_admin role")
        st.ensure_tenant(body.id, body.name)
        audit(p, body.id, "tenant.created", body.id)
        return {"id": body.id, "name": body.name}

    # projects -----------------------------------------------------------------
    T = "/api/v1/t/{tenant}"

    @app.get(T + "/projects", tags=["projects"])
    def list_projects(tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "project:read")
        return st.projects(tenant)

    @app.post(T + "/projects", tags=["projects"], status_code=201)
    def create_project(body: ProjectIn, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "project:write")
        st.ensure_tenant(tenant)
        row = st.add_project(tenant, body.name, body.description, p.id)
        audit(p, tenant, "project.created", row["id"], name=body.name)
        return row

    @app.get(T + "/projects/{pid}", tags=["projects"])
    def get_project(pid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "project:read")
        return get_or_404(st.project(tenant, pid))

    # connections ----------------------------------------------------------------
    @app.get(T + "/projects/{pid}/connections", tags=["connections"])
    def list_connections(pid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "connection:read")
        get_or_404(st.project(tenant, pid))
        rows = st.connections(tenant, pid)
        return [r for r in rows if not s.abac or p.cleared_for(r.get("labels"))]

    @app.post(T + "/projects/{pid}/connections", tags=["connections"], status_code=201)
    def create_connection(pid: str, body: ConnectionIn, tenant: str = Depends(tenant_of),
                          p: Principal = Depends(principal)):
        need(p, tenant, "connection:write")
        get_or_404(st.project(tenant, pid), "project not found")
        try:
            check_source(body.source, tenant, s)
            if body.secret_ref:
                sec.check_ref(body.secret_ref, tenant)
        except (SourceRefused, sec.SecretError) as e:
            audit(p, tenant, "connection.refused", pid, "denied", reason=str(e))
            raise HTTPException(422, str(e)) from e
        labels = dict(body.labels)
        if body.secret_is_url:
            labels["secret_is_url"] = "true"  # noqa: S105  # nosec B105 (a flag, not a password)
        if s.abac and not p.cleared_for(labels):
            raise Forbidden("you cannot create a connection above your own clearance")
        row = st.add_connection(tenant, pid, body.name, body.source, body.secret_ref, labels, p.id)
        audit(p, tenant, "connection.created", row["id"], kind=body.source.get("kind"))
        return row

    @app.get(T + "/connections/{cid}", tags=["connections"])
    def get_connection(cid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "connection:read")
        return connection_for(p, tenant, cid)

    @app.post(T + "/connections/{cid}/introspect", tags=["connections"], status_code=202)
    def introspect(cid: str, body: IntrospectIn, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        """Queue a job that drafts a spec from the connection; the draft and its review
        questions appear as a new spec version when the job succeeds."""
        need(p, tenant, "spec:write")
        c = connection_for(p, tenant, cid)
        row = st.add_run(tenant, c["project_id"], "", cid, "introspect", body.model_dump(), "internal",
                         s.max_attempts, p.id)
        audit(p, tenant, "introspect.queued", row["id"], connection=cid)
        return row

    # specs ------------------------------------------------------------------------
    @app.get(T + "/projects/{pid}/specs", tags=["specs"])
    def list_specs(pid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "spec:read")
        get_or_404(st.project(tenant, pid))
        return [{k: v for k, v in r.items() if k != "body"} for r in st.specs(tenant, pid)]

    @app.post(T + "/projects/{pid}/specs", tags=["specs"], status_code=201)
    def create_spec(pid: str, body: SpecIn, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "spec:write")
        get_or_404(st.project(tenant, pid), "project not found")
        if body.connection_id:
            connection_for(p, tenant, body.connection_id)
        try:
            raw = yaml.safe_load(body.body)
        except yaml.YAMLError as e:
            raise HTTPException(422, f"not valid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise HTTPException(422, "a spec is a YAML mapping")
        if "source" in raw or "target" in raw:
            raise HTTPException(422, "remove source and target: the service takes them from the connection "
                                     "and its own storage")
        row = st.add_spec(tenant, pid, body.name, body.body, list(raw.get("review") or []), body.connection_id, p.id)
        audit(p, tenant, "spec.saved", row["id"], name=body.name, version=row["version"])
        return row | {"errors": _spec_errors(raw)}

    @app.get(T + "/specs/{sid}", tags=["specs"])
    def get_spec(sid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "spec:read")
        row = get_or_404(st.spec(tenant, sid))
        return row | {"errors": _spec_errors(yaml.safe_load(row["body"]) or {})}

    # runs ---------------------------------------------------------------------------
    @app.post(T + "/runs", tags=["runs"], status_code=202)
    def create_run(body: RunIn, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "run:create")
        spec = get_or_404(st.spec(tenant, body.spec_id), "spec not found")
        cid = body.connection_id or spec["connection_id"]
        if not cid:
            raise HTTPException(422, "the spec has no connection; pass connection_id")
        connection_for(p, tenant, cid)
        why = release.can_run_in(body.context)
        if why:
            audit(p, tenant, "run.refused", body.spec_id, "denied", reason=why)
            raise HTTPException(422, why)
        params = body.model_dump(exclude={"spec_id", "connection_id", "context"})
        row = st.add_run(tenant, spec["project_id"], spec["id"], cid, "run", params, body.context, s.max_attempts, p.id)
        audit(p, tenant, "run.queued", row["id"], spec=spec["id"], connection=cid, context=body.context)
        return row

    @app.get(T + "/runs", tags=["runs"])
    def list_runs(tenant: str = Depends(tenant_of), p: Principal = Depends(principal),
                  project_id: str | None = None, state: str | None = None, limit: int = Query(100, le=500)):
        need(p, tenant, "run:read")
        return st.runs(tenant, project_id, state, limit)

    @app.get(T + "/runs/{rid}", tags=["runs"])
    def get_run(rid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "run:read")
        r = get_or_404(st.run(tenant, rid))
        appr = st.approvals(tenant, rid)
        d = release.decide(r, appr, s)
        return r | {"release": {"allow": d.allow, "approval_required": d.approval_required, "reasons": d.reasons},
                    "approvals": appr}

    @app.post(T + "/runs/{rid}/cancel", tags=["runs"])
    def cancel_run(rid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "run:cancel", rid)
        r = get_or_404(st.request_cancel(tenant, rid))
        audit(p, tenant, "run.cancel_requested", rid, state=r["state"])
        return r

    @app.get(T + "/runs/{rid}/artifacts", tags=["runs"])
    def list_artifacts(rid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "run:read")
        get_or_404(st.run(tenant, rid))
        return [{k: v for k, v in a.items() if k not in ("wrapped_key", "path")} for a in st.artifacts(tenant, rid)]

    def _report(tenant: str, rid: str, kind: str) -> dict:
        get_or_404(st.run(tenant, rid))
        arts_ = [a for a in st.artifacts(tenant, rid) if a["kind"] == kind]
        if not arts_:
            raise HTTPException(404, f"the run has no {kind}")
        return json.loads(arts.read_bytes(tenant, arts_[0]))

    @app.get(T + "/runs/{rid}/report", tags=["reports"])
    def get_report(rid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal),
                   kind: Literal["report", "quality"] = "report"):
        """The scorecard (``report``) or the calibrated quality report 2.0 (``quality``).
        The privacy items of the scorecard are shown only to data owners and privacy officers."""
        need(p, tenant, "report:read")
        rep = _report(tenant, rid, kind)
        if not p.has(tenant, "privacy:read"):
            rep = _strip_privacy(rep)
        audit(p, tenant, "report.read", rid, kind=kind)
        return rep

    @app.get(T + "/compare", tags=["reports"])
    def compare(a: str, b: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        """Side by side summary of two runs' scorecards."""
        need(p, tenant, "report:read")
        ra, rb = _report(tenant, a, "report"), _report(tenant, b, "report")
        return {"a": _summary(ra), "b": _summary(rb), "changed": _diff(ra, rb)}

    # approvals and downloads -----------------------------------------------------
    @app.post(T + "/runs/{rid}/approvals", tags=["approvals"], status_code=201)
    def decide(rid: str, body: ApprovalIn, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "approval:decide", rid)
        r = get_or_404(st.run(tenant, rid))
        if r["state"] != "succeeded" or r["kind"] != "run":
            raise HTTPException(409, f"the run is {r['state']}; only succeeded synthesis runs can be approved")
        if body.decision == "approve" and s.separation_of_duties and r["created_by"] == p.id:
            audit(p, tenant, "approval.refused", rid, "denied", reason="separation of duties")
            raise Forbidden("separation of duties: you started this run, so someone else must approve it")
        if body.decision != "approve" and not body.reason.strip():
            raise HTTPException(422, "a rejection or revocation needs a reason")
        a = st.add_approval(tenant, rid, body.decision, p.id, p.role_for(tenant, "approval:decide"), body.reason)
        new_state = release.release_state(r, st.approvals(tenant, rid), s)
        st.update_run(tenant, rid, release_state=new_state)
        audit(p, tenant, f"approval.{body.decision}", rid, release_state=new_state, reason=body.reason[:300])
        return a | {"release_state": new_state}

    @app.get(T + "/runs/{rid}/approvals", tags=["approvals"])
    def list_approvals(rid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "run:read")
        get_or_404(st.run(tenant, rid))
        return st.approvals(tenant, rid)

    def _released(p: Principal, tenant: str, rid: str) -> dict:
        need(p, tenant, "data:download", rid)
        r = get_or_404(st.run(tenant, rid))
        d = release.decide(r, st.approvals(tenant, rid), s)
        if not d.allow:
            audit(p, tenant, "download.refused", rid, "denied", reasons=d.reasons)
            raise HTTPException(409, {"detail": "not released", "reasons": d.reasons})
        return r

    @app.get(T + "/runs/{rid}/download", tags=["approvals"])
    def download(rid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        """The synthetic tables of an approved run, as one zip."""
        _released(p, tenant, rid)
        data = [a for a in st.artifacts(tenant, rid) if a["kind"] == "data"]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for a in data:
                with z.open(a["name"], "w") as f:
                    for chunk in arts.read(tenant, a):
                        f.write(chunk)
            z.writestr("MANIFEST.json", json.dumps({"run": rid, "tenant": tenant, "files": [
                {"name": a["name"], "sha256": a["sha256"], "size": a["size"]} for a in data]}, indent=1))
        audit(p, tenant, "data.downloaded", rid, files=len(data), bytes=sum(a["size"] or 0 for a in data))
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{rid}.zip"'})

    @app.get(T + "/artifacts/{aid}", tags=["approvals"])
    def download_artifact(aid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        a = get_or_404(st.artifact(tenant, aid))
        if a["kind"] == "data":
            _released(p, tenant, a["run_id"])
        else:
            need(p, tenant, "report:read", aid)
            if a["kind"] == "report" and not p.has(tenant, "privacy:read"):
                raise Forbidden("the full scorecard includes privacy items; use /runs/{id}/report")
        audit(p, tenant, "artifact.downloaded", aid, kind=a["kind"])
        return StreamingResponse(arts.read(tenant, a), media_type="application/octet-stream",
                                 headers={"Content-Disposition": f'attachment; filename="{Path(a["name"]).name}"'})

    # audit ------------------------------------------------------------------------
    @app.get(T + "/audit", tags=["audit"])
    def audit_log(tenant: str = Depends(tenant_of), p: Principal = Depends(principal), after: int = 0,
                  limit: int = Query(500, le=5000)):
        need(p, tenant, "audit:read")
        return st.audit_log(tenant, limit, after)

    @app.get(T + "/audit/verify", tags=["audit"])
    def audit_verify(tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "audit:read")
        return st.verify_audit(tenant)

    # service accounts ---------------------------------------------------------------
    @app.get(T + "/service-accounts", tags=["service accounts"])
    def list_sa(tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "sa:manage")
        return st.service_accounts(tenant)

    @app.post(T + "/service-accounts", tags=["service accounts"], status_code=201)
    def create_sa(body: ServiceAccountIn, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        """Returns the token once; only its hash is stored."""
        need(p, tenant, "sa:manage")
        row, token = st.add_service_account(tenant, body.name, body.roles, p.id)
        audit(p, tenant, "service_account.created", row["id"], roles=body.roles)
        return row | {"token": token}

    @app.delete(T + "/service-accounts/{sid}", tags=["service accounts"])
    def revoke_sa(sid: str, tenant: str = Depends(tenant_of), p: Principal = Depends(principal)):
        need(p, tenant, "sa:manage")
        if not st.revoke_service_account(tenant, sid):
            raise HTTPException(404, "not found")
        audit(p, tenant, "service_account.revoked", sid)
        return {"revoked": sid}

    # web UI ---------------------------------------------------------------------------
    if UI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/ui/")

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon():
            return FileResponse(UI_DIR / "favicon.svg", media_type="image/svg+xml")

    return app


# ------------------------------------------------------------------ helpers
def _safe_id(v: str) -> bool:
    return 0 < len(v) <= 64 and all(c.isalnum() or c in "-_." for c in v)


def _spec_errors(raw: dict) -> list[str]:
    try:
        from ..api import Spec
        r = dict(raw)
        r.setdefault("source", {"kind": "csv", "path": "."})
        return Spec.from_dict(r).errors()
    except Exception as e:  # noqa: BLE001
        return [str(e)]


PRIVACY_KEYS = ("privacy",)
PRIVACY_METRICS = ("dcr", "nndr", "membership", "inference", "singling", "linkab", "copied", "exact_match", "leak")


def _strip_privacy(rep: Any) -> Any:
    """The scorecard without its privacy section and privacy items, for readers
    without privacy:read (the privacy assessment names the records most at risk)."""
    if not isinstance(rep, dict):
        return rep
    out = {k: v for k, v in rep.items() if k not in PRIVACY_KEYS}
    sc = out.get("scorecard")
    if isinstance(sc, dict):
        keep = lambda i: not any(w in str(i.get("metric", "")).lower() for w in PRIVACY_METRICS)
        items = [i for i in sc.get("items", []) if keep(i)]
        out["scorecard"] = {**sc, "items": items, "failed": [i for i in sc.get("failed", []) if keep(i)],
                            "total": len(items), "passed": sum(1 for i in items if i.get("pass"))}
    return out


def _items(rep: dict) -> dict[str, dict]:
    sc = rep.get("scorecard") or {}
    return {f"{i.get('area')}|{i.get('metric')}": i for i in sc.get("items", [])}


def _summary(rep: dict) -> dict:
    sc = rep.get("scorecard") or {}
    total = int(sc.get("total", 0))
    return {"passed": int(sc.get("passed", 0)), "total": total,
            "score": int(sc.get("passed", 0)) / total if total else 1.0}


def _diff(a: dict, b: dict) -> list[dict]:
    """Checks whose verdict or value differs between two scorecards."""
    ia, ib = _items(a), _items(b)
    out = []
    for k in sorted(set(ia) | set(ib)):
        x, y = ia.get(k, {}), ib.get(k, {})
        if x.get("pass") != y.get("pass") or x.get("value") != y.get("value"):
            area, metric = k.split("|", 1)
            out.append({"area": area, "metric": metric, "a": x.get("value"), "b": y.get("value"),
                        "a_pass": x.get("pass"), "b_pass": y.get("pass"), "limit": y.get("limit", x.get("limit"))})
    return out


def __getattr__(name: str):            # `uvicorn relsynth.service.app:app` builds the app from the environment
    if name == "app":
        return create_app()
    raise AttributeError(name)
