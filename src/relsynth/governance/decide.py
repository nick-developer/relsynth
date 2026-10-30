# SPDX-License-Identifier: Apache-2.0
"""Release decisions: the policy rules, the invariants in code, and OPA (design sec. 8).

`decide(input)` evaluates relsynth's release rules (`rego/release.rego`,
query `data.relsynth.release.decision`) and returns ``{"allow",
"approval_required", "reasons"}``.  Two engines evaluate them:

- ``builtin``: a line-by-line Python port of release.rego, so that an
  on-premises or air-gapped install needs no extra binary.  The CI job
  ``policy`` runs ``opa test`` on the Rego and checks that both engines give the
  same decision on every fixture (tests/unit/governance/test_policy_parity.py).
- ``opa``: Open Policy Agent (`opa eval`), needed when the organisation adds its
  own Rego modules (``rego:`` in the policy).  Any OPA error means deny.

Whatever the engine says, `invariants()` is applied on top (sec. 8.3): policy
can tighten these rules, never loosen them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 - runs the OPA binary the operator installed, with fixed arguments
from importlib import resources
from pathlib import Path

QUERY = "data.relsynth.release.decision"


def _q(s) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def _v(x) -> str:
    """A value as OPA's sprintf %v prints a JSON input value: integral numbers as
    integers, other numbers as Go's shortest %v form, composites as Rego terms."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e21:
            return str(int(x))
        r = repr(x)
        if "e" in r:
            mant, ex = r.split("e")
            e = int(ex)
            if 16 <= e < 21:                          # Go switches to exponents only from 1e21
                return format(x, "f").rstrip("0").rstrip(".")
            return f"{mant}e{'-' if e < 0 else '+'}{abs(e):02d}"
        return r
    if isinstance(x, str):
        return x
    if x is None:
        return "null"
    return json.dumps(x, separators=(", ", ": "), ensure_ascii=False)


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def builtin(inp: dict) -> dict:
    """relsynth's release rules (a port of rego/release.rego)."""
    pol, m, led, req = inp.get("policy") or {}, inp.get("manifest") or {}, inp.get("ledger") or {}, inp.get("request") or {}
    cname = req.get("context")
    ctx = _get(pol, "contexts", cname)
    external = True if ctx is None else bool(ctx.get("external"))
    mode = _get(m, "privacy", "mode")
    deny = set()
    if ctx is None:
        deny.add(f"context {_q(cname)} is not defined in the organisation policy")
    if external and mode is not None and mode != "dp":
        deny.add(f"context {_q(cname)} is outside the organisation: differential privacy is required, and this run has none")
    if mode == "dp":
        dom = _get(m, "source", "domain")
        if "domain" in led and dom is not None and dom != led["domain"]:
            deny.add(f"the run's domain {_q(dom)} does not match the ledger's {_q(led.get('domain'))}")
        unit = _get(pol, "domains", dom, "unit", "entity")
        if unit is not None and _get(m, "privacy", "unit") != unit:
            deny.add(f"the run protects {_q(_get(m, 'privacy', 'unit'))} but the policy's privacy unit for domain "
                     f"{_q(dom)} is {_q(unit)}")
        if led.get("run_rho", 0) > led.get("per_release_rho_cap", float("inf")):
            deny.add(f"this release spends rho {_v(led['run_rho'])}, above the per-release cap "
                     f"{_v(led['per_release_rho_cap'])} (epsilon {_v(led.get('per_release_epsilon_cap'))} at delta "
                     f"{_v(led.get('delta'))})")
        if led.get("domain_rho_after", 0) > led.get("lifetime_rho_cap", float("inf")):
            deny.add(f"domain {_q(led.get('domain'))} would reach rho {_v(led['domain_rho_after'])}, above its lifetime "
                     f"cap {_v(led['lifetime_rho_cap'])} (epsilon {_v(led.get('lifetime_epsilon_cap'))} at delta "
                     f"{_v(led.get('delta'))})")
        if led.get("period_rho_cap") and led.get("period_rho_after", 0) > led["period_rho_cap"]:
            deny.add(f"domain {_q(led.get('domain'))} would spend rho {_v(led['period_rho_after'])} this period, above "
                     f"the period cap {_v(led['period_rho_cap'])}")
    if external and _get(m, "spec", "public_facts_only") is not True:
        deny.add("an external DP release must use only public schema facts or charged statistics "
                 "(spec.public_facts_only is not true)")
    verdict = _get(m, "verdicts", cname, default="missing")
    if verdict in ("fail", "blocked", "missing"):
        deny.add(f"the privacy verdict for context {_q(cname)} is {_q(verdict)}; it cannot be released or approved")
    for c in inp.get("privacy_checks") or []:
        if c.get("status") == "fail":
            deny.add(f"{c.get('attack')}: {_v(c.get('value'))} is above the limit {_v(c.get('limit'))}")
    mn = _get(pol, "quality", "min_pass_rate")
    pr = _get(inp, "quality", "pass_rate")
    if mn is not None and pr is not None and pr < mn:
        deny.add(f"quality pass rate {_v(pr)} is below the organisation's minimum {_v(mn)}")
    if req.get("target") not in ((ctx or {}).get("allowed_targets") or []):
        deny.add(f"target {_q(req.get('target'))} is not allowed for context {_q(cname)}")
    if external:
        pkg = set(req.get("package") or [])
        for r in m.get("reports") or []:
            if _get(r, "digest", "sha256") in pkg and not r.get("external_safe"):
                deny.add(f"the {r.get('kind')} report was computed on real data without DP and cannot leave the "
                         "organisation")
    approval_required = bool(external or (ctx is not None and verdict in _get(ctx, "approval", "on_verdicts", default=[])))
    roles = _get(ctx or {}, "approval", "approver_roles", default=[]) or []
    sod = _get(pol, "approvals", "separation_of_duties", default=True)
    operator = _get(m, "run", "operator", "key")
    valid = set()
    for a in inp.get("approvals") or []:
        if a.get("decision") == "approve" and a.get("manifest") == inp.get("manifest_digest") \
                and any(r in roles for r in a.get("roles") or []) and not (sod and a.get("key") == operator) \
                and not a.get("expired"):
            valid.add(a.get("key"))
    if approval_required and ctx is not None:
        need = _get(ctx, "approval", "quorum", default=1)
        if len(valid) < need:
            deny.add(f"context {_q(cname)} needs {need} approval(s) from {_v(roles)}; {len(valid)} valid")
    for a in inp.get("approvals") or []:
        if a.get("decision") == "reject" and a.get("manifest") == inp.get("manifest_digest"):
            deny.add(f"rejected by {a.get('approver')}: {a.get('reason', 'no reason given')}")
    return {"allow": not deny, "approval_required": approval_required, "reasons": sorted(deny)}


def invariants(inp: dict) -> list[str]:
    """The four rules enforced in code whatever the policy says (design sec. 8.3)."""
    pol, m, led, req = inp.get("policy") or {}, inp.get("manifest") or {}, inp.get("ledger") or {}, inp.get("request") or {}
    ctx = _get(pol, "contexts", req.get("context"))
    external = True if ctx is None else bool(ctx.get("external"))
    out = []
    if external:
        unit = _get(pol, "domains", _get(m, "source", "domain"), "unit", "entity")
        if _get(m, "privacy", "mode") != "dp" or unit is None or _get(m, "privacy", "unit") != unit:
            out.append("invariant: a release outside the organisation needs differential privacy for the privacy unit "
                       "of its domain")
        operator = _get(m, "run", "operator", "key")
        if not any(a.get("decision") == "approve" and a.get("manifest") == inp.get("manifest_digest")
                   and a.get("key") != operator and not a.get("expired") for a in inp.get("approvals") or []):
            out.append("invariant: a release outside the organisation needs a valid approval by someone other than "
                       "the operator")
    if _get(m, "verdicts", req.get("context")) in ("fail", "blocked"):
        out.append("invariant: a 'fail' or 'blocked' privacy verdict can never be released or approved")
    if _get(m, "privacy", "mode") == "dp" and led.get("domain_rho_after", 0) > led.get("lifetime_rho_cap", 0) + 1e-12:
        out.append("invariant: the domain's lifetime budget would be exceeded")
    return out


def rego_dir() -> Path:
    return Path(str(resources.files("relsynth.governance").joinpath("rego")))


def opa_binary() -> str | None:
    return os.environ.get("RELSYNTH_OPA") or shutil.which("opa")


def opa(inp: dict, modules=()) -> dict:
    """Evaluate the decision with Open Policy Agent; any failure is a deny."""
    exe = opa_binary()
    if not exe:
        return {"allow": False, "approval_required": True,
                "reasons": ["the policy needs Open Policy Agent (engine: opa) but no `opa` binary was found; "
                            "install OPA or set RELSYNTH_OPA"], "engine_error": True}
    args = [exe, "eval", "--format", "json", "--stdin-input", "-d", str(rego_dir() / "release.rego")]
    for mod in modules:
        args += ["-d", str(mod)]
    args.append(QUERY)
    try:
        p = subprocess.run(args, input=json.dumps(inp), capture_output=True, text=True,  # noqa: S603 # nosec B603
                           timeout=60, check=False)
        res = json.loads(p.stdout)["result"][0]["expressions"][0]["value"]
        return {"allow": bool(res["allow"]), "approval_required": bool(res["approval_required"]),
                "reasons": list(res["reasons"])}
    except Exception as e:  # noqa: BLE001 - any engine failure denies
        return {"allow": False, "approval_required": True,
                "reasons": [f"the policy engine failed ({type(e).__name__}); a failure is a deny"], "engine_error": True}


def opa_version() -> str | None:
    exe = opa_binary()
    if not exe:
        return None
    try:
        p = subprocess.run([exe, "version"], capture_output=True, text=True, timeout=20,  # noqa: S603 # nosec B603
                           check=False)
        return "opa " + p.stdout.split("\n", 1)[0].split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        return None


def decide(inp: dict, engine: str = "builtin", modules=()) -> dict:
    """The release decision: the engine's rules, then the invariants on top."""
    if engine == "opa":
        res = opa(inp, modules)
    elif engine == "builtin":
        res = builtin(inp)
    else:
        res = {"allow": False, "approval_required": True, "reasons": [f"unknown policy engine {engine!r}"]}
    inv = invariants(inp)
    if inv:
        res = dict(res, allow=False, reasons=sorted(set(res["reasons"]) | set(inv)))
    from .._version import __version__
    res["engine"] = (opa_version() or "opa") if engine == "opa" else f"relsynth-builtin {__version__}"
    return res
