# SPDX-License-Identifier: Apache-2.0
"""Release rules: when may the synthetic data of a run leave the service?

This is the service side of the governance design (docs/governance/design.md
§9, decisions D3 and D4).  The signed manifests, the privacy budget ledger and
the OPA release policy are built by T08 in ``relsynth.governance``; when that
package is installed and exposes ``evaluate_release`` (RFC-T09-01), its decision
is added to the rules here.  The rules below only ever make a release
stricter, never looser:

1. A run that did not succeed cannot be released.
2. ``external`` contexts are refused until the DP engine (T07) and the
   governance build (T08) are present: the user decided that every release
   outside the organisation needs differential privacy (2026-10-26).
3. Internal releases need ``approval_quorum`` approvals by a data owner or
   privacy officer (D4), none of them the person who started the run
   (separation of duties), and no rejection or revocation after them.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

CONTEXTS = ("internal", "external")


@dataclass
class Decision:
    allow: bool
    approval_required: bool
    reasons: list[str] = field(default_factory=list)


def governance():
    """The T08 governance package, if installed and new enough; else None."""
    try:
        g = importlib.import_module("relsynth.governance")
    except ImportError:
        return None
    return g if hasattr(g, "evaluate_release") else None


def can_run_in(context: str) -> str | None:
    """Why a run in this context cannot be started, or None."""
    if context not in CONTEXTS:
        return f"unknown release context {context!r}; use one of {CONTEXTS}"
    if context == "external":
        g = governance()
        if g is None or not getattr(g, "DP_AVAILABLE", False):
            return ("external releases need differential privacy (user decision 2026-10-26); the DP engine (T07) "
                    "and the governance build (T08) are not installed, so only internal runs are possible")
    return None


def decide(run: dict, approvals: list[dict], settings) -> Decision:
    reasons: list[str] = []
    if run["state"] != "succeeded":
        return Decision(False, False, [f"the run is {run['state']}, not succeeded"])
    if run.get("kind") != "run":
        return Decision(False, False, ["only synthesis runs produce releasable data"])
    why = can_run_in(run.get("context") or "internal")
    if why:
        return Decision(False, False, [why])
    needs = run["context"] == "external" or settings.approval_internal
    valid: list[dict] = []
    for a in sorted(approvals, key=lambda a: a["created_at"]):
        if a["decision"] == "approve":
            if settings.separation_of_duties and a["approver"] == run["created_by"]:
                continue
            if a["approver"] not in {v["approver"] for v in valid}:
                valid.append(a)
        else:                       # a rejection or revocation voids every earlier approval
            valid = []
            reasons.append(f"{a['decision']}ed by {a['approver']}: {a.get('reason') or 'no reason given'}")
    if needs and len(valid) < settings.approval_quorum:
        reasons.append(f"needs {settings.approval_quorum} approval(s) by a data owner or privacy officer other "
                       f"than the run's creator; {len(valid)} valid")
        return Decision(False, True, reasons)
    g = governance()
    if g is not None:
        d = g.evaluate_release(run=run, approvals=valid)
        if not d.get("allow", False):
            return Decision(False, bool(d.get("approval_required")), list(d.get("reasons", [])))
    return Decision(True, needs, [])


def release_state(run: dict, approvals: list[dict], settings) -> str:
    if run["state"] != "succeeded" or run.get("kind") != "run":
        return "none"
    if any(a["decision"] in ("reject", "revoke") for a in approvals) and not decide(run, approvals, settings).allow:
        return "rejected"
    return "approved" if decide(run, approvals, settings).allow else "pending_approval"
