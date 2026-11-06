# SPDX-License-Identifier: Apache-2.0
"""Drafts for people to finish: doctor fixes, data contracts and policies.

* `doctor_proposals` turns the findings of `relsynth doctor` into proposals.
  A fix that touches a privacy setting (pii, copy modes, text copying limits,
  DP) is never a proposal: it is advice addressed to the privacy officer,
  without a change attached.
* `draft_contract` drafts an ODCS v3.2.0 data contract of the spec with
  status ``draft``; the owner reviews it and sets it active.
* `draft_policy` drafts an organisation policy (relsynth.org-policy/1) with
  the spec's privacy unit and source filled in and every budget left empty:
  relsynth ships no epsilon and no agent chooses one (user decision
  2026-10-26).  The draft does not validate until the privacy owner sets the
  budgets, and only the owner's key can sign it.
"""
from __future__ import annotations

import copy
import re

import yaml

from .guard import plain_text

# spec paths whose change is a privacy decision (the agent advises, a person decides)
PRIVACY_PATH = re.compile(r"^tables\.[^.]+\.mode$|(^|\.)(pii|privacy|dp|epsilon|delta|rho|budget|contexts?|release|"
                          r"max_copy_ngram|min_token_rows|copy|quasi_identifiers|sensitive)(\.|$)", re.I)


def is_privacy_change(patch: dict) -> bool:
    """A patch that sets a privacy setting, or a value that declares or removes
    personal data."""
    path = str(patch.get("path", ""))
    if PRIVACY_PATH.search(path):
        return True
    v = patch.get("value")
    if isinstance(v, dict) and ("pii" in v or v.get("type") == "pii"):
        return True
    return isinstance(v, str) and v == "pii" and path.endswith(".type")


def doctor_proposals(bundle) -> list[dict]:
    """Proposals from a diagnostics bundle (a RunResult, a bundle dict or file)."""
    from ..api import doctor
    d = doctor(bundle)
    out = []
    for i, iss in enumerate(d.issues):
        ok = [p for p in iss.patches if not is_privacy_change(p)]
        held = [p for p in iss.patches if is_privacy_change(p)]
        out.append({"id": f"D{i + 1}", "rule": iss.id, "severity": iss.severity, "kind": iss.kind,
                    "where": plain_text(iss.where, 200), "message": plain_text(iss.message, 1200),
                    "evidence": iss.evidence, "patches": ok,
                    "advice_for_privacy_officer": [{"path": p["path"], "value": p.get("value"),
                                                    "why": "a privacy setting: decided by the privacy officer"}
                                                   for p in held],
                    "status": "proposed"})
    return out


def apply_doctor(spec: dict, proposals: list[dict], accepted: list[str], *, decided_by: str) -> dict:
    """The spec with the patches of the accepted proposals (privacy advice is never applied)."""
    if not decided_by or decided_by.split(":", 1)[0] in ("agent", "service"):
        raise PermissionError("doctor proposals are accepted by a person")
    from ..doctor import apply_patches
    patches = {}
    for p in proposals:
        if p["id"] in accepted:
            for x in p["patches"]:
                if not is_privacy_change(x):
                    patches[x["path"]] = x.get("value")
    return apply_patches(copy.deepcopy(spec), patches)


def draft_contract(spec: dict, *, contract_id: str | None = None, owner: str | None = None) -> dict:
    """An ODCS contract draft of a spec (status draft)."""
    from ..ir.odcs import to_odcs
    from ..ir.v1 import lift
    raw = copy.deepcopy(spec)
    raw.pop("review", None)
    c = to_odcs(lift(raw), contract_id=contract_id, status="draft")
    c.setdefault("customProperties", []).append(
        {"property": "relsynthDraftedBy", "value": "agent (review and set status: active by the data owner)"})
    if owner:
        c["team"] = [{"username": owner, "role": "owner"}]
    return c


def draft_policy(spec: dict, *, organisation: str = "Your organisation", owner: str = "Data Protection Officer",
                 source_id: str | None = None) -> str:
    """An organisation policy draft (YAML text) with the privacy unit and the
    source filled in from the spec and every budget left for the privacy owner."""
    from ..governance.policy import skeleton
    doc = yaml.safe_load(skeleton(organisation, owner))
    unit = None
    priv = (spec.get("privacy") or {})
    if isinstance(priv.get("unit"), dict):
        unit = priv["unit"].get("entity") or priv["unit"].get("table")
    elif isinstance(priv.get("unit"), str):
        unit = priv["unit"]
    if unit is None:
        people = [t for t, ts in (spec.get("tables") or {}).items()
                  if any(isinstance(c, dict) and c.get("type") == "pii" for c in (ts.get("columns") or {}).values())]
        unit = people[0] if people else None
    dom = doc["domains"]["people"]
    if unit:
        dom["unit"]["entity"] = unit
    dom["sources"] = [{"id": source_id or str(spec.get("name") or "my-source")}]
    for k in ("delta",):
        dom["budget"][k] = None
    dom["budget"]["lifetime"]["epsilon"] = None
    head = ("# DRAFT by a relsynth agent. The privacy owner sets every budget (delta, lifetime epsilon) and checks\n"
            "# the contexts, limits and approvers; an agent never chooses epsilon. Sign it with the owner's key:\n"
            "#   relsynth gov policy sign policy.yaml --registry REGISTRY --key owner.key\n")
    return head + yaml.safe_dump(doc, sort_keys=False)


__all__ = ["doctor_proposals", "apply_doctor", "draft_contract", "draft_policy", "is_privacy_change"]
