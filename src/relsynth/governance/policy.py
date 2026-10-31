# SPDX-License-Identifier: Apache-2.0
"""The organisation policy (relsynth.org-policy/1, design sec. 4).

The privacy owner of the organisation writes it and signs it; it holds the
privacy domains with their budgets, the release contexts with their limits and
approval rules.  relsynth has **no** built-in epsilon for external releases:
`skeleton()` writes a policy whose domains have no budget, and `load()`
refuses it until the privacy owner has written one.

`Policy.caps(domain)` converts the epsilon caps to zCDP rho at the domain's
delta (Prop. 800); the ledger compares in rho, where composition is exact
addition.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import timedelta
from importlib import resources
from pathlib import Path

import yaml

from ..errors import GovernanceError
from .accounting import rho_of
from .canon import canonical, sha256

SCHEMA_ID = "relsynth.org-policy/1"
POLICY_TYPE = "https://github.com/nick-developer/relsynth/org-policy/1"

GUIDANCE = ("see docs/governance/choosing-epsilon.md: NIST SP 800-226 sec. 2.2 gives no rule for epsilon; values "
            "below 1 are strong, deployments use 1 < epsilon <= 20 and the risk above 1 should be measured with "
            "`relsynth privacy`; delta should be at most 1/n^2 and never above 1e-5")


def schema() -> dict:
    return json.loads(resources.files("relsynth.governance").joinpath("schema/org-policy-1.schema.json").read_text())


_DUR = re.compile(r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?$")


def duration(s: str) -> timedelta:
    """An ISO 8601 duration of years, months, weeks and days (a year is 365 days, a month 30)."""
    m = _DUR.match(s or "")
    if not m or s == "P":
        raise GovernanceError(f"not an ISO 8601 duration: {s!r}")
    y, mo, w, d = (int(x or 0) for x in m.groups())
    return timedelta(days=365 * y + 30 * mo + 7 * w + d)


@dataclass
class Caps:
    """A domain's caps in epsilon and in rho at the domain's delta."""
    delta: float
    lifetime_epsilon: float
    lifetime_rho: float
    per_release_epsilon: float | None
    per_release_rho: float | None
    period: timedelta | None
    period_epsilon: float | None
    period_rho: float | None

    def to_dict(self):
        return {k: (str(v) if isinstance(v, timedelta) else v) for k, v in self.__dict__.items()}


class Policy:
    """A validated organisation policy."""

    def __init__(self, doc: dict, raw: bytes | None = None):
        self.doc = doc
        self.digest = sha256(canonical(doc))
        self._caps: dict = {}

    @property
    def version(self) -> str:
        return str(self.doc["version"])

    @property
    def domains(self) -> dict:
        return self.doc["domains"]

    @property
    def contexts(self) -> dict:
        return self.doc["contexts"]

    def context(self, name: str) -> dict:
        if name not in self.contexts:
            raise GovernanceError(f"context {name!r} is not defined in the organisation policy "
                                  f"(defined: {', '.join(self.contexts)})")
        return self.contexts[name]

    def domain_of_source(self, source_id: str | None) -> str | None:
        for name, d in self.domains.items():
            if any(s["id"] == source_id for s in d["sources"]):
                return name
        return None

    def caps(self, domain: str) -> Caps:
        if domain not in self._caps:
            b = self.domains[domain]["budget"]
            delta = float(b["delta"])
            le = float(b["lifetime"]["epsilon"])
            pr = b.get("per_release", {}).get("epsilon_max")
            pe = b.get("period")
            self._caps[domain] = Caps(
                delta=delta, lifetime_epsilon=le, lifetime_rho=rho_of(le, delta),
                per_release_epsilon=float(pr) if pr else None, per_release_rho=rho_of(float(pr), delta) if pr else None,
                period=duration(pe["length"]) if pe else None, period_epsilon=float(pe["epsilon"]) if pe else None,
                period_rho=rho_of(float(pe["epsilon"]), delta) if pe else None)
        return self._caps[domain]

    @property
    def engine(self) -> str:
        return self.doc.get("engine", "builtin")


def _budget_missing(doc: dict) -> list[str]:
    out = []
    for name, d in (doc.get("domains") or {}).items():
        b = (d or {}).get("budget") or {}
        if not b.get("delta") or not (b.get("lifetime") or {}).get("epsilon"):
            out.append(f"the organisation must set epsilon and delta for domain {name!r} (budget.delta and "
                       f"budget.lifetime.epsilon); relsynth has no default budget; {GUIDANCE}")
    return out


def validate(doc: dict) -> list[str]:
    """Problems of a policy document, as sentences ([] when valid)."""
    if not isinstance(doc, dict):
        return ["the policy is not a YAML mapping"]
    probs = _budget_missing(doc)
    if probs:
        return probs
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return ["validating a policy needs jsonschema: pip install 'relsynth[gov]'"]
    v = jsonschema.Draft202012Validator(schema())
    for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in e.absolute_path) or "(top)"
        probs.append(f"{where}: {e.message}")
    for name, c in (doc.get("contexts") or {}).items():
        if c.get("external") and not c.get("dp_required"):          # also in the schema; stated plainly here
            probs.append(f"context {name!r} is external: dp_required must be true (DP is required outside the "
                         "organisation)")
    for name, d in (doc.get("domains") or {}).items():
        b = d["budget"]
        pr = (b.get("per_release") or {}).get("epsilon_max")
        if pr and pr > b["lifetime"]["epsilon"]:
            probs.append(f"domain {name!r}: per_release.epsilon_max {pr} is above the lifetime cap")
        pe = b.get("period")
        if pe and pe["epsilon"] > b["lifetime"]["epsilon"]:
            probs.append(f"domain {name!r}: the period cap {pe['epsilon']} is above the lifetime cap")
    seen: dict = {}
    for name, d in (doc.get("domains") or {}).items():
        for s in d["sources"]:
            if s["id"] in seen:
                probs.append(f"source {s['id']!r} is in domains {seen[s['id']]!r} and {name!r}; a source belongs to "
                             "exactly one domain")
            seen[s["id"]] = name
    if doc.get("rego") and doc.get("engine", "builtin") != "opa":
        probs.append("the policy lists Rego modules: set engine: opa (the built-in engine runs relsynth's rules only)")
    return probs


def load(path_or_doc) -> Policy:
    """Read and validate a policy (YAML or JSON file, or a mapping)."""
    if isinstance(path_or_doc, dict):
        doc = path_or_doc
    else:
        p = Path(path_or_doc)
        if not p.exists():
            raise GovernanceError(f"no policy file {p}")
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    probs = validate(doc)
    if probs:
        raise GovernanceError("the organisation policy is invalid:\n  - " + "\n  - ".join(probs))
    doc = json.loads(json.dumps(doc, default=str))       # dates -> strings, so the digest is stable
    return Policy(doc)


def skeleton(organisation: str = "Your organisation", owner: str = "Data Protection Officer") -> str:
    """A policy to fill in: contexts and approval rules, and domains with no budget."""
    return f"""# Organisation policy (relsynth.org-policy/1).  Fill in every TODO, then sign it:
#   relsynth gov policy sign policy.yaml --registry REGISTRY --key owner.key
# relsynth ships no epsilon: the privacy owner chooses each budget.  {GUIDANCE}.
schema: relsynth.org-policy/1
organisation: {organisation}
version: "1"
privacy_owner:
  name: {owner}
  role: dpo
domains:
  people:                       # TODO: one domain per population of people
    unit:
      entity: Customer          # TODO: the IR privacy unit of every source below
    adjacency: add_remove
    sources:
      - id: my-source           # TODO: the source id given to `relsynth run --source-id`
    budget:
      accounting: zcdp
      delta:                    # TODO: e.g. at most 1/n^2 for n people; never above 1e-5
      lifetime:
        epsilon:                # TODO: hard cap for the life of the domain
contexts:
  internal:
    external: false
    attacker_knows: quasi_identifiers
    dp_required: false
    limits: {{risk: 0.10, attribute_leakage: 0.05, mia_auc: 0.65, mia_tpr_at_fpr_0.01: 0.10, record_auc: 0.85}}
    approval: {{on_verdicts: [review], approver_roles: [dpo], quorum: 1}}
    allowed_targets: [dev-sandbox]
  partner:
    external: true
    attacker_knows: all
    dp_required: true
    limits: {{risk: 0.05, attribute_leakage: 0.03, mia_auc: 0.60, mia_tpr_at_fpr_0.01: 0.05, record_auc: 0.75}}
    approval: {{on_verdicts: [pass, review], approver_roles: [dpo], quorum: 1}}
    allowed_targets: []
quality:
  min_pass_rate: 0.95
approvals:
  separation_of_duties: true
  expires_after: P30D
reidentification:
  allowed_roles: [dpo]
"""


# ----------------------------------------------------------------- signing
def sign(policy: Policy, signer, registry) -> int:
    """Sign a policy version and record it in the registry (only a key with the
    privacy owner's role may sign)."""
    from .sign import envelope
    trusted = registry.trusted()
    role = policy.doc["privacy_owner"]["role"]
    if signer.id not in trusted or role not in trusted[signer.id]["roles"]:
        raise GovernanceError(f"only a trusted key with the privacy owner's role {role!r} may sign the policy")
    st = {"_type": "https://in-toto.io/Statement/v1",
          "subject": [{"name": "org-policy", "digest": {"sha256": policy.digest}}],
          "predicateType": POLICY_TYPE, "predicate": policy.doc}
    d = registry.put_json(envelope(st, signer))
    return registry.append({"type": "policy", "envelope": d, "policy": policy.digest, "version": policy.version})


def current(registry) -> Policy:
    """The latest signed policy in the registry, with its signature checked."""
    from .sign import payload, verify_envelope
    recs = list(registry.entries("policy"))
    if not recs:
        raise GovernanceError("the registry has no signed organisation policy; sign one with `relsynth gov policy sign`")
    env = registry.get_json(recs[-1]["entry"]["envelope"])
    trusted = registry.trusted()
    doc = payload(env)["predicate"]
    role = doc["privacy_owner"]["role"]
    good = [k for k in verify_envelope(env, {i: v["public"] for i, v in trusted.items()}) if role in trusted[k]["roles"]]
    if not good:
        raise GovernanceError("the registry's current policy is not signed by a key with the privacy owner's role")
    return load(doc)
