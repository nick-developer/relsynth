# SPDX-License-Identifier: Apache-2.0
"""Records of agent actions (T08 registry, RFC-T16-02).

Every agent action is recorded with the agent's own identity, the person it
acted for, the backend and model, the data scope, and digests of what it read
and wrote: never the data, the prompts' contents or the source rows.

* `action_statement` builds an in-toto Statement v1 (predicate type
  ``agent-action/1``).
* `record_action` signs it with the agent's key (a key whose trusted role is
  ``agent``) and appends it to the governance registry as an ``agent.action``
  entry, so the registry's hash chain and `relsynth gov verify` cover it.
* `manifest_entry` is what a run manifest lists under ``agents`` when a spec
  shaped by accepted proposals is run: the action's digest, the agent, how
  many proposals were accepted and rejected, and by whom.

Agent keys cannot approve: `governance.approve` refuses them and
`verify_registry` flags any approval signed by one.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .._version import __version__

ACTION_TYPE = "https://github.com/nick-developer/relsynth/agent-action/1"
STATEMENT = "https://in-toto.io/Statement/v1"


def digest(obj) -> str:
    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
    else:
        b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    return hashlib.sha256(b).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def action_statement(*, task: str, agent: str, on_behalf_of: str, backend: dict, scope: str, inputs: dict,
                     output, proposals: int, flags: int = 0, started: str | None = None) -> dict:
    """An agent-action statement.  `inputs` maps names to objects (hashed here)
    or to ready digests ({"sha256": ...})."""
    ins = {k: (v if isinstance(v, dict) and set(v) == {"sha256"} else {"sha256": digest(v)})
           for k, v in inputs.items()}
    out_d = digest(output)
    return {"_type": STATEMENT, "subject": [{"name": f"agent-output:{task}", "digest": {"sha256": out_d}}],
            "predicateType": ACTION_TYPE,
            "predicate": {"task": task, "agent": {"identity": agent, "software": f"relsynth {__version__}",
                                                  "backend": backend.get("name"), "model": backend.get("model")},
                          "on_behalf_of": on_behalf_of, "scope": scope, "inputs": ins,
                          "output": {"sha256": out_d, "proposals": int(proposals), "flags": int(flags)},
                          "authority": "proposals only: nothing is applied without a person's decision",
                          "started": started or _now(), "finished": _now()}}


def record_action(registry, statement: dict, signer) -> tuple[str, int]:
    """Sign and append an agent-action statement; returns (statement digest, log seq).
    The signer must be a trusted key with the role ``agent``."""
    from ..errors import GovernanceError
    from ..governance.canon import canonical, sha256
    from ..governance.sign import envelope
    trusted = registry.trusted()
    if signer.id not in trusted or "agent" not in trusted[signer.id]["roles"]:
        raise GovernanceError("agent actions are signed with a trusted key whose role is 'agent'")
    env = envelope(statement, signer)
    d = registry.put_json(env)
    seq = registry.append({"type": "agent.action", "envelope": d, "task": statement["predicate"]["task"],
                           "agent": statement["predicate"]["agent"]["identity"]})
    return sha256(canonical(statement)), seq


def manifest_entry(action_digest: str, agent: str, task: str, decisions: dict, decided_by: str) -> dict:
    """The run manifest's ``agents`` item for one reviewed action."""
    acc = sum(1 for v in decisions.values() if v == "accept")
    rej = sum(1 for v in decisions.values() if v == "reject")
    return {"action": {"sha256": action_digest}, "agent": agent, "task": task, "accepted": acc, "rejected": rej,
            "accepted_by": decided_by}


__all__ = ["action_statement", "record_action", "manifest_entry", "digest", "ACTION_TYPE"]
