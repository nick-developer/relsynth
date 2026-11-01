# SPDX-License-Identifier: Apache-2.0
"""Agent-assisted operations (roadmap T16): agents that draft and explain,
never decide.

* `review` answers the `review:` questions of an introspected draft with
  evidence measured on every row; `apply_decisions` makes the changes a
  person accepted.
* `explain_quality`, `explain_privacy`, `explain_scorecard` put reports into
  plain language, with every number taken from the report.
* `doctor_proposals` turns `relsynth doctor` findings into proposals.
* `draft_contract` and `draft_policy` draft an ODCS data contract and an
  organisation policy for people to complete and sign.

Agents never see rows (scope.py), never approve, release or change epsilon,
privacy settings or policies (guard.py), and every action is recorded
(record.py).  docs/agents.md is the guide.
"""
from __future__ import annotations

from .backends import OpenAICompatBackend, RulesBackend, ScriptedBackend, make_backend
from .guard import FORBIDDEN_ACTIONS
from .items import ReviewItem, parse
from .review import ReviewResult, apply_decisions, review
from .scope import Scope, for_roles
from .drafts import apply_doctor, doctor_proposals, draft_contract, draft_policy
from .explain import explain as explain_report

__all__ = ["review", "apply_decisions", "ReviewResult", "ReviewItem", "parse", "Scope", "for_roles",
           "RulesBackend", "OpenAICompatBackend", "ScriptedBackend", "make_backend", "FORBIDDEN_ACTIONS",
           "explain_report", "doctor_proposals", "apply_doctor", "draft_contract", "draft_policy"]
