# SPDX-License-Identifier: Apache-2.0
"""What an agent may see of the source data (Prop. 1600).

An agent never sees source rows.  It works on the spec, the review text and
the aggregates its evidence tools compute, and the data scope decides how much
of those aggregates it gets:

``schema``      names, types and the spec only; no evidence tool reads data
``aggregates``  counts, shares and distinct counts; no individual value
``values``      as ``aggregates``, plus single values that occur at least
                ``k`` times in their column (default 10), the way a frequency
                table with small-cell suppression would publish them

The scope of a task is the lowest of the scope of the person who asked for it
(by their role) and the ceiling configured for the agent, so an agent acting
for an engineer never sees more than an engineer may, whatever the agent's
own account allows.  Roles map to scopes as below (``docs/agents.md``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

LEVELS = ("schema", "aggregates", "values")

# service roles (relsynth.service.auth.ROLES) -> the highest scope an agent acting for them may use
ROLE_SCOPE = {"consumer": "schema", "tenant_admin": "schema", "engineer": "aggregates",
              "data_owner": "values", "privacy_officer": "values", "agent": "aggregates"}
DEFAULT_K = 10


def lower(a: str, b: str) -> str:
    return LEVELS[min(LEVELS.index(a), LEVELS.index(b))]


def for_roles(roles, ceiling: str = "values") -> str:
    """The scope of a task asked for by someone holding `roles`, under the agent's ceiling."""
    best = "schema"
    for r in roles or ():
        s = ROLE_SCOPE.get(r, "schema")
        if LEVELS.index(s) > LEVELS.index(best):
            best = s
    return lower(best, ceiling)


@dataclass(frozen=True)
class Scope:
    level: str = "aggregates"
    k: int = DEFAULT_K

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"unknown data scope {self.level!r}; use one of {LEVELS}")
        if int(self.k) < 2:
            raise ValueError("the value threshold k must be at least 2")

    @property
    def reads_data(self) -> bool:
        return self.level != "schema"

    @property
    def sees_values(self) -> bool:
        return self.level == "values"

    def number(self, x):
        """An aggregate, or None in the schema scope."""
        if not self.reads_data or x is None:
            return None
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    def value(self, v, count: int):
        """A single source value with its frequency: shown only in the values
        scope and only when it occurs at least k times; otherwise None."""
        if self.sees_values and count is not None and int(count) >= self.k:
            return v
        return None


__all__ = ["Scope", "LEVELS", "ROLE_SCOPE", "for_roles", "lower", "DEFAULT_K"]
