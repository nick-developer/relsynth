"""Which tables are fitted from statistics computed where the data lives.

A fact table is pushed down when the relational model needs it only through
the aggregates of relsynth.pushdown.stats, and every row-level part of its model
can be fitted from a uniform sample:

* it is a fact (role fact, mode synthesize) that no table references: a fact
  that is the parent of a nested fact or of a link is an entity of its own rows;
* no computed column of another table is filled from it on the source
  (an aggregate rule whose column the source does not store);
* it has no type 2 history (scd).

A pushed fact may be nested in a loaded parent fact (its pairs come from GROUP BY
over the composite key and the partner), be split into order headers (the header
rows are the first row of each document key, computed in the database) or be the
base of a view (a view is rebuilt from the synthetic base; its real rows are
sampled for the evaluation only), and carry a running balance (the opening
balances are read from the first row of each group, computed in the database).

Entities (dimensions) are always loaded: each of their rows carries its own
latent segment.  Facts that fail a condition are loaded whole and the reason is
reported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("relsynth.pushdown")


@dataclass
class PushdownPlan:
    """The decision per table: `pushed` facts (fitted from statistics and a
    sample of `sample_rows`), the rest loaded; `reasons` says why a large fact
    was not pushed down."""
    mode: str = "off"
    sample_rows: int = 500_000
    pushed: dict[str, int] = field(default_factory=dict)       # table -> source rows
    loaded: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "sample_rows": self.sample_rows, "pushed": dict(self.pushed),
                "loaded": list(self.loaded), "reasons": dict(self.reasons)}


def ineligible_reason(spec: dict, t: str, stored: dict[str, set] | None = None) -> str | None:
    """Why fact t cannot be pushed down (None when it can)."""
    T = spec["tables"]
    ts = T[t]
    if ts.get("role") != "fact":
        return "not a fact (entities carry one latent segment per row)"
    if ts.get("mode") == "copy":
        return "copied table"
    rels = spec.get("relationships", [])
    if any(r["parent"] == t for r in rels):
        return "other tables reference it"
    if ts.get("scd"):
        return "type 2 history"
    for r in spec.get("rules", []):
        if r.get("type") == "aggregate" and r.get("from") == t:
            have = (stored or {}).get(r["table"])
            if have is not None and r["column"] not in have:
                return f"the source does not store {r['table']}.{r['column']}, which is computed from its rows"
    return None


def plan(spec: dict, conn, counts: dict[str, int] | None = None, stored: dict[str, set] | None = None) -> PushdownPlan:
    """Decide what to push down for this spec and source connector."""
    cfg = (spec.get("synthesis") or {}).get("pushdown") or {}
    mode = cfg.get("mode", "auto")
    p = PushdownPlan(mode=mode, sample_rows=int(cfg.get("sample_rows", 500_000)))
    names = list(spec["tables"])
    if mode in ("off", False, None) or not getattr(conn, "pushdown", False) and conn.kind not in ("csv", "parquet"):
        p.loaded = names
        return p
    min_rows = int(cfg.get("min_rows", 1_000_000))
    counts = dict(counts or {})
    for t in names:
        if spec["tables"][t].get("role") != "fact":
            p.loaded.append(t)
            continue
        n = counts.get(t)
        if n is None:
            n = counts[t] = conn.count(t)
        if mode == "auto" and n < min_rows:
            p.loaded.append(t)
            continue
        why = ineligible_reason(spec, t, stored)
        if why:
            p.loaded.append(t)
            p.reasons[t] = why
            log.warning("pushdown: %s (%s rows) is loaded whole: %s", t, f"{n:,}", why)
            continue
        p.pushed[t] = n
    return p
