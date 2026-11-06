# SPDX-License-Identifier: Apache-2.0
"""Domain checks of the golden databases in the form `relsynth autotune` takes
(`domain_checks: bench/autotune_domain.py:<dataset>`): function(real tables,
synthetic tables) -> {name: (real value, synthetic value)}.  An identity (real
value 0 or 1) becomes a hard constraint; other values are recorded only."""
from __future__ import annotations


def adventureworks(real, syn):
    """G1-G14 of bench/checks/adventureworks.py (layer-2 LOG.md ground truth)."""
    from bench.checks import adventureworks as m
    real = {t: d for t, d in (real or {}).items() if t in m.TABLES}
    syn = {t: d for t, d in (syn or {}).items() if t in m.TABLES}
    rs, ss = m.checks(real), m.checks(syn)
    return {k: (rs.get(k), ss.get(k)) for k in dict.fromkeys([*rs, *ss])}
