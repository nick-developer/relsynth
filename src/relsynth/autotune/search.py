# SPDX-License-Identifier: Apache-2.0
"""Proposals of the self-tuning loop.

1. The spec fixes of `relsynth doctor` for the incumbent's bundle (when the
   objectives allow spec patches).
2. Guided moves: for each failing family (by excess failures) or violated
   constraint, the settings whose registry entry names it in `affects`, one
   step in the registry's direction (both directions when it has none), on the
   setting's scale: x2 or /2 on a log scale, a tenth of the range on a bounded
   linear one, every other choice for a choice.
3. Model-based moves, when the guided ones are spent: a Parzen-style proposal
   (Bergstra et al. 2011) that perturbs the incumbent on its best settings'
   neighbourhood among the settings of the search space.
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..settings import Param, params, search_space


def key_of(delta: dict, patches: dict | None) -> str:
    return json.dumps({"delta": delta, "patches": patches or {}}, sort_keys=True, default=str)


def _current(p: Param, delta: dict, base: dict):
    if p.key in delta:
        return delta[p.key]
    cur = base
    for part in p.key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return p.default
        cur = cur[part]
    return cur


def step(p: Param, value, sign: int, accel: int = 0):
    """One step of p from value in direction sign (None at a bound).  accel
    counts the moves of p in this direction accepted in a row: each doubles the
    step's exponent on a log scale (x2, x4, x16, ...) and adds one tenth of the
    range on a linear one, as the expansion step of a pattern search (Hooke and
    Jeeves 1961)."""
    if p.type == "bool":
        return not value
    if p.type == "choice":
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        value = p.default
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
    lo = p.low if p.low is not None else -math.inf
    hi = p.high if p.high is not None else math.inf
    if p.scale == "log" and value > 0:
        f = 2.0 ** (2 ** accel if accel else 1)
        new = value * (f if sign > 0 else 1 / f)
    elif math.isfinite(lo) and math.isfinite(hi) and hi - lo <= 10:
        new = value + sign * 0.1 * (1 + accel) * (hi - lo)
    else:
        new = value * (1.5 if sign > 0 else 1 / 1.5) if value else (1 if sign > 0 else None)
        if new is None:
            return None
    new = min(max(new, lo), hi)
    if p.type == "int":
        new = int(round(new))
        if new == value:
            new = int(value + sign)
            new = min(max(new, lo), hi)
    else:
        new = round(float(new), 6)
    return None if new == value else new


def _learned(learned, k, f):
    """Weight of setting k for family f from the history's one-setting effects:
    0.25 when every move of k left f's failures unchanged (within half a check
    per seed), 2 when a move of k removed at least one failure per seed."""
    ch = (learned or {}).get(f, {}).get(k)
    if not ch:
        return 1.0
    if min(ch) <= -1:
        return 2.0
    return 0.25 if all(abs(x) < 0.5 for x in ch) else 1.0


def guided(failing: dict, violated: dict, delta: dict, base: dict, space: dict[str, Param],
           learned: dict | None = None, accel: dict | None = None) -> list:
    """Single-setting moves for the failing families ({family: severity}) and
    violated constraints ({family: severity}), most promising first.  learned:
    the history's one-setting effects (report.effects), which reweight the
    registry's prior map: a setting that never moved a family loses priority for
    it, one that did gains."""
    need = {**failing, **violated}
    scored: dict = {}
    for k, p in space.items():
        hit = [f for f in p.affects if f in need]
        if not hit:
            continue
        cur = _current(p, delta, base)
        if p.type == "choice":
            for c in p.choices:
                if c != cur and c != "map":
                    scored[(k, json.dumps(c))] = (sum(need[f] * _learned(learned, k, f) for f in hit),
                                                  {**delta, k: c})
            continue
        if cur != p.default and p.check(p.default) is None:
            # back to the default: the spec (or an earlier move) set this away from it
            gain = sum(need[f] * _learned(learned, k, f) for f in hit)
            scored[(k, json.dumps(p.default, default=str))] = (gain + 0.01, {kk: vv for kk, vv in delta.items()
                                                                             if kk != k} | {k: p.default})
        signs = set()
        for f in hit:
            d = p.direction.get(f)
            signs |= {d} if d else {+1, -1}
        for sgn in signs:
            new = step(p, cur, sgn, (accel or {}).get((k, sgn), 0))
            if new is None:
                continue
            gain = sum(need[f] * _learned(learned, k, f) for f in hit if p.direction.get(f) in (sgn, None))
            # moves against a family that is met lose a little priority (they may break it)
            loss = sum(0.5 for f, d in p.direction.items() if d == -sgn and f not in need)
            scored[(k, json.dumps(new))] = (gain - loss, {**delta, k: new})
    order = sorted(scored.items(), key=lambda kv: (-kv[1][0], kv[0][0]))
    return [d for _, (_, d) in order]


def model_based(history: list, delta: dict, base: dict, space: dict[str, Param], rng: np.random.Generator,
                n: int = 4) -> list:
    """Perturbations of the incumbent around its neighbourhood: 1-2 settings of
    the space, drawn on their scale around the values of the best trials."""
    if not space:
        return []
    good = [h["delta"] for h in sorted(history, key=lambda h: h["score"])[: max(2, len(history) // 4)]]
    keys = sorted(space)
    out = []
    for _ in range(n * 3):
        d = dict(delta)
        for k in rng.choice(keys, size=min(len(keys), int(rng.integers(1, 3))), replace=False):
            k = str(k)
            p = space[k]
            cur = _current(p, d, base)
            seen = [g[k] for g in good if k in g]
            centre = seen[int(rng.integers(len(seen)))] if seen and rng.random() < 0.5 else cur
            if p.type == "choice":
                ch = [c for c in p.choices if c != "map"]
                d[k] = ch[int(rng.integers(len(ch)))]
            elif p.type == "bool":
                d[k] = bool(rng.random() < 0.5)
            elif isinstance(centre, (int, float)) and not isinstance(centre, bool):
                if p.scale == "log" and centre > 0:
                    v = centre * math.exp(rng.normal(0, 0.7))
                else:
                    span = (p.high - p.low) if p.low is not None and p.high is not None and p.high - p.low <= 10 \
                        else max(abs(centre), 1.0)
                    v = centre + rng.normal(0, 0.25 * span)
                lo = p.low if p.low is not None else -math.inf
                hi = p.high if p.high is not None else math.inf
                v = min(max(v, lo), hi)
                d[k] = int(round(v)) if p.type == "int" else round(float(v), 6)
        d = {k: v for k, v in d.items() if v != params()[k].default or k in delta}
        if d != delta and d not in out:
            out.append(d)
        if len(out) >= n:
            break
    return out


def space_for(obj) -> dict[str, Param]:
    """The settings the loop may move: the objectives' search list, else every tunable one."""
    s = obj.search
    if s == "auto":
        return search_space()
    keys = list(s)
    sp = dict(search_space(keys))
    if isinstance(s, dict):
        from dataclasses import replace
        for k, b in s.items():
            if isinstance(b, dict):
                sp[k] = replace(sp[k], low=b.get("low", sp[k].low), high=b.get("high", sp[k].high), tunable=True)
            elif isinstance(b, list):
                sp[k] = replace(sp[k], type="choice", choices=tuple(b), tunable=True)
    return sp
