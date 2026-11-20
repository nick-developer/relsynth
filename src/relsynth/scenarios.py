# SPDX-License-Identifier: Apache-2.0
"""Scenarios: what-if interventions on a fitted model, declared in the spec.

A fitted relsynth model is a posterior over the parameters of the source
database.  A scenario changes that model before a dataset is drawn, so the
synthetic database answers "what if?": what if there were 20% more
customers, the SMB segment shrank, prices rose 5% in the top decile,
December orders doubled, or a few products drew most of the sales.  The
fitted parameters stay as they are; a scenario is applied while a dataset is
sampled and is recorded in the sample log, so the same model gives the
baseline and every scenario.

    synthesis:
      scenario: holiday_peak            # the scenario sampled (default: none)
      scenarios:
        holiday_peak:
          description: December demand doubles, prices up 5% in the top decile
          interventions:
            - {type: scale, table: Customer, factor: 1.2}
            - {type: shares, table: Customer, column: Segment, factors: {SMB: 0.7}}
            - {type: marginal, table: Orders, column: Amount, quantiles: [[0.9, 1.0], [1.0, 1.05]]}
            - {type: intensity, table: Orders, start: 2024-12-01, end: 2025-01-01, factor: 2.0}
            - {type: degree, table: Sales, key: ProductKey, concentration: 1.5}

Interventions (Props. 1824-1828, docs/scenarios.md):

* scale: entity tables are resized (their children follow, as with
  synthesis.scale); a fact's rows per anchor are multiplied (superposition
  and thinning of the fitted count process).
* shares: the law of an entity's rows is tilted by a weight on one
  categorical column, p'(x) proportional to p(x) w(x_c) (exponential tilting,
  the minimum-KL change with the required shares; Csiszar 1975).  Rows are
  drawn from the model and accepted with probability w / max w (rejection
  sampling), so the dependence of every other column on the tilted one, and
  the segments, follow exactly; `shares` gives the target shares instead of
  the factors (stratified rejection with those quotas).
* marginal: a monotone map of a number or date column's sampled values: x'
  = a x + b, or x' = x f(p) with f piecewise linear in the value's quantile
  level p, followed by the monotone rearrangement (Chernozhukov,
  Fernandez-Val and Galichon 2010), so ranks and therefore the copula with
  every other column are kept (Sklar's theorem).
* intensity: the event rate in a date window is multiplied by `factor`.
  For a fact, rows are superposed ceil(max w) times and thinned with
  probability w(t) / ceil(max w) (Lewis and Shedler 1979), which is exactly a
  Poisson process of intensity lambda(t) w(t).  For an entity table the rows
  are tilted by w(time) as for shares, and the count is multiplied by E w.
  For a sequence table the calendar intensity of the sequence model is
  multiplied on the window and the count by the integrated change; for a
  series table the values in the window are multiplied.
* degree: link-degree laws of one foreign key.  `concentration` c tempers the
  degree weights of the parent rows, w' proportional to w^c within each segment
  (c > 1: a few rows take most links, c < 1: links spread evenly, c = 0:
  uniform); `rate` multiplies the fact's rows per anchor (as scale).

Every intervention is post-processing of a posterior draw: it reads no
source data, so a differential privacy guarantee of the model (DP mode,
whose stage 2 is this engine) is kept (Dwork and Roth 2014, Prop. 2.1).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .spec import SpecError

KINDS = ("scale", "shares", "marginal", "intensity", "degree")
MAX_POOL = 50            # rows drawn per target row before a tilted table stops (shortfall logged)
_KEYS = {"scale": {"type", "table", "factor", "note"},
         "shares": {"type", "table", "column", "factors", "shares", "note"},
         "marginal": {"type", "table", "column", "multiply", "add", "quantiles", "note"},
         "intensity": {"type", "table", "column", "start", "end", "factor", "note"},
         "degree": {"type", "table", "key", "concentration", "rate", "note"}}
_NUM = ("integer", "float")
_TIME = ("date", "datetime")


def _num(x, lo=None, strict=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
        return False
    return lo is None or (x > lo if strict else x >= lo)


def _ctype(cs):
    return cs.get("type") if isinstance(cs, dict) else cs


def _days(x):
    from .timeseries.calendar import to_days
    try:
        d = to_days(pd.Series([str(x)], dtype=object))[0]
    except (TypeError, ValueError):
        return np.nan
    return d


def _time_column(T):
    """The time column of a table: its sequence's, else its fact time column."""
    if isinstance(T.get("sequence"), dict) and T["sequence"].get("time"):
        return T["sequence"]["time"]
    if isinstance(T.get("series"), dict) and T["series"].get("time"):
        return T["series"]["time"]
    return (T.get("fact") or {}).get("time_column") if isinstance(T.get("fact"), dict) else None


# ------------------------------------------------------------------ validation
def validate(spec: dict) -> list[str]:
    """Errors of synthesis.scenarios and synthesis.scenario (a normalised spec)."""
    syn = spec.get("synthesis") or {}
    sc_all = syn.get("scenarios") or {}
    errs = []
    if not isinstance(sc_all, dict):
        return ["synthesis.scenarios must be a mapping {name: {description, interventions}}"]
    act = syn.get("scenario")
    if act is not None and act not in sc_all:
        errs.append(f"synthesis.scenario {act!r} is not one of synthesis.scenarios ({', '.join(sc_all) or 'none'})")
    T = spec.get("tables") or {}
    rels = spec.get("relationships") or []
    for name, blk in sc_all.items():
        w = f"synthesis.scenarios.{name}"
        if not isinstance(blk, dict) or not isinstance(blk.get("interventions"), list):
            errs.append(f"{w} must be a mapping with a list `interventions`")
            continue
        extra = set(blk) - {"description", "interventions"}
        if extra:
            errs.append(f"{w}: unknown key(s) {', '.join(sorted(extra))} (allowed: description, interventions)")
        targets: dict[str, str] = {}
        for i, iv in enumerate(blk["interventions"]):
            errs += _validate_one(iv, f"{w}.interventions[{i}]", T, rels, targets)
            errs += _derived_target(iv, f"{w}.interventions[{i}]", spec.get("rules") or [])
    return errs


def _derived_target(iv, w, rules):
    """A marginal change of a column that a derive, aggregate or cumulative rule
    computes from others would break the rule (TaxAmt = 0.08 * ExtendedAmount) or be
    overwritten by it; the scenario changes the inputs."""
    if not isinstance(iv, dict) or iv.get("type") != "marginal":
        return []
    from .expr import split_assignment
    for r in rules:
        if r.get("type") == "derive" and r.get("table") == iv.get("table") and "=" in str(r.get("expr", "")):
            lhs = split_assignment(str(r["expr"]))[0]
            if lhs == iv.get("column"):
                return [f"{w}: {iv['table']}.{lhs} is computed by the rule {r['expr']!r}; change its inputs instead"]
        if (r.get("type") in ("aggregate", "cumulative") and r.get("table") == iv.get("table")
                and r.get("column") == iv.get("column")):
            return [f"{w}: {iv['table']}.{r['column']} is computed by a {r['type']} rule; change its inputs instead"]
    return []


def _validate_one(iv, w, T, rels, targets):
    errs = []
    if not isinstance(iv, dict) or iv.get("type") not in KINDS:
        return [f"{w}: type must be one of {', '.join(KINDS)}"]
    k = iv["type"]
    extra = set(iv) - _KEYS[k]
    if extra:
        errs.append(f"{w}: unknown key(s) {', '.join(sorted(extra))} for type {k} "
                    f"(allowed: {', '.join(sorted(_KEYS[k] - {'type'}))})")
    t = iv.get("table")
    if t not in T:
        return errs + [f"{w}: table {t!r} is not in the spec"]
    tab = T[t]
    role, cols = tab.get("role"), tab.get("columns") or {}
    if tab.get("mode") == "copy" or role == "view":
        return errs + [f"{w}: {t} is {'copied' if role != 'view' else 'a view'} and not generated"]
    col = iv.get("column")
    if k == "scale":
        if not _num(iv.get("factor"), 0, strict=True):
            errs.append(f"{w}: factor must be a number > 0")
        if tab.get("series"):
            errs.append(f"{w}: a series table has one series per entity; scale its entity table")
    elif k == "shares":
        if role != "entity" or tab.get("series"):
            errs.append(f"{w}: shares tilt entity tables (for a fact, use intensity or degree)")
        if _ctype(cols.get(col)) not in ("categorical", "boolean"):
            errs.append(f"{w}: column {col!r} must be a categorical or boolean column of {t}")
        has_f, has_s = iv.get("factors") is not None, iv.get("shares") is not None
        if has_f == has_s:
            errs.append(f"{w}: give exactly one of factors {{level: factor}} or shares {{level: share}}")
        elif has_f and not (isinstance(iv["factors"], dict) and iv["factors"]
                            and all(_num(v, 0) for v in iv["factors"].values())):
            errs.append(f"{w}: factors must map levels to numbers >= 0")
        elif has_s and not (isinstance(iv["shares"], dict) and iv["shares"]
                            and all(_num(v, 0) and v <= 1 for v in iv["shares"].values())
                            and sum(iv["shares"].values()) <= 1 + 1e-9):
            errs.append(f"{w}: shares must map levels to numbers in [0, 1] that sum to at most 1")
        prev = targets.get(t)
        if prev is not None and (has_s or prev == "shares"):
            errs.append(f"{w}: {t} has target shares; they cannot be combined with another tilt of the table")
        targets.setdefault(t, "shares" if has_s else "factors")
    elif k == "marginal":
        ty = _ctype(cols.get(col))
        if ty not in _NUM + _TIME:
            errs.append(f"{w}: column {col!r} must be a number, date or datetime column of {t}")
        keys = {tab.get("primary_key")} if isinstance(tab.get("primary_key"), str) else set(tab.get("primary_key") or [])
        for r in rels:
            for side, kc in (("child", "child_key"), ("parent", "parent_key")):
                if r.get(side) == t:
                    keys |= {r[kc]} if isinstance(r.get(kc), str) else set(r.get(kc) or [])
        if col in keys:
            errs.append(f"{w}: {t}.{col} is a key; a marginal map would break the links that use it")
        if tab.get("series"):
            errs.append(f"{w}: series values are changed with intensity")
        sq = tab.get("sequence") or {}
        if isinstance(sq, dict) and sq.get("time") == col:
            errs.append(f"{w}: {t}.{col} is the sequence time, drawn by the sequence model; use intensity")
        ops = [o for o in ("multiply", "add", "quantiles") if iv.get(o) is not None]
        if not ops:
            errs.append(f"{w}: give multiply, add or quantiles")
        if ty in _TIME and (iv.get("multiply") is not None or iv.get("quantiles") is not None):
            errs.append(f"{w}: a date column takes add (days) only")
        if iv.get("multiply") is not None and not _num(iv["multiply"], 0):
            errs.append(f"{w}: multiply must be a number >= 0")
        if iv.get("add") is not None and not _num(iv["add"]):
            errs.append(f"{w}: add must be a number")
        q = iv.get("quantiles")
        if q is not None:
            ok = isinstance(q, list) and len(q) >= 1 and all(
                isinstance(p, (list, tuple)) and len(p) == 2 and _num(p[0], 0) and p[0] <= 1 and _num(p[1], 0)
                for p in q)
            if not ok or sorted(p[0] for p in q) != [p[0] for p in q]:
                errs.append(f"{w}: quantiles must be [[level, factor], ...] with levels in [0, 1] increasing and "
                            "factors >= 0")
    elif k == "intensity":
        tc = col or _time_column(tab)
        if tc is None:
            errs.append(f"{w}: give column (the time column of {t})")
        elif _ctype(cols.get(tc)) not in _TIME and not (
                isinstance(cols.get(tc), dict) and cols[tc].get("type") == "lookup" and tab.get("sequence")):
            errs.append(f"{w}: column {tc!r} must be a date or datetime column of {t}")
        if not _num(iv.get("factor"), 0):
            errs.append(f"{w}: factor must be a number >= 0")
        a, b = _days(iv.get("start")), _days(iv.get("end"))
        if not (np.isfinite(a) and np.isfinite(b) and a < b):
            errs.append(f"{w}: start and end must be dates with start < end")
        if role == "entity" and not tab.get("sequence"):
            if targets.get(t) == "shares":
                errs.append(f"{w}: {t} has target shares; they cannot be combined with another tilt of the table")
            targets.setdefault(t, "factors")
    elif k == "degree":
        key = iv.get("key")
        if not any(r.get("child") == t and r.get("child_key") == key and r.get("kind", "many_to_one") ==
                   "many_to_one" for r in rels):
            errs.append(f"{w}: key {key!r} must be a foreign key column of {t}")
        c, r = iv.get("concentration"), iv.get("rate")
        if c is None and r is None:
            errs.append(f"{w}: give concentration and/or rate")
        if c is not None and not _num(c, 0):
            errs.append(f"{w}: concentration must be a number >= 0")
        if r is not None and (not _num(r, 0, strict=True) or role != "fact"):
            errs.append(f"{w}: rate must be a number > 0, on a fact table")
    return errs


# --------------------------------------------------------------- the scenario
def active(cfg, name=None):
    """(name, block) of the scenario to sample, or None."""
    name = cfg.get("scenario") if name is None else name
    if name is None:
        return None
    sc = cfg.get("scenarios") or {}
    if name not in sc:
        raise SpecError(f"scenario {name!r} is not defined in synthesis.scenarios ({', '.join(sc) or 'none'})")
    return name, sc[name]


class Scenario:
    """A scenario resolved against a fitted synthesizer, with the hooks the
    engine calls while it samples a dataset and the record of each intervention."""

    def __init__(self, S, name, block):
        self.name = name
        self.description = block.get("description")
        T = S.spec["tables"]
        self.scale_ent, self.fact_rate = {}, {}
        self.tilt, self.target = {}, {}
        self.maps = {}
        self._mapped: set[tuple[str, str]] = set()
        self.fact_windows = {}
        self.seq_windows = {}
        self.series_windows = {}
        self.temper_w = {}
        self.records = []
        for iv in block["interventions"]:
            rec = {k: (str(v) if k in ("start", "end") else copy.deepcopy(v)) for k, v in iv.items()}
            rec["effect"] = {}
            self.records.append(rec)
            k, t = iv["type"], iv["table"]
            if t not in T and t not in getattr(S, "series_spec", {}):
                t = self._header_of(S, iv)
            if k == "scale":
                if T[t]["role"] == "fact":
                    self.fact_rate.setdefault(t, []).append((float(iv["factor"]), rec))
                else:
                    self.scale_ent[t] = self.scale_ent.get(t, 1.0) * float(iv["factor"])
                    rec["effect"]["applied_to"] = t
            elif k == "shares":
                self._check_tiltable(S, t)
                col = iv["column"]
                if iv.get("shares") is not None:
                    self.target[t] = (col, {str(a): float(b) for a, b in iv["shares"].items()}, rec)
                else:
                    f = {str(a): float(b) for a, b in iv["factors"].items()}
                    self.tilt.setdefault(t, []).append(("level", col, f, rec))
            elif k == "marginal":
                tt = self._column_table(S, t, iv["column"])
                self.maps.setdefault(tt, []).append((iv["column"], iv, rec))
            elif k == "intensity":
                self._intensity(S, t, iv, rec)
            elif k == "degree":
                if iv.get("concentration") is not None:
                    tk = self._split_table(S, t, iv["key"])
                    self.temper_w[(tk, iv["key"])] = (float(iv["concentration"]), rec)
                if iv.get("rate") is not None:
                    self.fact_rate.setdefault(t, []).append((float(iv["rate"]), rec))
        self._seq_counts(S)
        both = sorted(set(self.target) & set(self.tilt))
        if both:
            # target shares fix the table's rows exactly, so no other tilt can act on them
            raise SpecError(f"scenario {name}: {', '.join(both)} has target shares; they cannot be combined with "
                            "another tilt of the table (factors or a date window)")

    def _seq_counts(self, S):
        """One integrated gain per sequence table, over the combined windows (what
        tilt_sequence applies), raising its count: the fact's rate or the entity's scale."""
        for t, wins in self.seq_windows.items():
            g = self._seq_gain(S.seq[t]["model"], [(a, b, f) for a, b, f, _r in wins])
            for *_w, rec in wins:
                rec["effect"]["count_factor"] = round(g, 6)
                if len(wins) > 1:
                    rec["effect"]["count_factor_of"] = "all intensity windows of the table together"
            if S.spec["tables"][t]["role"] == "fact":
                self.fact_rate.setdefault(t, []).append((g, None))
            else:
                self.scale_ent[t] = self.scale_ent.get(t, 1.0) * g

    # -------------------------------------------------------- resolution
    @staticmethod
    def _header_of(S, iv):
        """An intervention on a fact split into header and lines (fact.header)
        refers to the internal header table when the column lives there."""
        for h in S.plan.headers.values():
            if iv.get("column") in S.spec["tables"][h["table"]]["columns"]:
                return h["table"]
        raise SpecError(f"scenario {iv['type']}: table {iv['table']} is not generated by this model")

    @staticmethod
    def _split_table(S, t, col):
        """A fact split into header and lines (fact.header) generates the header's
        columns (order date, customer key, sequence) in the internal header table."""
        h = S.plan.headers.get(t)
        if col is not None and h is not None and col in S.spec["tables"][h["table"]]["columns"]:
            return h["table"]       # the fact keeps the moved columns only as lookups
        return t

    @staticmethod
    def _column_table(S, t, col):
        tt = Scenario._split_table(S, t, col)
        if col in S.spec["tables"][tt]["columns"]:
            return tt
        raise SpecError(f"scenario: {t}.{col} is not a generated column")

    @staticmethod
    def _check_tiltable(S, e):
        if e not in S.entities or e in S.copied or e in S.grain or \
                any(i["rel"].kind == "tree" and i["rel"].e1 is S.E[e] for i in S.edge_info.values()):
            raise SpecError(f"scenario: {e} cannot be tilted (only entity tables that are neither copied, a "
                            "self-referencing tree nor a unique-key grain)")

    @staticmethod
    def _check_drawn(df, e, col):
        """A tilt (shares, or an intensity window on an entity) accepts rows by a column the table model draws with the
        row; a column filled later (a dimension group's level table, a lookup, a
        computed column) is not there when rows are accepted."""
        if col not in df.columns:
            raise SpecError(f"scenario tilt: {e}.{col} is not drawn by the table model (it comes from a "
                            "dimension group, a lookup or an expression); tilt a column the table draws")

    def _intensity(self, S, t, iv, rec):
        a, b, f = _days(iv["start"]), _days(iv["end"]), float(iv["factor"])
        if t in getattr(S, "series_spec", {}):
            self.series_windows.setdefault(t, []).append((a, b, f, rec))
            return
        t = self._split_table(S, t, iv.get("column") or _time_column(S.spec["tables"][t]))
        T = S.spec["tables"][t]
        col = iv.get("column") or _time_column(T)
        if col is None or col not in T["columns"]:
            raise SpecError(f"scenario intensity: {iv['table']} has no generated time column {col!r}")
        seq = getattr(S, "seq", {}).get(t)
        if seq is not None and seq["time"] == col:
            # the count follows from all windows of the table together (_seq_counts):
            # integrated gains of separate windows do not multiply
            self.seq_windows.setdefault(t, []).append((a, b, f, rec))
            return
        if T["role"] == "fact":
            self.fact_windows.setdefault(t, []).append((col, a, b, f, rec))
        else:
            self._check_tiltable(S, t)
            self.tilt.setdefault(t, []).append(("window", col, (a, b, f), rec))

    # ------------------------------------------------------------- scale
    def apply_scale(self, sc_raw, plan):
        """The scale section with the scenario's entity factors (children follow
        the factor of their parent, as with synthesis.scale)."""
        if not self.scale_ent:
            return plan.scale(sc_raw)
        base = plan.scale(sc_raw)
        raw = {"default": sc_raw.get("default", 1.0), "tables": dict(sc_raw.get("tables", {}) or {})}
        for e, f in self.scale_ent.items():
            raw["tables"][e] = base["tables"].get(e, base["default"]) * f
        out = plan.scale(raw)
        for rec in self.records:
            if rec["type"] == "scale" and rec["effect"].get("applied_to") in out["tables"]:
                rec["effect"]["scale"] = round(float(out["tables"][rec["effect"]["applied_to"]]), 6)
        return out

    # ------------------------------------------------------ entity tilts
    def has_tilt(self, e):
        return e in self.tilt or e in self.target

    def _weights(self, e, df):
        w = np.ones(len(df))
        for kind, col, arg, _rec in self.tilt.get(e, []):
            if kind == "level":
                from .timeseries.hooks import _state_keys
                self._check_drawn(df, e, col)
                keys = _state_keys(df[col].to_numpy(dtype=object))
                fac = np.array([arg.get(k, 1.0) for k in keys]) if len(keys) else np.zeros(0)
                w = w * fac
            else:
                from .timeseries.calendar import to_days
                self._check_drawn(df, e, col)
                a, b, f = arg
                d = to_days(df[col])
                w = w * np.where((d >= a) & (d < b), f, 1.0)
        return w

    def _bound(self, e, df):
        """Product over the tilts of e of each tilt's largest weight: every named
        level's factor (seen in df or not, so a rare level with a large factor is
        covered), and the weight 1 of the levels not named when df shows one."""
        b = 1.0
        for kind, col, arg, _rec in self.tilt.get(e, []):
            if not len(df):
                continue
            if kind == "level":
                from .timeseries.hooks import _state_keys
                seen = set(_state_keys(df[col].to_numpy(dtype=object)).tolist())
                b *= max(list(arg.values()) + [1.0] * bool(seen - set(arg)))
            else:
                from .timeseries.calendar import to_days
                d = to_days(df[col])
                inw = (d >= arg[0]) & (d < arg[1])
                b *= max([arg[2]] * bool(inw.any()) + [1.0] * bool((~inw).any()))
        return b

    def entity_rows(self, S, e, n0, rng):
        """(z, rows) of entity e under its tilt: rows of the fitted model accepted
        with probability w / max w (Prop. 1825), n0 rows for a level tilt (shares
        change, the total stays), n0 E[w] when a window tilt changes the rate."""
        ent = S.E[e]
        p = ent.pi / ent.pi.sum()
        batch = max(int(n0), 200)
        got_z, got_df = [], []
        drawn = 0
        if e in self.target:
            return self._target_rows(S, e, n0, rng, p, batch)
        tilts = self.tilt[e]
        window = any(k == "window" for k, *_ in tilts)
        need = None
        wmax = None
        while drawn < MAX_POOL * max(n0, 1):
            zb = rng.permutation(np.repeat(np.arange(len(p)), rng.multinomial(batch, p)))
            db = S.tm[e].sample(zb, rng)
            wb = self._weights(e, db)
            drawn += batch
            if need is None:
                # the rejection bound: the product of each tilt's largest weight on the
                # rows the model draws (levels and window sides it supports), so a common
                # scaling of the factors leaves the acceptance rate unchanged
                wmax = self._bound(e, db)
                mw = float(wb.mean()) if len(wb) else 1.0
                if window:
                    # the rate changes in the window: E[w] over the window tilts only
                    ww = np.ones(len(db))
                    from .timeseries.calendar import to_days
                    for k, col, arg, _r in tilts:
                        if k == "window":
                            d = to_days(db[col])
                            ww *= np.where((d >= arg[0]) & (d < arg[1]), arg[2], 1.0)
                    need = int(round(n0 * float(ww.mean())))
                else:
                    need = int(n0)
                self._eff_mean_w = mw
            if len(wb) and float(wb.max()) > wmax:
                # a level not named and too rare for the first batch: the rows accepted
                # under the smaller bound are discarded, so every kept row has the same law
                wmax = float(wb.max())
                got_z, got_df = [], []
            acc = rng.random(len(wb)) * max(wmax, 1e-300) < wb
            got_z.append(zb[acc])
            got_df.append(db[acc])
            if sum(len(x) for x in got_z) >= need:
                break
        z = np.concatenate(got_z)[:need] if got_z else np.zeros(0, np.int64)
        df = pd.concat(got_df, ignore_index=True).iloc[:need].reset_index(drop=True) if got_df else \
            S.tm[e].sample(np.zeros(0, np.int64), rng)
        for _k, _c, _a, rec in tilts:
            rec["effect"].update({"rows_without": int(n0), "rows": int(len(z)), "rows_drawn": int(drawn),
                                  "shortfall": int(max(need - len(z), 0))})
        return z, df

    def _target_rows(self, S, e, n0, rng, p, batch):
        """Target shares: quotas per level by largest remainder (levels not named
        share the rest in proportion to the model), each filled with the first
        rows of that level drawn from the model (stratified rejection)."""
        from .timeseries.hooks import _state_keys
        col, shares, rec = self.target[e]
        zb = rng.permutation(np.repeat(np.arange(len(p)), rng.multinomial(max(batch, 2000), p)))
        db = S.tm[e].sample(zb, rng)
        self._check_drawn(db, e, col)
        keys = _state_keys(db[col].to_numpy(dtype=object))
        lv, cnt = np.unique(keys, return_counts=True)
        base = dict(zip(lv.tolist(), (cnt / cnt.sum()).tolist()))
        rest = [k for k in base if k not in shares]
        free = max(1.0 - sum(shares.values()), 0.0)
        tot_rest = sum(base[k] for k in rest)
        if free > 1e-9 and tot_rest <= 0:
            raise SpecError(f"scenario shares: {e}.{col} shares sum to {1 - free:.4g}, and no other level the "
                            f"model draws can take the remaining {free:.4g}; give shares that sum to 1")
        target = dict(shares)
        for k in rest:
            target[k] = free * base[k] / tot_rest if tot_rest > 0 else 0.0
        names = list(target)
        pr = np.array([target[k] for k in names], float)
        pr = pr / pr.sum() if pr.sum() > 0 else np.full(len(pr), 1 / len(pr))
        raw = pr * n0
        q = np.floor(raw).astype(int)
        q[np.argsort(-(raw - q))[: int(n0 - q.sum())]] += 1
        quota = dict(zip(names, q.tolist()))
        # streaming selection: from each batch only the rows a level still needs are
        # kept (the first rows of each level, in draw order), so memory stays O(n0)
        got_z, got_df, drawn = [], [], 0
        have = dict.fromkeys(names, 0)

        def keep(zb, db, kk):
            pick = []
            for k in names:
                idx = np.flatnonzero(kk == k)[: max(quota[k] - have[k], 0)]
                have[k] += len(idx)
                pick.append(idx)
            idx = np.sort(np.concatenate(pick)) if pick else np.zeros(0, np.int64)
            got_z.append(zb[idx])
            got_df.append(db.iloc[idx])

        keep(zb, db, keys)
        drawn += len(zb)
        while any(have[k] < quota[k] for k in names) and drawn < MAX_POOL * max(n0, 1):
            zb = rng.permutation(np.repeat(np.arange(len(p)), rng.multinomial(max(batch, 2000), p)))
            db = S.tm[e].sample(zb, rng)
            keep(zb, db, _state_keys(db[col].to_numpy(dtype=object)))
            drawn += len(zb)
        Z = np.concatenate(got_z)
        D = pd.concat(got_df, ignore_index=True)
        sel = rng.permutation(len(Z))
        rec["effect"].update({"rows": int(len(sel)), "rows_drawn": int(drawn),
                              "shares_without": {k: round(v, 4) for k, v in base.items()},
                              "shares": {k: round(quota[k] / max(n0, 1), 4) for k in names},
                              "shortfall": int(n0 - len(sel))})
        return Z[sel], D.iloc[sel].reset_index(drop=True)

    # ------------------------------------------------------- marginals
    def map_columns(self, t, df, S=None, out=None):
        """Monotone maps of number and date columns (Prop. 1826).  With the engine S
        (rows that are already complete), the derive rules reading a mapped column
        are computed again and rows breaking a check rule are counted."""
        done = []
        for col, iv, rec in self.maps.get(t, []):
            if col not in df.columns or not len(df):
                continue
            df[col] = _map_column(df[col], iv, rec)
            _keep_hidden(df, col)
            self._mapped.add((t, col))
            done.append((col, rec))
        if S is not None and done:
            df, redone, bad = S._after_map(t, df, [c for c, _r in done], out if out is not None else {})
            for _c, rec in done:
                rec["effect"]["recomputed"] = sorted(set(rec["effect"].get("recomputed", [])) | set(redone))
                rec["effect"]["rule_violations"] = int(rec["effect"].get("rule_violations", 0)) + bad
        return df

    def map_derived(self, out, S=None):
        """Maps of columns that are derived when the rows are complete (ratio_of and
        offset_from columns): applied once to the finished table, with the same
        monotone rearrangement; the derive rules reading them follow (with S)."""
        for t, maps in self.maps.items():
            df = out.get(t)
            if df is None or not len(df):
                continue
            todo = [(col, iv, rec) for col, iv, rec in maps if (t, col) not in self._mapped and col in df.columns]
            for col, iv, rec in todo:
                df[col] = _map_column(df[col], iv, rec)
                _keep_hidden(df, col)
                rec["effect"]["applied"] = "to the derived column, after the rows were complete"
                self._mapped.add((t, col))
            if S is not None and todo:
                df, redone, bad = S._after_map(t, df, [c for c, _i, _r in todo], out)
                for _c, _i, rec in todo:
                    rec["effect"]["recomputed"] = sorted(set(rec["effect"].get("recomputed", [])) | set(redone))
                    rec["effect"]["rule_violations"] = int(rec["effect"].get("rule_violations", 0)) + bad
                out[t] = df
        for t, maps in self.maps.items():
            for col, _iv, rec in maps:
                if (t, col) not in self._mapped:
                    rec["effect"]["note"] = f"{t}.{col} was not generated in this dataset; nothing was mapped"

    # ---------------------------------------------------- fact rates
    def has_fact(self, f):
        return f in self.fact_rate or f in self.fact_windows

    def note_fact_rows(self, f, n):
        """Rows of fact f after a partitioned run joined its partitions (the workers'
        own records stay in their processes)."""
        for _g, rec in self.fact_rate.get(f, []):
            if rec is not None:             # None: the count factor of sequence windows
                rec["effect"]["rows"] = int(n)
        for w in self.fact_windows.get(f, []):
            w[-1]["effect"]["rows"] = int(n)

    def fact_copies(self, f):
        """How many independent structure draws are superposed (ceil of the largest
        rate multiplier), and the rate multiplier that is not row-dependent."""
        c = float(np.prod([g for g, _r in self.fact_rate.get(f, [])])) if f in self.fact_rate else 1.0
        wmax = float(np.prod([max(fac, 1.0) for _col, _a, _b, fac, _r in self.fact_windows.get(f, [])]))
        return max(1, int(np.ceil(c * wmax - 1e-12))), c

    def thin_fact(self, f, df, rng, min_one=False):
        """Keep each row of K superposed draws with probability c w(t) / K
        (Prop. 1827).  With mandatory participation (min_one: every anchor has at
        least one row) an anchor left with none keeps one of its rows, chosen with
        probability proportional to its keep probability.  Returns the kept rows."""
        if not len(df):
            return df
        K, c = self.fact_copies(f)
        p = np.full(len(df), c / K)
        if f in self.fact_windows:
            from .timeseries.calendar import to_days
            for col, a, b, fac, _r in self.fact_windows[f]:
                d = to_days(df[col]) if col in df.columns else np.full(len(df), np.nan)
                p = p * np.where((d >= a) & (d < b), fac, 1.0)
        keep = rng.random(len(df)) < np.minimum(p, 1.0)
        if min_one and "__anchor_row__" in df.columns and not keep.all():
            keep = _keep_one_per_anchor(df["__anchor_row__"].to_numpy(), keep, p, rng)
        for _g, rec in self.fact_rate.get(f, []):
            if rec is not None:
                rec["effect"].setdefault("rows", 0)
                rec["effect"]["rows"] += int(keep.sum())
                rec["effect"]["copies"] = K
        for col, a, b, _fac, rec in self.fact_windows.get(f, []):
            from .timeseries.calendar import to_days
            d = to_days(df[col]) if col in df.columns else np.full(len(df), np.nan)
            inw = (d >= a) & (d < b)
            rec["effect"].setdefault("rows_in_window", 0)
            rec["effect"].setdefault("rows", 0)
            rec["effect"]["rows_in_window"] += int((keep & inw).sum())
            rec["effect"]["rows"] += int(keep.sum())
            rec["effect"]["copies"] = K
        return df[keep].reset_index(drop=True)

    # ------------------------------------------------------ link degrees
    def note_degree_law(self, t, key):
        """The key's degrees follow a fitted link-degree law (fact.link_degree), which
        deals each segment's rows by its own count law, not by the degree weights: a
        concentration change cannot act on them, and the record says so."""
        hit = self.temper_w.get((t, key))
        if hit is not None:
            hit[1]["effect"]["note"] = (
                f"the degrees of {t}.{key} follow the fitted link-degree law (fact.link_degree), which does not "
                "use the degree weights, so concentration has no effect; set fact.link_degree: segment on this "
                "fact to let the scenario reshape them")
            hit[1]["effect"]["applied"] = False

    def temper(self, t, key, w, z, K):
        """Degree weights w' proportional to w^c within each segment, rescaled to the
        segment's total (Prop. 1828)."""
        hit = self.temper_w.get((t, key))
        if hit is None:
            return w
        if w is None or not len(w):
            hit[1]["effect"].setdefault("note", "the fit found no degree heterogeneity for this key "
                                                 "(facts.frailty off), so there is nothing to temper")
            return w
        c, rec = hit
        w = np.asarray(w, dtype=float)
        pos = w > 0
        v = np.zeros(len(w))
        if c == 0:
            v[pos] = 1.0
        else:
            lw = np.full(len(w), -np.inf)
            lw[pos] = c * np.log(w[pos])
            mx = np.full(K, -np.inf)
            np.maximum.at(mx, z, lw)
            v[pos] = np.exp(lw[pos] - mx[z[pos]])
        tot = np.bincount(z, weights=w, minlength=K)
        new = np.bincount(z, weights=v, minlength=K)
        out = np.where(new[z] > 0, v * tot[z] / np.maximum(new[z], 1e-300), 0.0)
        rec["effect"]["gini_without"] = round(_gini(w), 4)
        rec["effect"]["gini"] = round(_gini(out), 4)
        if np.allclose(w[pos], 1.0):
            rec["effect"]["note"] = ("the fit found no degree heterogeneity within segments for this key "
                                     "(facts.frailty off), so there is nothing to temper")
        return out

    # -------------------------------------------------------- sequences
    @staticmethod
    def _seq_gain(m, windows):
        """Integrated intensity change of a sequence model over its time range."""
        I = m.intensity
        before = I.cell_mass.sum(1)
        mass = _tilted_mass(I, windows)
        w = I.total.sum(1) if getattr(I, "total", None) is not None else np.ones(I.K)
        g = mass.sum(1) / np.maximum(before, 1e-300)
        return float(np.sum(g * w) / max(np.sum(w), 1e-300))

    def tilt_sequence(self, t, m):
        """Context: the sequence model's calendar intensity multiplied on the windows."""
        return _Tilt(m.intensity, [(a, b, f) for a, b, f, _r in self.seq_windows.get(t, [])])

    # ---------------------------------------------------------- series
    def apply_series(self, t, df, time_col, values):
        from .timeseries.calendar import to_days
        if t not in self.series_windows or not len(df):
            return df
        d = to_days(df[time_col])
        for a, b, f, rec in self.series_windows[t]:
            inw = (d >= a) & (d < b)
            for c in values:
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float, copy=True)
                    v[inw] = v[inw] * f
                    if pd.api.types.is_integer_dtype(df[c].dtype):
                        v = np.round(v)       # an integer value column stays integer (rounded)
                    try:
                        df[c] = pd.Series(v, index=df.index).astype(df[c].dtype)
                    except (TypeError, ValueError):
                        df[c] = v
            rec["effect"]["rows_in_window"] = int(inw.sum())
        return df

    # ------------------------------------------------------------- log
    def log(self):
        return {"name": self.name, "description": self.description, "interventions": self.records}


def _gini(w):
    w = np.sort(np.asarray(w, float))
    if not len(w) or w.sum() <= 0:
        return 0.0
    n = len(w)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(w) / (n * w.sum()))


def _keep_one_per_anchor(anchor, keep, p, rng):
    """keep with one row added for every anchor that has rows but none kept: the
    row with the largest p * (a uniform), i.e. a draw proportional to p (Efraimidis
    and Spirakis 2006, one item)."""
    has = np.zeros(int(anchor.max()) + 1 if len(anchor) else 0, bool)
    has[anchor[keep]] = True
    lost = ~has[anchor]
    if not lost.any():
        return keep
    idx = np.flatnonzero(lost)
    key = np.log(rng.random(len(idx))) / np.maximum(p[idx], 1e-300)    # larger is chosen
    o = np.lexsort((-key, anchor[idx]))
    first = np.r_[True, anchor[idx][o][1:] != anchor[idx][o][:-1]]
    keep = keep.copy()
    keep[idx[o][first]] = True
    return keep


def _keep_hidden(df, col):
    """A column drawn from a stated distribution keeps its draw in a hidden column
    (engine._apply_distributions) that later recomputations copy back: the mapped
    value goes there too, so it is the one kept."""
    hc = f"__dist_{col}"
    if hc in df.columns:
        df[hc] = df[col].to_numpy(dtype=object) if df[hc].dtype == object else df[col].to_numpy()


def _tilted_mass(I, windows):
    """Cell masses with the intensity multiplied on the windows: each cell's mass
    times the integral over the cell of the pointwise product of the window factors
    (mass uniform within a cell), so overlapping or adjacent windows combine exactly."""
    mass = I.cell_mass.copy()
    n = mass.shape[1]
    ws = []
    for a, b, f in windows:
        x0, x1 = max((a - I.d0) / I.width, 0.0), min((b - I.d0) / I.width, float(n))
        if x1 > x0:
            ws.append((x0, x1, float(f)))
    if not ws:
        return np.maximum(mass, 1e-12)
    lo, hi = int(np.floor(min(w[0] for w in ws))), int(np.ceil(max(w[1] for w in ws)))
    pts = np.unique(np.r_[np.arange(lo, hi + 1, dtype=float), [w[0] for w in ws], [w[1] for w in ws]])
    mid, ln = (pts[:-1] + pts[1:]) / 2, np.diff(pts)
    fac = np.ones(len(mid))
    for x0, x1, f in ws:
        fac *= np.where((mid >= x0) & (mid < x1), f, 1.0)
    gain = np.bincount(np.floor(mid).astype(np.int64) - lo, weights=ln * fac, minlength=hi - lo)
    mass[:, lo:hi] *= gain[None, :]
    return np.maximum(mass, 1e-12)


class _Tilt:
    """Multiply a calendar intensity on date windows and restore it afterwards."""

    def __init__(self, I, windows):
        self.I, self.windows = I, windows

    def __enter__(self):
        I = self.I
        self.saved = (I.cell_mass, I.C)
        if self.windows:
            I.cell_mass = _tilted_mass(I, self.windows)
            I.C = np.concatenate([np.zeros((I.K, 1)), np.cumsum(I.cell_mass, 1)], 1)
        return self

    def __exit__(self, *exc):
        self.I.cell_mass, self.I.C = self.saved
        return False


def _map_column(col, iv, rec):
    """x' = a x + b or x f(p), then the monotone rearrangement (ranks kept)."""
    from .timeseries.calendar import from_days, to_days
    s = pd.Series(col)
    is_time = str(s.dtype).startswith("datetime") or (s.dtype == object and len(s.dropna()) and
                                                      not isinstance(s.dropna().iloc[0], (int, float, np.number)))
    if is_time:
        d = to_days(s)
        ok = np.isfinite(d)
        d2 = d + float(iv.get("add") or 0.0)
        out = s.copy()
        vals = pd.Series(from_days(d2[ok], True))
        if str(s.dtype).startswith("datetime"):
            out.loc[ok] = vals.to_numpy().astype(s.dtype)
        else:
            first = s.dropna().iloc[0]
            if isinstance(first, str):
                fmt = "%Y-%m-%d %H:%M:%S" if any(":" in str(v) for v in s.dropna().iloc[:20]) else "%Y-%m-%d"
                out.loc[ok] = pd.to_datetime(vals).dt.strftime(fmt).to_numpy(dtype=object)
            else:
                out.loc[ok] = [x.date() if not hasattr(first, "hour") else x.to_pydatetime()
                               for x in pd.to_datetime(vals)]
        rec["effect"]["shift_days"] = float(iv.get("add") or 0.0)
        return out
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x)
    y = x.copy()
    if ok.any():
        xv = x[ok]
        if iv.get("quantiles") is not None:
            q = np.array(iv["quantiles"], dtype=float)
            r = (pd.Series(xv).rank(method="average").to_numpy() - 0.5) / len(xv)
            yv = xv * np.interp(r, q[:, 0], q[:, 1])
        else:
            yv = xv.copy()
        if iv.get("multiply") is not None:
            yv = yv * float(iv["multiply"])
        if iv.get("add") is not None:
            yv = yv + float(iv["add"])
        # monotone rearrangement: the sorted new values in the order of the old ones
        order = np.argsort(xv, kind="stable")
        out = np.empty(len(xv))
        out[order] = np.sort(yv)
        y[ok] = out
        rec["effect"].setdefault("mean_without", 0.0)
        rec["effect"]["mean_without"] = round(float(xv.mean()), 6)
        rec["effect"]["mean"] = round(float(out.mean()), 6)
    if pd.api.types.is_integer_dtype(s.dtype) or (s.dtype == object and ok.any() and
                                                  np.allclose(x[ok], np.round(x[ok]))):
        y = np.round(y)
        res = pd.Series(y, index=s.index)
        try:
            return res.astype(s.dtype)
        except (TypeError, ValueError):
            return res
    return pd.Series(y, index=s.index)
