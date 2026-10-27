# SPDX-License-Identifier: Apache-2.0
"""The 2.0 quality report: every check with its definition, null limit and
confidence level, grouped in a quality vector (fidelity, utility, integrity).

    from relsynth.quality import assess
    rep = assess(spec, real_tables, synthetic_tables, holdout=None, config=None)

`rep` validates against relsynth/quality/schema/quality-report-2.0.schema.json.
The 1.0 scorecard (relsynth.evaluate) is carried over unchanged: its items keep
their limits and outcomes and are placed in an area; the new metrics add checks
with limits calibrated on real-versus-real splits (relsynth.quality.calibrate).
There is no single overall score: the vector shows each area's checks, failures,
expected false failures and the calibrated number of failures it tolerates.
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
from scipy.stats import ks_2samp

from .._version import __version__
from . import calibrate as cal
from . import multi as mt
from . import single as st
from . import utility as ut
from .encode import attribute_columns, freeze_kinds
from .registry import AREAS, METRICS, SCORECARD_AREAS

SCHEMA_ID = "https://github.com/nick-developer/relsynth/schemas/quality-report-2.0.schema.json"
SCHEMA_VERSION = "2.0.0"

DEFAULTS = {
    "confidence": 0.99,           # level of each calibrated check
    "area_confidence": 0.99,      # level of each area's pass rule
    "null_splits": 20,            # real-vs-real splits for single-table metrics
    "multi_splits": 10,           # splits for multi-table and utility metrics
    "max_rows": 20000,            # rows per table for column and marginal metrics
    "max_rows_model": 5000,       # rows per side for precision/recall, detection and utility
    "k3_max": 50,                 # 3-way marginals evaluated per table
    "max_hops": 2,
    "classifier": "logistic",     # aggregation detection: logistic | xgboost
    "targets": {},                # {table: column} utility targets
    "mt_targets": {},             # {parent table: column} multi-table utility targets
    "tolerances": {},             # {metric id: floor}
    "calibrate": True,            # False: tolerances only (fast; no null limits)
    "scale": None,                # expected synthetic/real size: a number or {default, tables}; None = 1
    "seed": 0,
}


def _cap(df, n, seed):
    return df.sample(n, random_state=seed).reset_index(drop=True) if len(df) > n else df.reset_index(drop=True)


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def poisson_binomial_quantile(alphas, q):
    """Smallest k with P(F <= k) >= q for F a sum of independent Bernoulli(alpha_i)."""
    pmf = np.array([1.0])
    for a in alphas:
        if a <= 0:
            continue
        pmf = np.r_[pmf * (1 - a), 0.0] + np.r_[0.0, pmf * a]
    return int(np.searchsorted(np.cumsum(pmf), q - 1e-12))


class _Builder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.checks = []
        self.details = {}

    def tol(self, metric):
        return float(self.cfg["tolerances"].get(metric, METRICS[metric]["tolerance"]))

    def add(self, metric, scope, value, draws=None, sizes=None, split_sizes=None, note=None):
        """One calibrated check: limit = max(tolerance, null prediction bound)."""
        m = METRICS[metric]
        conf = self.cfg["confidence"]
        tol = self.tol(metric)
        null, alpha = None, None
        log = m["center"] == 0
        if draws is not None and sizes is not None and split_sizes is not None:
            k = cal.kappa(sizes, split_sizes)
            null = cal.null_summary(draws, m["center"], conf, k, log)
            if null["limit"] is not None:
                alpha = cal.exceed_prob(draws, max(tol, null["limit"]), m["center"], k, log)
        limit = tol if null is None or null["limit"] is None else max(tol, null["limit"])
        v = _f(value)
        ok = v is not None and v <= limit
        p_value = None
        if not ok and v is not None and alpha is not None and null["method"].endswith("(log scale)"):
            # how far out the failure is: the null probability of a value this large (Prop. 111).
            # Only for distances calibrated on the log scale: signed gaps (TSTR) have skewed nulls
            # whose normal tail is too thin for p-values near 1e-4 (C79: a false severe failure)
            p_value = cal.exceed_prob(draws, v, m["center"], k, log)
        it = {"id": f"{metric}:{':'.join(str(x) for x in scope.values())}", "metric": metric, "area": m["area"],
              "scope": scope, "value": None if v is None else round(v, 6), "limit": round(limit, 6),
              "tolerance": tol, "direction": m["direction"], "pass": bool(ok),
              "alpha": None if alpha is None else round(alpha, 6), "confidence": conf,
              **({"p_value": float(f"{p_value:.3g}")} if p_value is not None else {}),
              "null": null or {"method": "fixed tolerance", "splits": 0}, "source": "quality-2.0"}
        if v is None:
            it["note"] = "synthetic side could not be evaluated"
        if note:
            it["note"] = note
        self.checks.append(it)


def _cap_idx(n, cap, seed):
    return np.sort(np.random.default_rng(seed).permutation(n)[:cap]) if n > cap else np.arange(n)


def _table_checks(b, spec, real, syn, holdout, t, rng):
    cfg = b.cfg
    seed = cfg["seed"]
    cols = attribute_columns(spec, t)
    if not cols or spec["tables"][t].get("mode") == "copy" or len(real[t]) < 20 or not len(syn[t]):
        return
    ix = _cap_idx(len(real[t]), cfg["max_rows"], seed)
    r, s = real[t].iloc[ix].reset_index(drop=True), _cap(syn[t], cfg["max_rows"], seed)
    ids = cal.cluster_ids(spec, real, t)[ix]
    R = cfg["null_splits"] if cfg["calibrate"] else 0
    masks = [cal.cluster_halves(ids, rng) for _ in range(R)]
    halves = [(r[mk].reset_index(drop=True), r[~mk].reset_index(drop=True)) for mk in masks]
    sizes = (len(r), len(s))
    ssz = (int(np.mean([len(h[0]) for h in halves])), int(np.mean([len(h[1]) for h in halves]))) if R else None
    det = b.details.setdefault(t, {})
    # columns
    cm = st.column_metrics(r, s, spec, t, cols)
    nulls = [st.column_metrics(A, B, spec, t, cols) for A, B in halves]
    det["columns"] = cm
    for c, d in cm.items():
        for metric in ("w1_raw_sd", "w1_rank_normal", "js_distance"):
            if metric in d:
                b.add(metric, {"table": t, "column": c}, d[metric],
                      [n[c][metric] for n in nulls] if R else None, sizes, ssz)
    # k-way marginals
    kw = st.kway_tvd(r, s, spec, t, cols, k3_max=cfg["k3_max"], seed=seed)
    det["kway"] = kw
    if kw:
        kn = [st.kway_tvd(A, B, spec, t, cols, k3_max=cfg["k3_max"], seed=seed) for A, B in halves]
        for metric, key in (("kway2_tvd_mean", "k2_mean"), ("kway2_tvd_max", "k2_max"), ("kway3_tvd_mean", "k3_mean")):
            if key in kw:
                note = f"worst: {kw[key.replace('mean', 'worst').replace('max', 'worst')]}" if key.endswith("max") else None
                b.add(metric, {"table": t}, kw[key], [n.get(key) for n in kn] if R else None, sizes, ssz, note)
    # alpha-precision / beta-recall
    mr = cfg["max_rows_model"]
    pr = st.precision_recall(r, s, spec, t, cols, max_rows=mr, seed=seed)
    det["precision_recall"] = pr
    if pr:
        pn = [st.precision_recall(A, B, spec, t, cols, max_rows=mr, seed=seed) for A, B in halves]
        n = min(len(r), len(s), mr)
        psz = (n, n)
        pss = (min(*ssz, mr),) * 2 if R else None
        for metric in ("alpha_precision_gap", "beta_recall_gap"):
            b.add(metric, {"table": t}, pr[metric], [n.get(metric) for n in pn] if R else None, psz, pss)
    # utility
    target = cfg["targets"].get(t) or ut.pick_target(spec, t, r, cols)
    if target and len(cols) >= 2:
        _utility_checks(b, spec, t, r, s, holdout.get(t) if holdout else None, target, cols, "tstr_gap", "trts_gap",
                        {"table": t}, rng, ids)


def _utility_checks(b, spec, t, r, s, hold, target, cols, m_tstr, m_trts, scope, rng, ids):
    cfg = b.cfg
    mr = cfg["max_rows_model"]
    note = None
    if hold is None:
        u_ids = np.unique(ids)
        test = set(rng.permutation(u_ids)[: int(round(0.3 * len(u_ids)))])
        mk = ~np.isin(ids, list(test))
        train, hold, ids = r[mk].reset_index(drop=True), r[~mk].reset_index(drop=True), ids[mk]
        note = "held-out rows are an internal split of the data the generator saw: TSTR is optimistic"
    else:
        train = r
    u = ut.utility(train, s, hold, spec, t, target, cols, seed=cfg["seed"], max_rows=mr)
    b.details.setdefault(scope.get("table", t), {})[m_tstr.replace("_gap", "")] = u
    R = cfg["multi_splits"] if cfg["calibrate"] else 0
    feats = [c for c in cols if c != target]
    d_tstr, d_trts, hs = [], [], []
    for _ in range(R):
        mk = cal.cluster_halves(ids, rng)
        A, B = train[mk].reset_index(drop=True), train[~mk].reset_index(drop=True)
        hs.append((len(A), len(B)))
        sA = ut.score(train, A, hold, spec, t, target, feats, cfg["seed"], mr)
        d_tstr.append(sA - ut.score(train, B, hold, spec, t, target, feats, cfg["seed"], mr))
        d_trts.append(abs(sA - ut.score(train, A, B, spec, t, target, feats, cfg["seed"], mr)))
    n, m, h = min(len(train), mr), min(len(s), mr), min(len(hold), mr)
    half = (min(int(np.mean([h[0] for h in hs])), mr), min(int(np.mean([h[1] for h in hs])), mr)) if R else None
    sc = dict(scope, target=target)
    b.add(m_tstr, sc, u["tstr_gap"], d_tstr if R else None, (n, m), half if R else None, note)
    b.add(m_trts, sc, u["trts_gap"], d_trts if R else None, (h, m), (h, half[1]) if R else None, note)


def _relationship_checks(b, spec, real, syn, rng):
    cfg = b.cfg
    R = cfg["multi_splits"] if cfg["calibrate"] else 0
    bench = {}
    for name, p, pk, c, fk in mt.relations(spec):
        if not len(real[p]) or not len(syn[p]):
            continue
        cr = mt.counts_per_parent(real[p], pk, real[c], fk)
        f = fanout_factor(spec, cfg, p, c)
        cs = mt.counts_per_parent(syn[p], pk, syn[c], fk) / f
        law = mt.fanout_law(cr, cs)
        if not law:
            continue
        law["scale_factor"] = f
        bench[name] = 1 - float(ks_2samp(cr, cs).statistic)
        b.details.setdefault("_relationships", {})[name] = law
        ids = cal.cluster_ids(spec, real, p)
        nulls, hs = [], []
        for _ in range(R):
            mk = cal.cluster_halves(ids, rng)
            nulls.append(mt.fanout_law(cr[mk], cr[~mk]))
            hs.append((mk.sum(), (~mk).sum()))
        sizes = (len(cr), len(cs))
        ssz = (int(np.mean([h[0] for h in hs])), int(np.mean([h[1] for h in hs]))) if R else None
        # resized tables: counts are compared per unit of the expected rate (Prop. 113); the share of
        # childless parents and the dispersion are not invariant under resizing and are not checked
        keys = (("fanout_w1_rel", "w1_rel_mean"),) if f != 1 else (
            ("fanout_w1_rel", "w1_rel_mean"), ("fanout_zero_share_diff", "zero_share_diff"),
            ("fanout_dispersion_log_ratio", "dispersion_log_ratio"))
        note = f"synthetic counts divided by the scale factor {f:g}" if f != 1 else None
        for metric, key in keys:
            b.add(metric, {"relationship": name}, law[key], [n[key] for n in nulls] if R else None, sizes, ssz, note)
    for name, gp, gpk, p, ppk, fk1, c, fk2 in mt.two_hop_chains(spec):
        if not len(real[gp]) or not len(syn[gp]):
            continue
        cr = mt.grandchild_counts(real, gp, gpk, p, ppk, fk1, c, fk2)
        f = fanout_factor(spec, cfg, gp, c)
        cs = mt.grandchild_counts(syn, gp, gpk, p, ppk, fk1, c, fk2) / f
        law = mt.fanout_law(cr, cs)
        if not law:
            continue
        ids = cal.cluster_ids(spec, real, gp)
        nulls, hs = [], []
        for _ in range(R):
            mk = cal.cluster_halves(ids, rng)
            nulls.append(mt.fanout_law(cr[mk], cr[~mk]))
            hs.append((mk.sum(), (~mk).sum()))
        b.add("fanout2_w1_rel", {"path": name}, law["w1_rel_mean"], [n["w1_rel_mean"] for n in nulls] if R else None,
              (len(cr), len(cs)), (int(np.mean([h[0] for h in hs])), int(np.mean([h[1] for h in hs]))) if R else None,
              f"synthetic counts divided by the scale factor {f:g}" if f != 1 else None)
    return bench


def _path_checks(b, spec, real, syn, rng):
    cfg = b.cfg
    R = cfg["multi_splits"] if cfg["calibrate"] else 0
    ht = mt.hop_trends(spec, real, syn, cfg["max_hops"], cfg["max_rows"], cfg["seed"])
    b.details["_paths"] = ht
    if not ht:
        return {}
    top = {}
    for name, p, _pk, _c, _fk in mt.relations(spec):
        top[name] = p
    for name, gp, *_ in mt.two_hop_chains(spec):
        top[name] = gp
    by_top = {}
    for name in ht:
        by_top.setdefault(top[name], []).append(name)
    nulls = {name: [] for name in ht}
    for T, names in by_top.items():
        for _ in range(R):
            A, B = cal.split_db_root(spec, real, T, rng)
            hn = mt.hop_trends(spec, A, B, cfg["max_hops"], cfg["max_rows"], cfg["seed"])
            for name in names:
                if name in hn:
                    nulls[name].append(hn[name])
    for name, d in ht.items():
        mx = cfg["max_rows"]
        n_r = min(_bottom_rows(spec, real, name), mx)
        n_s = min(_bottom_rows(spec, syn, name), mx)
        ssz = (min(n_r // 2, mx), min(n_r - n_r // 2, mx))
        for metric, key in (("hop_trend_gap", "mean"), ("hop_trend_worst_gap", "min")):
            draws = [1 - n[key] for n in nulls[name]] if R else None
            note = f"worst pair: {d['worst']}" if key == "min" else None
            b.add(metric, {"path": name, "hops": d["hops"]}, 1 - d[key], draws, (n_r, n_s), ssz if R else None, note)
    bench = {}
    for d in ht.values():
        bench.setdefault(f"{d['hops']}-hop", []).append(d["mean"])
    return {k: float(np.mean(v)) for k, v in bench.items()}


def _bottom_rows(spec, tables, path):
    rel = {n: c for n, _p, _pk, c, _fk in mt.relations(spec)}
    return len(tables[rel[path.split(" | ")[-1]]])


def _detection_checks(b, spec, real, syn, holdout, rng):
    cfg = b.cfg
    R = cfg["multi_splits"] if cfg["calibrate"] else 0
    mr = cfg["max_rows_model"]
    bench = {}
    for p in mt.parents_with_children(spec):
        if len(real[p]) < 20 or len(syn[p]) < 20:
            continue
        fr, kinds = mt.aggregated(spec, real, p)
        factors = {name: fanout_factor(spec, cfg, p, c) for name, pp, _k, c, _f in mt.relations(spec) if pp == p}
        fs, _ = mt.aggregated(spec, syn, p, factors)
        acc = mt.detection_accuracy(fr, fs, kinds, cfg["classifier"], max_rows=mr, seed=cfg["seed"])
        bench[p] = acc
        draws = []
        for _ in range(R):
            A, B = cal.split_db_root(spec, real, p, rng)
            draws.append(mt.detection_accuracy(mt.aggregated(spec, A, p)[0], mt.aggregated(spec, B, p)[0], kinds,
                                               cfg["classifier"], max_rows=mr, seed=cfg["seed"]))
        N = min(len(fr), len(fs), mr)
        Nh = min(len(fr) // 2, len(fr) - len(fr) // 2, mr)
        b.add("dda_accuracy", {"table": p, "classifier": cfg["classifier"]}, acc, draws if R else None,
              (N, N), (Nh, Nh) if R else None)
        b.details.setdefault(p, {})["dda"] = {"accuracy": acc, "features": list(kinds), "rows": N}
        # multi-table utility on the aggregated parent table
        sp = mt._as_spec_cols(kinds)
        acols = list(kinds)
        target = cfg["mt_targets"].get(p) or ut.pick_target(sp, "_agg", fr, attribute_columns(spec, p))
        if target and len(acols) >= 2:
            hold = None
            if holdout and p in holdout and all(c in holdout for _n, pp, _k, c, _f in mt.relations(spec) if pp == p):
                hold = mt.aggregated(spec, holdout, p)[0]
            ix = _cap_idx(len(fr), cfg["max_rows"], cfg["seed"])
            _utility_checks(b, sp, "_agg", fr.iloc[ix].reset_index(drop=True), fs, hold, target, acols,
                            "mt_tstr_gap", "mt_trts_gap", {"table": p}, rng, cal.cluster_ids(spec, real, p)[ix])
    return bench


def _scorecard_checks(base):
    out = []
    for i in (base or {}).get("scorecard", {}).get("items", []):
        area = SCORECARD_AREAS.get(i["metric"], "fidelity")
        a = i.get("alpha")
        out.append({"id": f"{i['metric']}:{i['area']}", "metric": i["metric"], "area": area,
                    "scope": {"name": i["area"]}, "value": i["value"], "limit": i["limit"],
                    "tolerance": None, "direction": "higher" if i["metric"] in (
                        "fk_integrity", "dcr_ratio", "text_tstr_ratio", "tree_depth_ppc_pvalue") else "lower",
                    "pass": bool(i["pass"]), "alpha": a, "confidence": None if a is None else round(1 - a, 6),
                    "null": {"method": "scorecard 1.0 (asymptotic null of a posterior predictive draw)"
                             if a is not None else "fixed limit"},
                    "source": "scorecard-1.0", **({"note": i["note"]} if "note" in i else {})})
    return out


def quality_vector(checks, q=0.99):
    """Per area: checks, failures, expected false failures under the null (sum
    of the levels) and the failures tolerated (Prop. 111).  The level 1 - q is
    split in two halves: an area fails when its failures exceed the
    (1 - (1-q)/2)-quantile of the Poisson-binomial law of false failures, or
    when one failed check is severe, its null p-value below (1-q)/(2m) for the
    area's m calibrated checks (Bonferroni).  A check without a level (a fixed
    limit, an identity) tolerates no failure."""
    out = {}
    q_count = 1 - (1 - q) / 2
    for a in AREAS:
        cs = [c for c in checks if c["area"] == a]
        al = [float(c["alpha"]) for c in cs if c.get("alpha")]
        fails = [c for c in cs if not c["pass"]]
        allowed = poisson_binomial_quantile(al, q_count)
        cut = (1 - q) / (2 * len(al)) if al else 0.0
        severe = [c["id"] for c in fails if c.get("p_value") is not None and c["p_value"] < cut]
        out[a] = {"checks": len(cs), "passed": len(cs) - len(fails), "failed": len(fails),
                  "pass_rate": round((len(cs) - len(fails)) / len(cs), 4) if cs else None,
                  "expected_false_failures": round(float(sum(al)), 3), "allowed_failures": allowed,
                  "severe_p_value": float(f"{cut:.3g}"), "severe_failures": severe,
                  "confidence": q, "pass": len(fails) <= allowed and not severe,
                  "failed_checks": [c["id"] for c in fails][:50]}
    out["verdict"] = {"pass": all(out[a]["pass"] for a in AREAS),
                      "rule": "every area passes; no weighted overall score is computed"}
    return out


def resolve_config(config=None):
    """DEFAULTS overridden by `config`; ValueError for an unknown key or value."""
    config = dict(config or {})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **config}
    for k in ("targets", "mt_targets", "tolerances"):
        cfg[k] = dict(cfg.get(k) or {})
    unknown = set(cfg["tolerances"]) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metric ids in tolerances: {sorted(unknown)}")
    if cfg["classifier"] not in ("logistic", "xgboost"):
        raise ValueError("classifier must be 'logistic' or 'xgboost'")
    if not 0.5 < float(cfg["confidence"]) < 1 or not 0.5 < float(cfg["area_confidence"]) < 1:
        raise ValueError("confidence levels must be between 0.5 and 1")
    sc = cfg["scale"]
    if sc is None:
        sc = {"default": 1.0, "tables": {}}
    elif isinstance(sc, (int, float)) and not isinstance(sc, bool):
        sc = {"default": float(sc), "tables": {}}
    elif isinstance(sc, dict) and set(sc) <= {"default", "tables"}:
        sc = {"default": float(sc.get("default", 1.0)), "tables": {str(k): float(v) for k, v in (sc.get("tables") or {}).items()}}
    else:
        raise ValueError("scale must be a number or {default: number, tables: {table: number}}")
    if sc["default"] <= 0 or any(v <= 0 for v in sc["tables"].values()):
        raise ValueError("scale factors must be positive")
    cfg["scale"] = sc
    return cfg


def table_scale(spec, cfg, t, _seen=()):
    """Expected synthetic/real row ratio of table t under cfg['scale'], by the
    engine's scale rule: copied tables keep their rows, a fact table follows
    its anchor (rows per anchor are preserved), any other table takes its own
    factor or the default."""
    T = spec["tables"][t]
    sc = cfg["scale"]
    if T.get("mode") == "copy":
        return 1.0
    if t in sc["tables"]:
        return sc["tables"][t]
    anchor = (T.get("fact") or {}).get("anchor")
    if T.get("role") == "fact" and anchor in spec["tables"] and anchor not in _seen:
        return table_scale(spec, cfg, anchor, _seen + (t,))
    return sc["default"]


def fanout_factor(spec, cfg, parent, child):
    """Expected change of the child rows per parent row when the tables are
    resized (1 when both keep their size ratio)."""
    return table_scale(spec, cfg, child) / table_scale(spec, cfg, parent)


def assess(spec, real, syn, *, holdout=None, base=None, config=None, scorecard=True, privacy=None):
    """The 2.0 quality report of synthetic tables against real ones.

    spec: a resolved spec dict; real, syn: {table: DataFrame}; holdout: real
    tables the generator did not see (for TSTR/TRTS), optional; base: the 1.0
    report from relsynth.evaluate (computed when None and scorecard=True);
    config: overrides of DEFAULTS; privacy: the reference to (or the whole of)
    the privacy report of the same data, copied as is (RFC T06-02)."""
    cfg = resolve_config(config)
    rng = np.random.default_rng(cfg["seed"])
    spec_1 = spec                                   # the 1.0 evaluator keeps its own rules
    spec = freeze_kinds(spec, real)
    b = _Builder(cfg)
    for t in spec["tables"]:
        if t in real and t in syn:
            _table_checks(b, spec, real, syn, holdout, t, rng)
    card = _relationship_checks(b, spec, real, syn, rng)
    hops = _path_checks(b, spec, real, syn, rng)
    dda = _detection_checks(b, spec, real, syn, holdout, rng)
    if base is None and scorecard:
        from ..evaluate import evaluate
        base = evaluate(spec_1, real, syn)
    old = _scorecard_checks(base)
    checks = [c for c in old if c["area"] != "privacy"] + b.checks
    used = sorted({c["metric"] for c in b.checks})
    sc = (base or {}).get("scorecard", {})
    extra = {"privacy": privacy} if privacy is not None else {}
    return {
        "$schema": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "relsynth_version": __version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {k: v for k, v in cfg.items()},
        "data": {"real_rows": {t: int(len(df)) for t, df in real.items()},
                 "synthetic_rows": {t: int(len(df)) for t, df in syn.items()},
                 "holdout": sorted(holdout) if holdout else None},
        "vector": quality_vector(checks, cfg["area_confidence"]),
        "checks": checks,
        "privacy_passthrough": [c for c in old if c["area"] == "privacy"],
        "scorecard_1": {"passed": sc.get("passed"), "total": sc.get("total"),
                        "expected_false_failures": sc.get("expected_false_failures")},
        "benchmark": {"syntherela": {
            "cardinality_similarity": card,
            "k_hop_trend": hops,
            "c2st_agg_accuracy": dda,
            "c2st_agg_accuracy_max": max(dda.values()) if dda else None,
            "classifier": cfg["classifier"],
            "note": "SyntheRela-compatible summaries (1 - KS of per-parent counts; mean SDMetrics pair-trend "
                    "score per hop count; C2ST-Agg accuracy per parent table and its maximum)"}},
        "metrics": {m: {k: v for k, v in METRICS[m].items()} for m in used},
        "details": _clean(b.details),
        **extra,
    }


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _f(x)
    return x
