"""v0.9 checks of the evaluator's scorecard calibration (relsynth/evaluate.py).

Each test returns a list of {"what", "value", "target", "pass"} items (as run_checks.py).
Run all of them with
    python3 -P checks/v09_eval.py            or one:  python3 -P checks/v09_eval.py t_join_z
Null cases draw the synthetic data from the law of the real data (or a bootstrap of a
real Chinook table) at the real sizes and measure the false-failure rate of the
scorecard limit; power cases show that real defects still fail.  Simulation sizes are
kept small (minutes in total), so rate targets carry Monte Carlo slack.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RELSYNTH_QUIET", "1")
warnings.filterwarnings("ignore")

E = __import__("importlib").import_module("relsynth.evaluate")  # the module (rs.evaluate is the API function)

CHINOOK = "/tmp/claude-0/-home-claude/bb3ce53c-120b-56d9-8782-51605abd2883/scratchpad/chinook/Chinook.db"


def item(what, value, target, ok):
    if isinstance(value, (float, np.floating)):
        value = round(float(value), 4)
    return {"what": what, "value": value, "target": target, "pass": bool(ok)}


def _empty(**kw):
    rep = {"tables": {}, "relations": {}, "trees": {}, "time": {}, "text": {}, "privacy": {}, "rules": {},
           "grain": {}, "removed": {}}
    rep.update(kw)
    return rep


def _items(rep, metric):
    return [i for i in E.scorecard(rep)["items"] if i["metric"] == metric]


_CH = {}


def _chinook():
    if "spec" not in _CH:
        import yaml
        from relsynth.data import load_tables
        from relsynth.spec import load_spec
        raw = yaml.safe_load(open(ROOT / "realdata/chinook/spec.yaml"))
        raw["source"] = {"kind": "sqlite", "path": CHINOOK}
        spec = load_spec(raw)
        _CH["spec"], _CH["tables"] = spec, load_tables(spec)
    return _CH["spec"], _CH["tables"]


def _model_cols(spec, t):
    from relsynth.marginals import MODELLED
    ts = spec["tables"][t]
    keys = {(r["child"], c) for r in spec["relationships"]
            for c in (r["child_key"] if r.get("composite") else [r["child_key"]])}
    return [c for c, cs in ts["columns"].items() if cs["type"] in MODELLED and (t, c) not in keys
            and c != ts.get("primary_key")]


# ------------------------------------------------------------------ identities (performance rewrites)
def t_identity_fast_paths():
    rng = np.random.default_rng(5)
    n = 2000
    df = pd.DataFrame({"x": rng.normal(size=n), "i": rng.poisson(3, n), "c": rng.choice(list("abcdefg"), n),
                       "d": pd.Series(rng.choice(["u", "v", None, "w"], n), dtype=object), "b": rng.random(n) < 0.3,
                       "t": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(900, size=n), unit="D")})
    df.loc[rng.random(n) < 0.1, "x"] = np.nan
    ts = {"columns": {"x": {"type": "float"}, "i": {"type": "integer"}, "c": {"type": "categorical"},
                      "d": {"type": "categorical"}, "b": {"type": "boolean"}, "t": {"type": "datetime"}}}
    cols = list(ts["columns"])
    iscat = {c: E._is_cat(ts["columns"][c], df[c]) for c in cols}
    mx = 0.0
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            o = E._assoc(df[a], df[b], iscat[a], iscat[b])
            f = E._assoc_fast(E._prep_col(df, a, iscat[a]), E._prep_col(df, b, iscat[b]), iscat[a], iscat[b])
            mx = max(mx, abs(o - f))
    # dcr nearest neighbours against brute force
    d = pd.DataFrame({"c": rng.integers(5, size=400).astype(str), "x": rng.normal(size=400).round(1),
                      "y": rng.integers(3, size=400)})
    d = pd.concat([d, d.iloc[:40]], ignore_index=True)
    s = pd.concat([d.iloc[:30], pd.DataFrame({"c": rng.integers(5, size=300).astype(str),
                                              "x": rng.normal(size=300).round(1), "y": rng.integers(3, size=300)})],
                  ignore_index=True)
    tsd = {"columns": {"c": {"type": "categorical"}, "x": {"type": "float"}, "y": {"type": "integer"}}}
    o = E.dcr(d, s, tsd, ["c", "x", "y"])
    X = E._design([d.sample(len(d), random_state=0)[["c", "x", "y"]], s.sample(len(s), random_state=0)[["c", "x", "y"]]],
                  tsd, ["c", "x", "y"])[:, :-1]
    Xr, Xs = X[:len(d)], X[len(d):]
    ds = np.sqrt(((Xs[:, None] - Xr[None]) ** 2).sum(-1).min(1))
    D = ((Xr[:, None] - Xr[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    dr = np.sqrt(D.min(1))
    if np.percentile(dr, 5) == 0 and (dr > 0).any() and (ds > 0).any():
        dr, ds = dr[dr > 0], ds[ds > 0]
    return [item("assoc fast path vs _assoc, max |difference| over 15 pairs", mx, "<= 1e-12", mx <= 1e-12),
            item("dcr p5 (syn, real) by matrix product vs brute force",
                 [o["dcr_p5_syn"], o["dcr_p5_real"]], [round(float(np.percentile(ds, 5)), 6), round(float(np.percentile(dr, 5)), 6)],
                 np.isclose(o["dcr_p5_syn"], np.percentile(ds, 5), atol=1e-9)
                 and np.isclose(o["dcr_p5_real"], np.percentile(dr, 5), atol=1e-9))]


# ------------------------------------------------------------------ discriminator
def t_auc():
    spec, T = _chinook()
    rng = np.random.default_rng(1)
    df, ts = T["Customer"], spec["tables"]["Customer"]
    cols = _model_cols(spec, "Customer")
    f = []
    for _ in range(120):
        r, s = (df.iloc[rng.integers(len(df), size=len(df))].reset_index(drop=True) for _ in range(2))
        tabs = {f"t{j}": {"rows_real": 1, "rows_syn": 1, "columns": {}, "association": {"max_abs_diff": 0.0},
                          "discriminator_auc": 0.5, "discriminator_n": 1000, "discriminator_real_ok": True} for j in range(7)}
        tabs["Customer"] = {"rows_real": 59, "rows_syn": 59, "columns": {}, "association": {"max_abs_diff": 0.0},
                            "discriminator_auc": E.discriminator_auc(r, s, ts, cols), "discriminator_n": 59,
                            "discriminator_real_ok": True}
        f += [not i["pass"] for i in _items(_empty(tables=tabs), "discriminator_auc") if i["area"] == "Customer"]
    il = T["InvoiceLine"]   # Quantity is constant: no information at all
    a_const = E.discriminator_auc(il, il.sample(frac=1, random_state=4), spec["tables"]["InvoiceLine"], ["Quantity"])
    tr = T["Track"]
    a_tr = np.mean([E.discriminator_auc(tr.iloc[rng.integers(len(tr), size=len(tr))], tr.iloc[rng.integers(len(tr), size=len(tr))],
                                        spec["tables"]["Track"], _model_cols(spec, "Track")) for _ in range(8)])
    # power: a shifted numeric column on 59 rows
    r = pd.DataFrame({"x": rng.normal(size=59)})
    s = pd.DataFrame({"x": rng.normal(size=59) + 1.0})
    a_pow = E.discriminator_auc(r, s, {"columns": {"x": {"type": "float"}}}, ["x"])
    lim59 = max(0.65, 0.5 + E._sidak_z(0.01, 8, two_sided=False) * E.auc_null_sd(59))
    return [item("Customer (59 rows) null false-failure rate, limit Sidak over 8 tables", np.mean(f), "<= 0.03",
                 np.mean(f) <= 0.03),
            item("AUC with a constant feature (was 0.487)", a_const, "== 0.5", abs(a_const - 0.5) < 1e-12),
            item("Track (3503 rows) null mean AUC over 8 draws (was 0.484)", a_tr, "0.5 +- 0.006", abs(a_tr - 0.5) <= 0.006),
            item("1-sd shift on 59 rows: AUC above the 59-row limit", a_pow, f"> {lim59:.3f}", a_pow > lim59)]


# ------------------------------------------------------------------ joins
def _join_db(rng, nanc, mean_rows, heavy, shift=0.0):
    NA, NC = 12, 4
    Z = rng.normal(size=(nanc, NA))
    u = rng.normal(size=nanc)
    A = 0.3 * u[:, None] + np.sqrt(1 - 0.09) * Z
    cust = {"cid": np.arange(nanc)}
    for j in range(NA):
        cust[f"a{j}"] = np.digitize(A[:, j], [-0.8, 0, 0.8]).astype(str) if j % 2 else A[:, j]
    w = rng.pareto(1.2, nanc) + 1 if heavy else np.ones(nanc)
    cid = np.repeat(np.arange(nanc), rng.poisson(mean_rows * w / w.mean()))
    o = {"oid": np.arange(len(cid)), "cid": cid}
    for k in range(NC):
        o[f"m{k}"] = np.sqrt(0.5) * u[cid] + np.sqrt(0.5) * rng.normal(size=len(cid)) + shift * A[cid, 0] * (k == 0)
    return {"Cust": pd.DataFrame(cust), "Ord": pd.DataFrame(o)}


_JSPEC = {"tables": {
    "Cust": {"role": "entity", "primary_key": "cid",
             "columns": {"cid": {"type": "id"}, **{f"a{j}": {"type": "categorical" if j % 2 else "float"} for j in range(12)}}},
    "Ord": {"role": "fact", "primary_key": "oid", "fact": {"anchor": "Cust"},
            "columns": {"oid": {"type": "id"}, "cid": {"type": "id"}, **{f"m{k}": {"type": "float"} for k in range(4)}}}},
    "relationships": [{"name": "Ord.cid->Cust", "child": "Ord", "parent": "Cust", "child_key": "cid", "parent_key": "cid",
                       "kind": "many_to_one"}]}


def t_join_z():
    rng = np.random.default_rng(2)
    out = []
    for nanc, heavy, old in ((200, True, "27.5%"), (2000, True, "90%")):
        f = []
        for _ in range(40):
            j = E.join_metrics(_JSPEC, _join_db(rng, nanc, 10, heavy), _join_db(rng, nanc, 10, heavy), "Ord")
            f += [not i["pass"] for i in _items(_empty(relations={"Ord.joined": j}), "join_assoc_max_z")]
        out.append(item(f"48 cells, {nanc} ancestors, skewed rows per ancestor: null false-failure rate (was {old})",
                        np.mean(f), "<= 0.05", np.mean(f) <= 0.05))
    j = E.join_metrics(_JSPEC, _join_db(rng, 2000, 10, False), _join_db(rng, 2000, 10, False, shift=0.6), "Ord")
    it = _items(_empty(relations={"Ord.joined": j}), "join_assoc_max_z")[0]
    out.append(item("measure that depends on an ancestor attribute in the synthetic data only: fails",
                    it["value"], f"> {it['limit']}", not it["pass"]))
    return out


# ------------------------------------------------------------------ cardinality
def _rel(rc, sc, rp, sp, ratio):
    d = E.cardinality(rc, sc, "fk", rp, sp, "id", ratio)
    d.update(fk_integrity=1.0, null_fk_real=float(rc.fk.isna().mean()), null_fk_syn=float(sc.fk.isna().mean()),
             rows_child_real=len(rc), rows_child_syn=len(sc), null_fk_count_real=int(rc.fk.isna().sum()),
             null_fk_count_syn=int(sc.fk.isna().sum()))
    return d


def _counts(rng, P, lam, ratio=1.0, pnull=0.0):
    c = rng.poisson(lam * ratio * rng.gamma(2, .5, P))
    ch = pd.DataFrame({"fk": np.repeat(np.arange(P), c).astype(object)})
    ch.loc[rng.random(len(ch)) < pnull, "fk"] = None
    return ch, pd.DataFrame({"id": np.arange(P)})


def t_cardinality():
    rng = np.random.default_rng(3)
    out = []
    for P, lam, ratio, pnull, old in ((500, 0.6, 2.0, 0.0, "ks 100%"), (500, 3, 1.15, 0.0, "ks 86%"),
                                      (100, 3, 1.0, 0.1, "mean 6.7%"), (606, 2, 1.0, 0.345, "null_fk 26%")):
        f = {"cardinality_ks": [], "cardinality_mean_rel_err": [], "null_fk_diff": []}
        for _ in range(150):
            rc, rp = _counts(rng, P, lam, 1, pnull)
            sc, sp = _counts(rng, P, lam, ratio, pnull)
            for i in E.scorecard(_empty(relations={"x": _rel(rc, sc, rp, sp, ratio)}))["items"]:
                if i["metric"] in f:
                    f[i["metric"]].append(not i["pass"])
        rates = {k: round(float(np.mean(v)), 3) for k, v in f.items()}
        out.append(item(f"null, {P} parents, mean {lam}, ratio {ratio}, null FK {pnull} (old {old})", rates,
                        "each <= 0.04", max(rates.values()) <= 0.04))
    # power: Chinook InvoiceLine per track (counts 0/1/2) against a Poisson count model
    if Path(CHINOOK).exists():
        spec, T = _chinook()
        rc = T["InvoiceLine"]["TrackId"].value_counts().reindex(T["Track"]["TrackId"], fill_value=0).to_numpy(float)
        null = [E.ks(rc, rng.choice(rc, len(rc))) for _ in range(200)]
        pois = E.ks(rc, rng.poisson(rc.mean(), len(rc)).astype(float))
        lim = max(E.THRESHOLDS["card_ks"], E.ks_crit(len(rc), len(rc), pp=True))
        out += [item("Chinook InvoiceLine.TrackId: null q99 of KS (same count law)", float(np.quantile(null, .99)), "< 0.03",
                     np.quantile(null, .99) < 0.03),
                item("Poisson count model vs real counts: KS (misfit, 4x the null q99)", pois, "> 3 * null q99",
                     pois > 3 * np.quantile(null, .99)),
                item("seed-4 value 0.1202 against the new limit", 0.1202, f"> {lim:.4f} (fails)", 0.1202 > lim)]
    return out


def t_entity_child_ratio():
    from relsynth.spec import normalise
    rng = np.random.default_rng(4)
    spec = normalise({"tables": {
        "P": {"role": "entity", "primary_key": "pid", "columns": {"pid": {"type": "id"}, "g": {"type": "categorical"}}},
        "C": {"role": "entity", "primary_key": "cid", "columns": {"cid": {"type": "id"}, "pid": {"type": "id"}, "v": {"type": "float"}}}},
        "relationships": [{"name": "C.pid->P", "child": "C", "parent": "P", "child_key": "pid", "parent_key": "pid",
                           "kind": "many_to_one"}]})

    def db(nc):
        return {"P": pd.DataFrame({"pid": np.arange(200), "g": rng.choice(list("ab"), 200)}),
                "C": pd.DataFrame({"cid": np.arange(nc), "pid": rng.integers(200, size=nc), "v": rng.normal(size=nc)})}
    real = db(1000)
    res = {}
    for k in (1, 3):
        rep = E.evaluate(spec, real, db(1000 * k), privacy=False)
        res[k] = [i for i in rep["scorecard"]["items"] if i["metric"] == "cardinality_mean_rel_err"][0]
    return [item("entity child at the planned size: mean_rel_err passes", res[1]["value"], f"<= {res[1]['limit']}", res[1]["pass"]),
            item("entity child 3x too many rows per parent: fails (was 0.0 and passed)", res[3]["value"],
                 f"> {res[3]['limit']}", not res[3]["pass"])]


# ------------------------------------------------------------------ columns
def t_null_rates():
    rng = np.random.default_rng(6)
    ts = {"columns": {"c": {"type": "float"}}}
    out = []
    for pn, old in ((0.01, "6.5%"), (0.02, "11.1%")):
        f = []
        for _ in range(800):
            r = pd.DataFrame({"c": np.where(rng.random(59) < pn, np.nan, rng.normal(size=59))})
            s = pd.DataFrame({"c": np.where(rng.random(59) < pn, np.nan, rng.normal(size=59))})
            m = E.column_metrics(r, s, ts, ["c"])["c"]
            f.append(m["null_diff"] > max(0.03, m["null_crit"]))
        out.append(item(f"null share {pn}, 59 rows: false-failure rate (was {old})", np.mean(f), "<= 0.02", np.mean(f) <= 0.02))
    r = pd.DataFrame({"c": rng.normal(size=500)})
    s = pd.DataFrame({"c": np.where(rng.random(500) < 0.15, np.nan, rng.normal(size=500))})
    m = E.column_metrics(r, s, ts, ["c"])["c"]
    out.append(item("15% nulls added in the synthetic column (500 rows): fails", m["null_diff"],
                    f"> {max(0.03, m['null_crit']):.4f}", m["null_diff"] > max(0.03, m["null_crit"])))
    return out


def t_association_selection():
    rng = np.random.default_rng(7)
    real = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000),
                         "c": np.r_[rng.normal(size=100), [np.nan] * 900]})
    syn = real.copy()
    want = {("a", "b"): 0.149, ("a", "c"): 0.44, ("b", "c"): 0.0}
    calls = []
    orig_fast, orig = E._assoc_fast, E._assoc

    def patched(x, y, cx, cy):          # real association 0, synthetic association want[pair]
        side = len(calls) % 2
        calls.append(1)
        return want[(x.name, y.name)] if side else 0.0
    ts = {"columns": {c: {"type": "float"} for c in "abc"}}
    E._assoc_fast, E._assoc = (lambda A, B, cx, cy: None), patched
    try:
        a = E.assoc_metrics(real, syn, ts, list("abc"))
    finally:
        E._assoc_fast, E._assoc = orig_fast, orig
    rep = _empty(tables={"X": {"rows_real": 1000, "rows_syn": 1000, "columns": {}, "association": a,
                               "discriminator_auc": float("nan")}})
    it = _items(rep, "association_max_diff")[0]
    return [item("pair a~c differs by 0.44 beyond its own limit while a~b sits under the 0.15 floor: scorecard fails",
                 f"{a['worst_pair']} {it['value']} vs {it['limit']}", "fail on a~c", not it["pass"] and a["worst_pair"] == "a~c")]


def t_unevaluable_synthetic():
    rng = np.random.default_rng(8)
    ts = {"columns": {"x": {"type": "float"}, "g": {"type": "categorical"}}}
    real = pd.DataFrame({"x": rng.normal(size=100), "g": rng.choice(list("ab"), 100)})
    syn = pd.DataFrame({"x": [np.nan] * 100, "g": rng.choice(list("ab"), 100)})
    d = {"rows_real": 100, "rows_syn": 100, "columns": E.column_metrics(real, syn, ts, ["x", "g"]),
         "association": {"max_abs_diff": 0.0}, "discriminator_auc": E.discriminator_auc(real, syn.iloc[:10], ts, ["x", "g"]),
         "discriminator_n": 10, "discriminator_real_ok": True}
    its = E.scorecard(_empty(tables={"T": d}))["items"]
    ks_it = [i for i in its if i["metric"] == "marginal_ks"]
    auc_it = [i for i in its if i["metric"] == "discriminator_auc"]
    rule = _items(_empty(rules={"r": {"kind": "check", "violation_rate_real": 0.0, "violation_rate_syn": float("nan")}}),
                  "rule_violation_rate")
    return [item("all-null synthetic column: marginal_ks recorded as a failure", [i["pass"] for i in ks_it], "[False]",
                 ks_it and not ks_it[0]["pass"]),
            item("synthetic table of 10 rows: discriminator recorded as a failure", [i["pass"] for i in auc_it], "[False]",
                 auc_it and not auc_it[0]["pass"]),
            item("rule that cannot be evaluated on the synthetic side: failure", [i["pass"] for i in rule], "[False]",
                 rule and not rule[0]["pass"])]


# ------------------------------------------------------------------ grain
_GSPEC = {"tables": {"A": {"role": "entity", "primary_key": "aid", "columns": {"aid": {"type": "id"}}},
                     "B": {"role": "entity", "primary_key": "bid", "columns": {"bid": {"type": "id"}}},
                     "F": {"role": "fact", "primary_key": "fid", "fact": {"anchor": "A", "partner": "B"},
                           "columns": {"fid": {"type": "id"}, "aid": {"type": "id"}, "bid": {"type": "id"}}}},
          "relationships": [{"name": "F.aid->A", "child": "F", "parent": "A", "child_key": "aid", "parent_key": "aid",
                             "kind": "many_to_one"},
                            {"name": "F.bid->B", "child": "F", "parent": "B", "child_key": "bid", "parent_key": "bid",
                             "kind": "many_to_one"}], "rules": []}


def t_grain():
    rng = np.random.default_rng(9)

    def fdb(na, lam, nb):
        aid = np.repeat(np.arange(na), rng.poisson(lam, na))
        pb = 1 / np.arange(1, nb + 1)
        return {"A": pd.DataFrame({"aid": np.arange(na)}), "B": pd.DataFrame({"bid": np.arange(nb)}),
                "F": pd.DataFrame({"fid": np.arange(len(aid)), "aid": aid, "bid": rng.choice(nb, len(aid), p=pb / pb.sum())})}
    f = [not i["pass"] for _ in range(150) for i in _items(_empty(grain=E.grain_metrics(_GSPEC, fdb(100, 3, 30), fdb(100, 3, 30))),
                                                          "pair_repeat_share_diff")]
    gs = {**_GSPEC, "tables": {**_GSPEC["tables"], "A": {**_GSPEC["tables"]["A"], "mode": "copy"},
                               "B": {**_GSPEC["tables"]["B"], "mode": "copy"},
                               "F": {**_GSPEC["tables"]["F"], "fact": {}}}}
    pa = 1 / np.arange(1, 21)
    pa /= pa.sum()

    def sdb(n, novel=0.0):
        a, b = rng.choice(20, n, p=pa), rng.choice(20, n, p=pa)
        k = rng.random(n) < novel
        a[k], b[k] = 25, 25          # a combination the real data never has
        return {"A": pd.DataFrame({"aid": np.arange(26)}), "B": pd.DataFrame({"bid": np.arange(26)}),
                "F": pd.DataFrame({"fid": np.arange(n), "aid": a, "bid": b})}
    g = [not i["pass"] for _ in range(150) for i in _items(_empty(grain=E.grain_metrics(gs, sdb(4000), sdb(4000))),
                                                          "novel_combination_rate")]
    pw = _items(_empty(grain=E.grain_metrics(gs, sdb(4000), sdb(4000, novel=0.08))), "novel_combination_rate")[0]
    return [item("pair repeat share, 100 anchors: null false-failure rate (was 4.7%)", np.mean(f), "<= 0.03", np.mean(f) <= 0.03),
            item("novel combinations, Zipf 20x20, 4000 rows: null false-failure rate (was 100%)", np.mean(g), "<= 0.04",
                 np.mean(g) <= 0.04),
            item("8% of synthetic rows on a never-seen combination: fails", pw["value"], f"> {pw['limit']}", not pw["pass"])]


# ------------------------------------------------------------------ privacy
def t_privacy():
    rng = np.random.default_rng(10)
    law = lambda n: pd.DataFrame({"c1": rng.integers(10, size=n), "c2": rng.integers(12, size=n),  # noqa: E731
                                  "c3": rng.integers(15, size=n)}).astype(str)
    coarse = lambda n: pd.DataFrame({"c1": rng.integers(4, size=n), "c2": rng.integers(5, size=n),  # noqa: E731
                                     "c3": rng.integers(6, size=n)}).astype(str)
    tsp = {"columns": {c: {"type": "categorical"} for c in ("c1", "c2", "c3")}}
    cols = ["c1", "c2", "c3"]
    fe = [not i["pass"] for _ in range(80) for i in _items(_empty(privacy={"X": E.dcr(law(300), law(300), tsp, cols)}),
                                                          "exact_match_rate")]
    o = E.dcr(coarse(1500), coarse(1500), tsp, cols)
    fd = _items(_empty(privacy={"X": o}), "dcr_ratio")[0]
    tsn = {"columns": {"x": {"type": "float"}, "y": {"type": "float"}, "z": {"type": "float"}}}
    r = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("xyz"))
    s = pd.concat([r.iloc[:100], pd.DataFrame(rng.normal(size=(400, 3)), columns=list("xyz"))], ignore_index=True)
    pw = _items(_empty(privacy={"X": E.dcr(r, s, tsn, list("xyz"))}), "exact_match_rate")[0]
    return [item("exact-match rate, 1800-cell categorical space, 300 rows: null false-failure rate (was 38%)", np.mean(fe),
                 "<= 0.06", np.mean(fe) <= 0.06),
            item("coarse space where every row has a twin: dcr_ratio (was 0 and failed)", fd["value"], ">= 0.5", fd["pass"]),
            item("20% of synthetic rows copied from real continuous rows: exact_match fails", pw["value"], f"> {pw['limit']}",
                 not pw["pass"])]


# ------------------------------------------------------------------ time, soft rules, trees, text
_TSPEC = {"tables": {"C": {"role": "entity", "primary_key": "cid", "columns": {"cid": {"type": "id"}, "seg": {"type": "categorical"}}},
                     "O": {"role": "fact", "primary_key": "oid", "fact": {"anchor": "C", "time_column": "ts", "time_bucket": "month"},
                           "columns": {"oid": {"type": "id"}, "cid": {"type": "id"}, "ts": {"type": "datetime"}}}},
          "relationships": [{"name": "O.cid->C", "child": "O", "parent": "C", "child_key": "cid", "parent_key": "cid",
                             "kind": "many_to_one"}]}


def t_time():
    rng = np.random.default_rng(11)

    def tdb(nc, months, mean, start="2020-01-01", span=None):
        cid = np.repeat(np.arange(nc), rng.poisson(mean, nc))
        ts = pd.Timestamp(start) + pd.to_timedelta(rng.integers((span or months) * 30, size=len(cid)), unit="D")
        return {"C": pd.DataFrame({"cid": np.arange(nc), "seg": rng.integers(3, size=nc).astype(str)}),
                "O": pd.DataFrame({"oid": np.arange(len(cid)), "cid": cid, "ts": ts})}
    f = [not i["pass"] for _ in range(80) for i in _items(_empty(time={"O": E.time_metrics(_TSPEC, tdb(60, 36, 7), tdb(60, 36, 7), "O")}),
                                                         "time_bucket_tvd")]
    pw = _items(_empty(time={"O": E.time_metrics(_TSPEC, tdb(3000, 36, 5), tdb(3000, 36, 5, span=18), "O")}), "time_bucket_tvd")[0]
    return [item("420 rows over 36 months (60 anchors): null false-failure rate (was 100%)", np.mean(f), "<= 0.03", np.mean(f) <= 0.03),
            item("synthetic times squeezed into the first half of the period: fails", pw["value"], f"> {pw['limit']}", not pw["pass"])]


def t_soft_pair_ppc():
    rng = np.random.default_rng(12)
    f = []
    for r0, n in ((0.1, 100), (0.02, 1000)):
        for _ in range(1500):
            x, y = rng.binomial(n, r0) / n, rng.binomial(n, r0) / n
            f += [not i["pass"] for i in _items(_empty(rules={"r": {"kind": "soft", "violation_rate_real": x, "violation_rate_syn": y,
                                                                     "rows_real": n, "rows_syn": n}}), "soft_rule_rate_diff")]
    pw = _items(_empty(rules={"r": {"kind": "soft", "violation_rate_real": 0.1, "violation_rate_syn": 0.0,
                                    "rows_real": 5000, "rows_syn": 5000}}), "soft_rule_rate_diff")[0]
    from relsynth.relational import tree_depth

    def tree(N):
        par = np.array([-1] + [int(rng.integers(i)) if rng.random() < 0.8 else -1 for i in range(1, N)])
        return pd.DataFrame({"id": np.arange(N), "boss": pd.array([x if x >= 0 else None for x in par], dtype=object)}), par
    g = []
    for _ in range(250):
        real, _ = tree(60)
        ppc = [np.bincount(tree_depth(tree(60)[1].copy())).tolist() for _ in range(40)]
        g += [not i["pass"] for i in _items(_empty(trees={"t": E.tree_metrics(real, real, "id", "boss", [], ppc)}),
                                            "tree_depth_ppc_pvalue")]
    return [item("soft rule (r0 0.1 on 100 rows, 0.02 on 1000): null false-failure rate (was 56% / 11%)", np.mean(f), "<= 0.025",
                 np.mean(f) <= 0.025),
            item("soft rule never broken in the synthetic data (source 10%, 5000 rows): fails", pw["value"], f"> {pw['limit']}",
                 not pw["pass"]),
            item("tree depth PPC, 60 nodes, 40 draws: null failure rate (nominal 0.05)", np.mean(g), "<= 0.08", np.mean(g) <= 0.08)]


def t_text_tokens():
    from relsynth.text import TextModel
    rng = np.random.default_rng(13)
    real = [f"Model{rng.integers(10**5, 10**6)} works well" for _ in range(30)] + ["Model alone"] + ["works well indeed"] * 5
    tm = TextModel(order=2, min_token_rows=3).fit(real)
    syn = tm.generate([None] * 200, np.random.default_rng(1))
    m = E.text_metrics(pd.DataFrame({"t": real}), pd.DataFrame({"t": syn}), "t", None, k=3, L=8)
    real2 = real + ["Zyxwv secret phrase"]
    m2 = E.text_metrics(pd.DataFrame({"t": real2}), pd.DataFrame({"t": syn + ["Zyxwv again"]}), "t", None, k=3, L=8)
    return [item("generator's own k-anonymous vocabulary ('Model' common once ids are scrubbed): leaks (was 166/200)",
                 m["rare_token_leaks"], "0", m["rare_token_leaks"] == 0),
            item("a token held by 1 source row copied into a synthetic text: still a leak", m2["rare_token_leaks"], ">= 1",
                 m2["rare_token_leaks"] >= 1)]


def t_chinook_scorecard():
    if not Path(CHINOOK).exists():
        return [item("Chinook database available", False, True, False)]
    from relsynth.data import coerce
    spec, R = _chinook()
    d = ROOT / "realdata/chinook/synthetic_seed1"
    S = {t: coerce(pd.read_csv(d / f"{t}.csv"), spec["tables"][t]) for t in R}
    t0 = time.time()
    sc = E.evaluate(spec, R, S)["scorecard"]
    fails = sorted(f"{i['area']}:{i['metric']}" for i in sc["failed"])
    return [item("Chinook seed-1 scorecard runs; failed items", fails, "listed", True),
            item("expected false failures reported", sc["expected_false_failures"], "0 < e < 2", 0 < sc["expected_false_failures"] < 2),
            item("evaluate() seconds", round(time.time() - t0, 1), "< 30", time.time() - t0 < 30)]


TESTS = [t_identity_fast_paths, t_auc, t_join_z, t_cardinality, t_entity_child_ratio, t_null_rates, t_association_selection,
         t_unevaluable_synthetic, t_grain, t_privacy, t_time, t_soft_pair_ppc, t_text_tokens, t_chinook_scorecard]


def main(argv):
    sel = [t for t in TESTS if not argv or t.__name__ in argv]
    bad = 0
    for t in sel:
        t0 = time.time()
        try:
            items = t()
        except Exception as e:                      # noqa: BLE001
            import traceback
            items = [item(f"{type(e).__name__}: {e}", traceback.format_exc().splitlines()[-3:], None, False)]
        ok = all(i["pass"] for i in items)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
        for i in items:
            print(f"   {'ok ' if i['pass'] else 'BAD'} {i['what']}: {i['value']} (target {i['target']})", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
