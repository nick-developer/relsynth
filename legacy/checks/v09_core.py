# SPDX-License-Identifier: Apache-2.0
"""v0.9 checks of the core statistics and the sampling engine.

Each test returns [(what, value, target, ok), ...].  Run all of them with
    python3 -P checks/v09_core.py            or one:  python3 -P checks/v09_core.py t_hidden_draws
The cases are those of the v0.9 review: the DP count marginal underflowing to
-inf, the sparse Dirichlet grid losing mass, structural blocks drawn with fixed
weights (a trickle of free blocks in complete relations), per-row stated draws
redrawn between the rule check and the output, key labels sorted as text, the
affinity windows overlapping, mean mode drawing per-row shares, sampling
warnings leaking into later datasets, an empty parent crashing, nullable keys,
the participation prior of small segments (customers without a sale), and the
identity of the faster kernels with the code they replace.
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("RELSYNTH_QUIET", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fakedata"))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from scipy.special import gammaln       # noqa: E402

from relsynth import Synthesizer, load_spec     # noqa: E402


def _probe(qty_dist=None, chan_dist=None, rules=None, mode=None, seed=66, nc=300, npd=60, ns=2000):
    """Copied customer and product tables and one fact (the review's probe)."""
    rng = np.random.default_rng(seed)
    cust = pd.DataFrame({"ck": np.arange(1, nc + 1), "region": rng.choice(["N", "S", "E", "W"], nc),
                         "age": rng.integers(18, 80, nc)})
    prod = pd.DataFrame({"pk": np.arange(1, npd + 1), "cat": rng.choice(["A", "B", "C"], npd),
                         "price": rng.gamma(3, 20, npd).round(2)})
    sales = pd.DataFrame({"sk": np.arange(1, ns + 1), "ck": rng.choice(cust.ck, ns), "pk": rng.choice(prod.pk, ns),
                          "qty": rng.integers(1, 6, ns), "channel": rng.choice(["web", "shop"], ns)})
    qty = {"type": "integer"} if qty_dist is None else {"type": "integer", "distribution": qty_dist}
    ch = {"type": "categorical"} if chan_dist is None else {"type": "categorical", "distribution": chan_dist}
    spec = {"version": 1, "name": "probe", "source": {"kind": "csv", "path": "."}, "tables": {
        "cust": {"primary_key": "ck", "mode": "copy", "columns": {"ck": "id", "region": "categorical", "age": "integer"}},
        "prod": {"primary_key": "pk", "mode": "copy", "columns": {"pk": "id", "cat": "categorical", "price": "float"}},
        "sales": {"primary_key": "sk", "role": "fact", "fact": {"anchor": "ck", "partner": "pk"},
                  "columns": {"sk": "id", "ck": "id", "pk": "id", "qty": qty, "channel": ch}}},
        "relationships": [{"child": "sales", "child_key": "ck", "parent": "cust"},
                          {"child": "sales", "child_key": "pk", "parent": "prod"}],
        "rules": rules or [], "synthesis": {"seed": 1}}
    if mode:
        spec["synthesis"]["posterior"] = {"mode": mode}
    tables = {"cust": cust, "prod": prod, "sales": sales}
    return Synthesizer(load_spec(spec)).fit(tables), tables


# ---------------------------------------------------------------- core statistics
def _dp_urn(t, a=1.0):
    """Sequential Polya urn (the definition; the v0.8 code)."""
    t = np.asarray(t, np.int64)
    lam = max(float(t.mean()), 1e-12)
    lg0 = t * np.log(lam) - lam - gammaln(t + 1)
    seen, out = {}, 0.0
    for i, x in enumerate(t.tolist()):
        c = seen.get(x, 0)
        out += np.log(a * np.exp(lg0[i]) + c) - np.log(a + i)
        seen[x] = c + 1
    return float(out)


def t_dp_closed_form():
    from relsynth.counts import dp_logmarginal
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        t = rng.poisson(rng.gamma(2, 2), rng.integers(1, 300))
        a = float(rng.choice([0.1, 1.0, 5.0]))
        o, n = _dp_urn(t, a), dp_logmarginal(t, a)
        worst = max(worst, abs(o - n) / max(1.0, abs(o)))
    tail = np.r_[np.full(1000, 7), 400]
    v = dp_logmarginal(tail)
    return [("max relative difference to the sequential urn, 300 cases", f"{worst:.1e}", "<= 1e-10", worst <= 1e-10),
            ("count far in the Poisson tail (400 among 7s): finite marginal", round(v, 2), "finite (urn: -inf)",
             bool(np.isfinite(v)) and _dp_urn(tail) == -np.inf)]


def t_sparse_dirichlet():
    """Number of unobserved levels included: Monte Carlo of the sampler against the
    exact posterior P(|J| = j) propto C(m, j) rho^j Gamma(A_o + j a) / Gamma(A_o + j a + n)."""
    from relsynth.relational import sparse_dirichlet_draw
    out = []
    for counts, alpha, pi_s in [(np.r_[30, 20, np.zeros(8)], 0.5, 0.5), (np.r_[5, np.zeros(12)], 0.05, 0.3),
                                (np.r_[200, 150, 50, np.zeros(6)], 1.0, 0.9)]:
        m = int((counts == 0).sum())
        Ao = alpha * (counts > 0).sum()
        n = counts.sum()
        rho = (1 - pi_s) / pi_s
        j = np.arange(m + 1)
        lp = gammaln(m + 1) - gammaln(j + 1) - gammaln(m - j + 1) + j * np.log(rho) \
            + gammaln(Ao + j * alpha) - gammaln(Ao + j * alpha + n)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        exact = float((j * p).sum())
        sd = float(np.sqrt((j * j * p).sum() - exact ** 2))
        rng = np.random.default_rng(1)
        D = 4000
        inc = np.array([(sparse_dirichlet_draw(counts, np.full(len(counts), alpha), pi_s, rng)[counts == 0] > 0).sum()
                        for _ in range(D)])
        se = sd / np.sqrt(D)
        out.append((f"E|J| (m={m}, alpha={alpha}, pi_s={pi_s}): sampler vs exact", round(float(inc.mean()), 3),
                    f"{exact:.3f} +- {4 * se:.3f}", abs(inc.mean() - exact) <= 4 * se + 1e-9))
    return out


def t_zoib_pooling():
    from relsynth.relational import grouped_zoib_draw, zoib_weights
    rng = np.random.default_rng(0)
    n1 = np.array([40, 1, 1, 39, 39, 1, 39, 1, 13, 27, 22, 18, 39, 1, 4, 36], float)
    T = np.repeat(n1[:, None], 7, 1)
    S = T.copy()
    w = zoib_weights(S, T, 0.5, 0.5, 0.9, rng)
    frac = {lab: float(np.mean([(grouped_zoib_draw(S, T, 0.5, 0.5, 0.9, None, np.random.default_rng(i), ww) < 1).any()
                                for i in range(1000)])) for lab, ww in (("fixed", None), ("learned", w))}
    # data from the model itself: indicators with weights (0.4, 0.2, 0.4), slab Beta(1/2, 1/2)
    lab = rng.choice(3, size=(30, 30), p=[.4, .2, .4])
    p_ = np.where(lab == 0, 0.0, np.where(lab == 1, 1.0, rng.beta(0.5, 0.5, size=lab.shape)))
    T2 = np.full(lab.shape, 50.0)
    S2 = rng.binomial(50, p_).astype(float)
    truth = np.array([(lab == 0).mean(), (lab == 1).mean(), (lab == 2).mean()])
    ws = np.mean([zoib_weights(S2, T2, 0.5, 0.5, 0.9, np.random.default_rng(i)) for i in range(20)], 0)
    nfree = {k: float(np.mean([(grouped_zoib_draw(S, T, 0.5, 0.5, 0.9, None, np.random.default_rng(i), ww) < 1).sum()
                               for i in range(1000)])) for k, ww in (("fixed", None), ("learned", w))}
    # as the engine draws them: empirical-Bayes slabs per row (a complete row has a
    # slab near 1) and weights per row around the relation's
    from relsynth.relational import eb_beta_rows
    ea, eb = eb_beta_rows(S, T, 0.5, 0.5)
    wr = zoib_weights(S, T, ea, eb, 0.9, np.random.default_rng(0))
    eng = float(np.mean([(grouped_zoib_draw(S, T, ea, eb, 0.9, None, np.random.default_rng(i), wr) < 0.99).any()
                         for i in range(1000)]))
    # groups of rows with 1-2 trials each: complete group-columns are structural ones
    # at the level the draw decides on (a header form filed with all its items)
    rg = np.random.default_rng(5)
    G, R, L = 6, 12, 20
    grp = np.repeat(np.arange(G), R)
    Tg = rg.integers(1, 3, size=(G * R, L)).astype(float)
    kind = rg.choice(3, size=(G, L), p=[0.4, 0.3, 0.3])[grp]          # zero, one, free per group-column
    Sg = np.where(kind == 0, 0.0, np.where(kind == 1, Tg, rg.binomial(Tg.astype(int), 0.5).astype(float)))
    ga, gb = eb_beta_rows(Sg, Tg, 0.5, 0.5)
    wg = np.mean([zoib_weights(Sg, Tg, ga, gb, 0.9, np.random.default_rng(i), groups=grp) for i in range(10)], 0)
    wb = np.mean([zoib_weights(Sg, Tg, ga, gb, 0.9, np.random.default_rng(i)) for i in range(10)], 0)
    comp = (kind == 1)
    full_g = float(np.mean([(grouped_zoib_draw(Sg, Tg, ga, gb, 0.9, grp, np.random.default_rng(i), wg)[comp] == 1).mean()
                            for i in range(50)]))
    # the blocks inside free group-columns keep the fixed prior: complete thin blocks
    # there are not drawn from the slab more often than in v0.8 (bi lost 4-7% of
    # its reported values when the row level was learned too)
    rows = {k: float(np.mean([(grouped_zoib_draw(Sg, Tg, ga, gb, 0.9, grp, np.random.default_rng(i), ww) * Tg).sum()
                              for i in range(50)])) for k, ww in (("fixed", None), ("learned", wg))}
    # per-row empirical-Bayes slabs (K x 1) broadcast over the blocks (the v0.9 crash)
    pa = np.full((30, 1), 0.7)
    w3 = zoib_weights(S2, T2, pa, pa, 0.9, rng)
    return [("complete relation (bi-like 16 x 7): P(a block drawn free) fixed weights", frac["fixed"], "> 0.2",
             frac["fixed"] > 0.2),
            ("same, weights learned across the blocks (hierarchical Dirichlet)", frac["learned"], "< 0.5",
             frac["learned"] < 0.5),
            ("free blocks per dataset, fixed vs learned (112 complete blocks)",
             f"{nfree['fixed']:.2f} vs {nfree['learned']:.2f}", "learned < fixed / 5",
             nfree["learned"] < nfree["fixed"] / 5),
            ("mixed relation: learned (zero, one, free) weights", np.round(ws, 3).tolist(),
             f"realised shares {np.round(truth, 3).tolist()} +- 0.05 (3 posterior sd)", bool(np.abs(ws - truth).max() < 0.05)),
            ("engine draw (row slabs, learned weights): P(some block of a dataset below 0.99)", eng, "< 0.01",
             eng < 0.01),
            ("grouped thin rows: w1 learned on group-columns vs on blocks", f"{wg[1]:.3f} vs {wb[1]:.3f}",
             f"group level = share of complete group-columns {(kind[::R] == 1).mean():.3f} +- 0.06",
             abs(wg[1] - (kind[::R] == 1).mean()) < 0.06),
            ("grouped thin rows: present pairs drawn, learned vs fixed weights", f"{rows['learned']:.0f} vs {rows['fixed']:.0f}",
             "within 1%", abs(rows["learned"] / rows["fixed"] - 1) < 0.01),
            ("grouped thin rows: complete group-columns drawn as structural ones", round(full_g, 4), ">= 0.99",
             full_g >= 0.99),
            ("per-row slabs (K x 1) accepted", np.round(w3, 3).tolist(), "3 weights summing to 1",
             len(w3) == 3 and abs(w3.sum() - 1) < 1e-9)]


def t_tree_cache():
    from relsynth import relational as R
    rng = np.random.default_rng(0)
    n, K = 20_000, 8
    par = np.full(n, -1)
    par[1:] = rng.integers(0, np.arange(1, n))

    def fresh(z, seed):
        e = R.Entity("t", n, K, np.zeros((n, 0)), np.zeros((n, 0), np.int64), [])
        e.z = z.copy()
        tr = R.TreeRel("tree", e, par.copy())
        tr.draw_params(np.random.default_rng(seed))
        return e, tr
    z = np.random.default_rng(1).integers(0, K, n)
    e, tr = fresh(z, 2)
    a = np.vstack([tr.loglik_rows(0, lo, min(n, lo + 3000)) for lo in range(0, n, 3000)])
    ok1 = np.array_equal(a, fresh(z, 2)[1].loglik_rows(0))
    e.z[:100] = (e.z[:100] + 1) % K                  # segments change: the cache must refresh
    b = tr.loglik_rows(0)
    ok2 = np.array_equal(b, tr._loglik_all())
    return [("cached chunks equal a fresh full computation", ok1, True, ok1),
            ("after a segment change the cache refreshes", ok2, True, ok2)]


# ---------------------------------------------------------------- engine
def t_hidden_draws():
    out = []
    syn, _ = _probe(chan_dist={"kind": "categorical", "probabilities": {"web": 0.7, "shop": 0.3}, "nulls": 0.2},
                    qty_dist={"kind": "poisson", "mean": 3})
    syn.rng = np.random.default_rng(0)
    df = pd.DataFrame({"channel": [None] * 10, "qty": [0] * 10})
    a = syn._apply_distributions("sales", df.copy())
    b = syn._apply_distributions("sales", a.copy())
    same = bool(((a.channel == b.channel) | (a.channel.isna() & b.channel.isna())).all())
    out.append(("stated draw with nulls kept on a recompute", same, True, same))
    # a check rule on a stated column: the rows checked are the rows kept
    syn, _ = _probe(qty_dist={"kind": "poisson", "mean": 3}, rules=[{"type": "check", "table": "sales", "expr": "qty >= 2"}])
    o = syn.sample(index=0)
    bad = int((o["sales"].qty < 2).sum())
    out.append(("rows with qty < 2 under rule qty >= 2 (stated Poisson)", bad, 0, bad == 0))
    rv = syn.sample_log["redrawn_rows"]["sales"]["still_invalid"]
    out.append(("rows still invalid after the per-row redraws", rv, 0, rv == 0))
    keep = {}
    orig = syn._finish

    def rec(t, df_, z, out_):
        r = orig(t, df_, z, out_)
        keep[t] = r.copy()
        return r
    syn._finish = rec
    syn.sample(index=1)
    fin = keep["sales"]
    again = syn._computed("sales", fin.copy(), syn.internal_out)
    ch = int((again.qty.to_numpy() != fin.qty.to_numpy()).sum())
    out.append(("stated values changed by a recompute after _finish (the SCD step)", ch, 0, ch == 0))
    hid = [c for c in syn.internal_out["sales"].columns if c.startswith("__")]
    out.append(("hidden columns in the output", hid, [], hid == []))
    return out


def t_key_order():
    from relsynth.engine import _natural_sort, _remap
    S = types.SimpleNamespace()
    S.key_index = {"orders": pd.Index([f"O{i}" for i in range(1, 10)])}
    S.spec = {"tables": {"orders": {"primary_key": "ok", "key_order": "d", "columns": {}},
                         "lines": {"primary_key": "lk", "columns": {}}},
              "relationships": [{"child": "lines", "child_key": "ok", "parent": "orders", "kind": "many_to_one"}]}
    S.copied, S.edge_info, S.E, S.sample_log = set(), {}, {}, {}
    n = 12
    keys = {"orders": Synthesizer._make_keys(S, "orders", n)}
    d = pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    out = {"orders": pd.DataFrame({"ok": keys["orders"], "d": d}),
           "lines": pd.DataFrame({"lk": [1, 2], "ok": pd.array(["O2", "O10"], dtype=object)})}
    before = list(out["orders"].ok)
    Synthesizer._renumber_by_order(S, "orders", out, keys)
    S2 = types.SimpleNamespace(spec=S.spec, copied=set(), edge_info={}, E={}, sample_log={})
    out2 = {"orders": pd.DataFrame({"ok": [1, 2, 3], "d": ["2024-01-03", "2024-01-01", "2024-01-02"]}),
            "lines": pd.DataFrame({"ok": pd.array([1, None, 3], dtype="Int64")})}
    try:
        Synthesizer._renumber_by_order(S2, "orders", out2, {"orders": np.array([1, 2, 3])})
        na = list(out2["lines"].ok)
        na_ok = na[0] == 3 and pd.isna(na[1]) and na[2] == 2
    except TypeError as e:
        na, na_ok = str(e), False
    return [("keys O1..O12 already in date order stay as they are", list(out["orders"].ok) == before, True,
             list(out["orders"].ok) == before),
            ("links unchanged", list(out["lines"].ok), ["O2", "O10"], list(out["lines"].ok) == ["O2", "O10"]),
            ("natural order of mixed labels", _natural_sort(np.array(["O10", "O2", "O1"], dtype=object)).tolist(),
             ["O1", "O2", "O10"], _natural_sort(np.array(["O10", "O2", "O1"], dtype=object)).tolist() == ["O1", "O2", "O10"]),
            ("nullable (pd.NA) links renumbered", [None if pd.isna(x) else x for x in na] if isinstance(na, list) else na,
             [3, None, 2], na_ok),
            ("_remap keeps nulls", str(list(_remap({1: 5}, pd.Series([1, None, pd.NA], dtype=object)))),
             "[5, None, <NA>]", str(list(_remap({1: 5}, pd.Series([1, None, pd.NA], dtype=object)))) == "[5, None, <NA>]")]


def t_affinity_windows():
    rng = np.random.default_rng(3)
    n, ns = 1000, 40
    store = pd.DataFrame({"sid": np.arange(ns), "area": np.arange(ns) % 4})
    area = rng.integers(0, 4, n)
    sid = np.array([rng.choice(store.sid[store.area != a]) for a in area])
    df = pd.DataFrame({"oid": np.arange(n), "area": area, "store_id": sid, "flag": (rng.random(n) < 0.3).astype(int)})

    def resolver(t, frame, out_):
        def res(p):
            fk, col = p.split(".")
            return store.set_index("sid")[col].reindex(frame[fk].to_numpy()).to_numpy()
        return res
    S = types.SimpleNamespace()
    S.affinity = {"orders": [{"a": "area", "b": "store_id.area", "fk": "store_id", "col": "area", "s": 0.10}]}
    S.fk = {"orders": [{"child_key": "store_id", "parent": "store"}]}
    S.cons = {"orders": {"unique": [], "key_rules": [{"expr": "flag == 1"}], "attr_rules": []}}
    S.spec = {"tables": {"store": {"primary_key": "sid"}}}
    S.rng, S.sample_log, S._resolver = np.random.default_rng(0), {}, resolver
    S._computed = lambda t, sub, out_, strict=True: sub
    res = Synthesizer._apply_affinity(S, "orders", df.copy(), {"store": store})
    same = int((store.set_index("sid").area.reindex(res.store_id).to_numpy() == res.area.to_numpy()).sum())
    changed = int((res.store_id != df.store_id).sum())
    return [("rows matching after the moves (target round(0.1 n) = 100)", same, 100, same == 100),
            ("rows moved = moves logged", f"{changed} / {S.sample_log['affinity_moves']['orders.store_id']}", "equal",
             changed == S.sample_log["affinity_moves"]["orders.store_id"])]


def t_mean_mode_shares():
    syn, _ = _probe(mode="mean")
    info = syn.fact_info["sales"]
    e1 = info["rel"].e1
    ws = []
    for i in range(2):
        syn.sample(index=i)
        ws.append(syn._weights(info["frailty1"], e1.z, {}, None, ent=e1, obs=info["obs1"]))
    d = float(np.abs(ws[0] - ws[1]).max())
    return [("per-row shares of a copied anchor, two mean-mode datasets: max difference", d, 0.0, d == 0.0)]


def t_sampling_warnings():
    rng = np.random.default_rng(1)
    cust = pd.DataFrame({"ck": np.arange(1, 61), "seg": rng.choice(["a", "b"], 60)})
    acct = pd.DataFrame({"ak": np.arange(1, 56), "ck": rng.permutation(np.arange(1, 61))[:55],
                         "bal": rng.gamma(2, 10, 55).round(2)})
    spec = {"version": 1, "name": "w", "source": {"kind": "csv", "path": "."}, "tables": {
        "cust": {"primary_key": "ck", "mode": "copy", "columns": {"ck": "id", "seg": "categorical"}},
        "acct": {"primary_key": "ak", "unique": [["ck"]], "columns": {"ak": "id", "ck": "id", "bal": "float"}}},
        "relationships": [{"child": "acct", "child_key": "ck", "parent": "cust"}],
        "synthesis": {"seed": 1, "scale": {"default": 1.0, "tables": {"acct": 1.5}}}}
    syn = Synthesizer(load_spec(spec)).fit({"cust": cust, "acct": acct})
    n0 = len(syn.log["warnings"])
    counts = []
    for i in range(3):
        syn.sample(index=i)
        counts.append(len(syn.sample_log["warnings"]))
    return [("sampling warnings per dataset (infeasible unique key)", counts, "[1, 1, 1]", counts == [1, 1, 1]),
            ("fit log unchanged by sampling", len(syn.log["warnings"]) - n0, 0, len(syn.log["warnings"]) == n0)]


def t_empty_parent():
    rng = np.random.default_rng(1)
    cust = pd.DataFrame({"ck": np.arange(1, 201), "seg": rng.choice(["a", "b"], 200), "age": rng.integers(18, 80, 200)})
    acct = pd.DataFrame({"ak": np.arange(1, 401), "ck": rng.choice(np.arange(1, 201), 400),
                         "bal": rng.gamma(2, 10, 400).round(2)})
    spec = {"version": 1, "name": "w", "source": {"kind": "csv", "path": "."}, "tables": {
        "cust": {"primary_key": "ck", "columns": {"ck": "id", "seg": "categorical", "age": "integer"}},
        "acct": {"primary_key": "ak", "columns": {"ak": "id", "ck": "id", "bal": "float"}}},
        "relationships": [{"child": "acct", "child_key": "ck", "parent": "cust"}],
        # acct has its own factor: since v1.0 a child entity without one follows its parent
        "synthesis": {"seed": 1, "scale": {"default": 1.0, "tables": {"cust": 0.001, "acct": 1.0}}}}
    syn = Synthesizer(load_spec(spec)).fit({"cust": cust, "acct": acct})
    try:
        o = syn.sample(index=0)
        nul = int(o["acct"].ck.isna().sum())
        return [("parent scaled to 0 rows: child links null, no crash", nul, 400, nul == 400 and len(o["cust"]) == 0),
                ("warned", len(syn.sample_log["warnings"]), ">= 1", len(syn.sample_log["warnings"]) >= 1)]
    except Exception as e:                      # noqa: BLE001
        return [(f"{type(e).__name__}: {e}", None, "no error", False)]


def t_mandatory_participation():
    """Every source customer has an order: no synthetic customer without one
    (the pooled participation prior; a flat Beta(1, 1) per segment left some)."""
    from fakedata.scenarios import order_dw
    from relsynth.engine import _pooled_beta
    tables, spec, _ = order_dw()
    fs, c = tables["fact_sales"], tables["dim_customer"]
    tables["dim_customer"] = c[c.customer_key.isin(set(fs.customer_key))].reset_index(drop=True)
    syn = Synthesizer(load_spec(spec)).fit(tables)
    shares = []
    for i in range(5):
        o = syn.sample(index=i)
        shares.append(float(o["dim_customer"].customer_key.isin(set(o["fact_sales"].customer_key)).mean()))
    # the pooled prior recovers a common share and keeps a real difference
    a, b = _pooled_beta(np.zeros(20), np.full(20, 50.0))
    a2, b2 = _pooled_beta(np.r_[np.full(10, 5.0), np.full(10, 45.0)], np.full(20, 50.0))
    return [("synthetic customers with an order, 5 datasets", shares, "all 1.0", shares == [1.0] * 5),
            ("pooled prior mean when no parent is empty (20 x 50)", round(a / (a + b), 6), "< 1e-3", a / (a + b) < 1e-3),
            ("pooled prior when shares differ (0.1 vs 0.9): little shrinkage", round(a2 + b2, 2), "<= 2",
             a2 + b2 <= 2)]


def t_unique_moves_fast():
    """Unique keys over links, and over a link and an attribute (ignored for entities
    before v0.9): the array-based moves keep every key unique (identity with the
    v0.8 loop was shown on keys over links: same output, 5-9x faster)."""
    rng = np.random.default_rng(1)
    n = 3000
    cust = pd.DataFrame({"ck": np.arange(1, n + 1), "seg": rng.choice(["a", "b"], n)})
    prod = pd.DataFrame({"pk": np.arange(1, 41), "cat": rng.choice(["x", "y"], 40)})
    acct = pd.DataFrame({"ak": np.arange(1, 2501), "ck": rng.permutation(np.arange(1, n + 1))[:2500],
                         "pk": rng.integers(1, 41, 2500), "bal": rng.gamma(2, 10, 2500).round(2)})
    acct = acct[~acct[["pk", "bal"]].duplicated()].reset_index(drop=True)      # the source keeps both keys
    spec = {"version": 1, "name": "w", "source": {"kind": "csv", "path": "."}, "tables": {
        "cust": {"primary_key": "ck", "mode": "copy", "columns": {"ck": "id", "seg": "categorical"}},
        "prod": {"primary_key": "pk", "mode": "copy", "columns": {"pk": "id", "cat": "categorical"}},
        "acct": {"primary_key": "ak", "unique": [["ck"], ["pk", "bal"]],
                 "columns": {"ak": "id", "ck": "id", "pk": "id", "bal": "float"}}},
        "relationships": [{"child": "acct", "child_key": "ck", "parent": "cust"},
                          {"child": "acct", "child_key": "pk", "parent": "prod"}],
        "synthesis": {"seed": 1, "scale": {"default": 1.0, "tables": {"acct": 1.15}}}}
    syn = Synthesizer(load_spec(spec)).fit({"cust": cust, "prod": prod, "acct": acct})
    t0 = time.time()
    o = syn.sample(index=0)
    sec = time.time() - t0
    d = o["acct"]
    dup1 = int(d.ck.dropna().duplicated().sum())
    dup2 = int(d[["pk", "bal"]].dropna().duplicated().sum())
    return [("duplicate ck / (pk, bal) after the moves", f"{dup1} / {dup2}", "0 / 0", dup1 == 0 and dup2 == 0),
            ("sample seconds (reported; the (pk, bal) key over a link and an attribute was ignored before v0.9)",
             round(sec, 2), "reported", True)]


def t_time_buckets():
    out = []
    for f in "DWMQY":
        t = pd.Series(pd.to_datetime(["2020-01-03", "2021-07-15", None, "2019-12-30", "2023-02-28"]))
        per = t.dt.to_period(f)
        lo = per.min()
        a = np.asarray(per.map(lambda x: (x - lo).n if pd.notna(x) else -1), dtype=np.int64)
        b = np.where(per.notna().to_numpy(), per.array.asi8 - lo.ordinal, -1).astype(np.int64)
        out.append((f"bucket numbers by ordinals, freq {f}", b.tolist(), a.tolist(), a.tolist() == b.tolist()))
    return out


def t_fixed_weights_option():
    """structure.structural_pooling: 3, the default, learns the structural weights
    on group-columns; 0 keeps the fixed weights of v0.8 (drawn by the v0.8 path).
    The reporting scenario's undeclared structure is exact with fixed weights and
    within 0.3% with learned ones."""
    sys.path.insert(0, str(ROOT / "checks"))
    import scenarios as S
    from run_checks import _s8_stats
    out = []
    for kappa in (0.0, None):
        t, spec, truth = S.reporting()
        if kappa is not None:
            spec["synthesis"].setdefault("structure", {})["structural_pooling"] = kappa
        sp = load_spec(spec)
        syn = Synthesizer(sp).fit(t)
        f = syn.sample(index=0)
        st = _s8_stats(t, f)
        iv = f["item_value"].merge(f["report"][["report_id", "version_id"]], on="report_id")
        ver = float((iv.version_id.to_numpy() == iv.item_id.map(truth["item_version"]).to_numpy()).mean())
        w = getattr(syn.fact_info["item_value"]["rel"], "struct_w", None)
        if kappa == 0:
            out += [("pooling 0: no learned weights (v0.8 path)", w, "None", w is None),
                    ("fixed weights: header form complete whenever filed", round(st["header_complete"], 4), ">= 0.99",
                     st["header_complete"] >= 0.99),
                    ("fixed weights: item version matches report version", round(ver, 5), ">= 0.9999", ver >= 0.9999)]
        else:
            out += [("default pooling 3: weights learned", (sp["synthesis"]["structure"]["structural_pooling"],
                     None if w is None else np.round(w, 3).tolist()), "(3.0, 3 weights)",
                     sp["synthesis"]["structure"]["structural_pooling"] == 3 and w is not None and len(w) == 3),
                    ("learned weights: header form complete whenever filed", round(st["header_complete"], 4), ">= 0.98",
                     st["header_complete"] >= 0.98),
                    ("learned weights: item version matches report version", round(ver, 5), ">= 0.999", ver >= 0.999)]
    return out


TESTS = [t_dp_closed_form, t_sparse_dirichlet, t_zoib_pooling, t_tree_cache, t_hidden_draws, t_key_order,
         t_affinity_windows, t_mean_mode_shares, t_sampling_warnings, t_empty_parent, t_mandatory_participation,
         t_unique_moves_fast, t_time_buckets, t_fixed_weights_option]


def main(argv):
    sel = [t for t in TESTS if not argv or t.__name__ in argv]
    bad = 0
    for t in sel:
        t0 = time.time()
        try:
            items = t()
        except Exception as e:                      # noqa: BLE001
            items = [(f"{type(e).__name__}: {e}", None, None, False)]
        ok = all(i[3] for i in items)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {t.__name__} ({time.time() - t0:.1f}s)")
        for what, value, target, good in items:
            print(f"   {'ok ' if good else 'BAD'} {what}: {value} (target {target})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
