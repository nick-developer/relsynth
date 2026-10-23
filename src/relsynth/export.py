"""Export of the fitted data model (v0.8): the complete Bayesian model of a
specification and its data, as data (model.json), as text (model.md), as a
document (model.tex / model.pdf) and as code (replicate.py, run_model.py).

    relsynth export-model spec.yaml --out model/ [--draws 5] [--formats json,md,pdf,py]

What is exported
----------------
The fitted model is the posterior p(theta | data) of every parameter block given
the fitted latent partitions (segments).  The export holds

* the structure: tables in generation order, entity sizes, segment counts,
  links and their count laws, the fact models (single, Poisson block,
  occupancy), the column codecs (levels, types, bounds), strata, trees, time
  columns, lookups, derived columns and aggregates;
* M posterior draws theta^(1..M) and the posterior mean of every block.  Draw m
  is the parameter set that `relsynth run` uses for synthetic dataset m (the
  same parameter stream, posterior.run_streams), so a dataset sampled from draw
  m is a draw from the same posterior predictive distribution;
* the verification of the model (`verify`): every block checked against the
  conditions under which the propositions of docs/layer2-proofs.pdf hold.

Two ways to replicate
---------------------
* run_model.py: the relsynth API calls of `relsynth run` (bit-identical output).
* replicate.py: a standalone sampler (numpy, pandas, scipy) of the core
  generative process from model.json; the engine layers it leaves out are listed
  in model.json["replica_scope"] and described in model.md.

Privacy: model.json holds aggregates only (segment shares, rates, level
probabilities, 49 quantile knots per column and segment, count histograms).  It
discloses what synthetic data drawn from it discloses; text columns, free
identifiers and copied reference rows are not exported.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import shutil
# subprocess runs pdflatex only (write_pdf): fixed program and arguments, no shell
import subprocess  # nosec B404
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .expr import split_assignment

FORMAT_VERSION = 1
FORMATS = ("json", "md", "pdf", "py")
OFF1, OFF2 = 10**6, 2 * 10**6


# ====================================================================== JSON helpers
def _js(v):
    """A JSON value for a level, constant or bound."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, int):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, tuple):
        return [_js(x) for x in v]
    return str(v)


def _arr(a, inf_as_none=False):
    """Nested lists of floats; infinite values become None where allowed."""
    a = np.asarray(a, dtype=float)
    if inf_as_none:
        return [_arr(x, True) for x in a] if a.ndim > 1 else [None if not math.isfinite(x) else float(x) for x in a]
    if not np.isfinite(a).all():
        raise ValueError("non-finite parameter value")
    return a.tolist()


def _ilist(a):
    return [int(x) for x in np.asarray(a).ravel()]


# ====================================================================== codecs and marginals
def codec_dict(cd):
    if getattr(cd, "parts", None):
        return {"name": cd.name, "kind": "discrete", "ctype": "categorical",
                "levels": [[int(i) for i in lv] for lv in cd.levels],
                "parts": [codec_dict(p) for p in cd.parts]}
    d = {"name": cd.name, "kind": cd.kind, "ctype": cd.ctype, "min": _js(cd.min), "max": _js(cd.max),
         "decimals": cd.decimals}
    if cd.kind == "discrete":
        d["levels"] = [_js(v) for v in cd.levels]
        tail = getattr(cd, "tail", None)
        d["tail"] = [[_js(v) for v in tail[0]], [float(c) for c in tail[1]]] if tail is not None else None
    return d


def marginal_dict(m):
    if m.kind == "discrete":
        return {"kind": "discrete", "p": _arr(m.p), "p_null": float(m.p_null)}
    return {"kind": "continuous", "knots_v": _arr(m.knots_v), "knots_c": _arr(m.knots_c), "p_null": float(m.p_null)}


def _key_layout(syn, t):
    """Synthetic primary keys: 1..n, or the source's prefix + zero-padded number."""
    src = syn.key_index.get(t)
    if src is not None and len(src) and isinstance(src[0], str):
        s0 = str(src[0])
        pre = s0.rstrip("0123456789")
        return pre, max(len(s0) - len(pre), 1)
    return None, None


def _dp(dp):
    if not dp:
        return None
    return {"a": float(dp["a"]), "min": int(dp.get("min", 0)),
            "counts": [_ilist(c) for c in dp["counts"]]}


# ====================================================================== structure
def structure(syn):
    """The model's fixed part: everything a sampler needs besides theta.  Returns
    (structure, scope) where scope lists what replicate.py leaves to the engine."""
    spec, cfg = syn.spec, syn.cfg
    T = spec["tables"]
    scope = []
    sc = syn.plan.scale(cfg["scale"])
    hidden = sorted({h["table"] for h in syn.plan.headers.values()}
                    | {d["table"] for ds in syn.plan.dims.values() for d in ds})
    S = {"format_version": FORMAT_VERSION, "name": spec.get("name"), "seed": cfg["seed"],
         "tables": {}, "entity_order": list(syn.order), "entities": {}, "trees": {}, "attributes": {},
         "links": {}, "fact_order": [], "facts": {}, "table_order": list(syn.all_order),
         "lookups": {}, "derive": {}, "aggregates": [], "placeholders": {}, "constants": {}, "hidden": hidden,
         "offsets": {}, "sequences": {},
         "paths": {t: {r["child_key"]: r["parent"] for r in spec["relationships"]
                       if r["child"] == t and isinstance(r["child_key"], str)} for t in T}}
    computed = {}
    for t, ts in T.items():
        fks = {r["child_key"] for r in spec["relationships"] if r["child"] == t and isinstance(r["child_key"], str)}
        S["tables"][t] = {"role": ts["role"], "primary_key": ts.get("primary_key"), "columns": list(ts["columns"]),
                          "types": {c: cs.get("type") for c, cs in ts["columns"].items()},
                          "copied": t in syn.copied}
        computed[t] = set(syn.lookups[t]) | set(syn.derived[t]) | set(syn.aggregated[t])
        seq = [{"column": c, "within": cs["within"], "start": int(cs.get("start", 1))}
               for c, cs in ts["columns"].items() if cs["type"] == "sequence" and cs.get("within")]
        if seq and t not in syn.copied:
            S["sequences"][t] = seq
        ph = [c for c, cs in ts["columns"].items()
              if cs["type"] in ("id", "text", "pii", "binary", "sequence") and c != ts.get("primary_key")
              and c not in fks and c not in computed[t] and c not in {q["column"] for q in seq}]
        if ph and t not in syn.copied:
            S["placeholders"][t] = ph
            scope.append(f"{t}: {', '.join(ph)} (text, identifiers and sequences; the engine's text and "
                         "identifier models are not exported because they encode source strings)")
        cons = {c: _js(syn.pii[(t, c)][1]) for c, cs in ts["columns"].items()
                if cs["type"] == "constant" and (t, c) in getattr(syn, "pii", {})}
        if cons:
            S["constants"][t] = cons
    # ---- entities
    for e in syn.order:
        ent = syn.E[e]
        pre, width = _key_layout(syn, e)
        d = {"rows": int(ent.n), "K": int(ent.K), "scale": float(sc["tables"].get(e, sc["default"])),
             "copied": e in syn.copied, "key_prefix": pre, "key_width": width}
        if e in syn.copied:
            d["z"] = _ilist(ent.z)
        if e in syn.grain:
            d["unsupported"] = "grain entity (rows are occupied key pairs)"
            scope.append(f"{e}: grain entity (rows drawn as occupied key pairs) is not in the replica")
        if e in syn.attr_strata and ent.n // max(cfg["segments"]["min_rows_per_segment"], 1) < 2:
            d["fixed_counts"] = _ilist(np.bincount(ent.z, minlength=ent.K))
        st = syn.strata_info.get(e)
        if st:
            d["strata"] = {"seg_stratum": _ilist(ent.seg_stratum),
                           "labels": [[_js(x) for x in (lab if isinstance(lab, tuple) else (lab,))]
                                      for lab in st["labels"]],
                           "by": list(st["by"]), "attr": st.get("attr"),
                           "by_parent": {c: next(r["parent"] for r in syn.fk[e] if r["child_key"] == c)
                                         for c in st["by"]}}
        if T[e].get("key_order"):
            scope.append(f"{e}: keys numbered in the order of {T[e]['key_order']} (engine only)")
        S["entities"][e] = d
    # ---- trees
    maxd = cfg["trees"]["max_depth"]
    for name, info in syn.edge_info.items():
        rel = info["rel"]
        if rel.kind != "tree":
            continue
        ent = rel.e1
        cnt = np.zeros((info["max_depth"] + 1, ent.K))
        np.add.at(cnt, (rel.depth, ent.z), 1.0)
        if maxd != "auto":
            cnt = cnt[: int(maxd) + 1]
        S["trees"][ent.name] = {"name": name, "child_key": info["spec"]["child_key"], "level_counts": _arr(cnt)}
    # ---- attribute models
    for t, tm in syn.tm.items():
        if t in syn.copied or not hasattr(tm, "group_m"):
            continue
        S["attributes"][t] = {"codecs": [codec_dict(c) for c in tm.codecs], "out_cols": list(tm.out_cols),
                              "aux": list(tm.aux), "independent": list(getattr(tm, "independent", []))}
    # ---- links between entities
    for e in syn.order:
        fixed = set((syn.strata_info.get(e) or {}).get("by", []))
        for r in syn.fk[e]:
            if r["child_key"] in fixed:
                continue
            info = syn.edge_info.get(r["name"])
            if info is None:
                continue
            rel = info["rel"]
            pc = bool(getattr(rel.e2, "copied", False))
            S["links"][r["name"]] = {"name": r["name"], "child": e, "parent": r["parent"],
                                     "child_key": r["child_key"], "count_dp": _dp(info.get("count_dp")),
                                     "min_one": bool(info.get("min_one")), "parent_copied": pc,
                                     "obs": _arr(info["obs"]) if pc else None}
            if r.get("anchor"):
                scope.append(f"{r['name']}: anchored link (parent chosen given {r['anchor']}) drawn as a plain link")
    # ---- facts
    for f in syn.fact_order:
        info = syn.fact_info[f]
        rel = info["rel"]
        if info["nested"] or info.get("vdim"):
            scope.append(f"{f}: {'nested fact' if info['nested'] else 'fact with a virtual dimension'} "
                         "(engine only)")
            continue
        S["fact_order"].append(f)
        fk = {"model": "single" if rel.kind == "single" else info["model"], "anchor": info["anchor"],
              "anchor_key": info["fks"][0]["child_key"], "count_dp": _dp(info.get("count_dp")),
              "min_one": bool(info.get("min_one")), "anchor_copied": bool(getattr(rel.e1, "copied", False)),
              "aux_map": [list(x) for x in (getattr(syn, "aux_map", {}) or {}).get(f, [])]}
        fk["obs1"] = _arr(info["obs1"]) if fk["anchor_copied"] else None
        if rel.kind != "single":
            r2 = info["fks"][1]
            fk.update({"partner": r2["parent"], "partner_key": r2["child_key"],
                       "m_real": _arr(rel.e2.counts()), "fallback": info.get("fallback", "anchor"),
                       "fitted_groups": sorted(int(g) for g in info["fitted_groups"]),
                       "partner_copied": bool(getattr(rel.e2, "copied", False))})
            fk["obs2"] = _arr(info["obs2"]) if fk["partner_copied"] else None
            if rel.kind == "occ":
                fk["mult_by"] = rel.mult_by
                fk["occ_min_one"] = bool(info.get("occ_min_one"))
                ra = getattr(rel, "row_active", None)
                fk["row_active"] = [bool(x) for x in ra] if ra is not None else None
            fk["extras"] = []
            for r, xp, rx in zip(info["fks"][2:], info.get("extra_p", []), info.get("extra", [])):
                xc = bool(getattr(rx.e2, "copied", False))
                fk["extras"].append({"parent": r["parent"], "child_key": r["child_key"], "null": bool(xp.get("null")),
                                     "parent_copied": xc, "obs": _arr(xp["obs"]) if xc else None,
                                     "backoff": xp.get("backoff")})
        if "time" in info:
            ti = info["time"]
            fk["time"] = {"col": ti["col"], "starts": [str(x) for x in ti["buckets"].start_time],
                          "ends": [str(x) for x in ti["buckets"].end_time], "is_dt": bool(ti["is_dt"]),
                          "tmin": str(ti["tmin"]), "tmax": str(ti["tmax"]), "bucket": T[f]["fact"].get("time_bucket", "month")}
        S["facts"][f] = fk
    # ---- computed columns
    for t in T:
        off = [{"child_key": r["child_key"], "parent": r["parent"], "offset_from": r["offset_from"],
                "offset_column": r["offset_column"], "order_by": r["order_by"], "unit": r["unit"]}
               for r in syn.derived_keys.get(t, []) if r.get("offset_column")]
        if off:
            S["offsets"][t] = off
        lk = []
        for c, p in syn.lookups[t].items():
            if isinstance(p, str):
                cs = T[t]["columns"].get(c, {})
                lk.append({"column": c, "path": p, "map": cs.get("map"), "default": _js(cs.get("default"))})
        if lk:
            S["lookups"][t] = lk
        if len(lk) < len(syn.lookups[t]):
            scope.append(f"{t}: keyed lookups (engine only)")
        dv = []
        for r in spec["rules"]:
            if r["table"] == t and r["type"] == "derive":
                lhs, rhs = split_assignment(r["expr"])
                d = getattr(syn, "derive_decimals", {}).get((t, lhs))
                dv.append({"column": lhs, "expr": rhs, "type": T[t]["columns"].get(lhs, {}).get("type"),
                           "decimals": None if d is None else int(d)})
        if dv:
            S["derive"][t] = dv
    for r in spec["rules"]:
        if r["type"] == "aggregate":
            rel = next((x for x in spec["relationships"] if x["child"] == r["from"] and x["parent"] == r["table"]
                        and r.get("via") in (None, x["name"]) and not x.get("composite")), None)
            if rel is None:
                scope.append(f"{r['table']}.{r['column']}: aggregate over a composite key (engine only)")
                continue
            S["aggregates"].append({"table": r["table"], "column": r["column"], "from": r["from"],
                                    "agg": r.get("agg", "count"), "of": r.get("of"), "child_key": rel["child_key"]})
        elif r["type"] in ("check", "cumulative", "soft"):
            scope.append(f"{r['table']}: {r['type']} rule '{r.get('expr', r.get('column', ''))}' (engine only)")
    for name, what in (("scd", "slowly changing versions"), ("affinity", "link affinity"),
                       ("participation", "participation hurdle"), ("group_effects", "group effects"),
                       ("inclusion", "conditional inclusion")):
        obj = getattr(syn, name, None)
        if obj:
            scope.append(f"{what}: {', '.join(sorted(map(str, obj)))} (engine only)")
    if syn.plan.views:
        scope.append(f"views {', '.join(syn.plan.views)} (engine only)")
    if syn.plan.unknown_rows:
        scope.append("unknown members (engine only)")
    if syn.plan.headers:
        scope.append("order headers: generated as hidden tables and joined back by lookups")
    return S, scope


# ====================================================================== parameters
def theta(syn):
    """The current value of every parameter block (after redraw)."""
    th = {"entities": {}, "attributes": {}, "links": {}, "trees": {}, "facts": {}}
    for e in syn.order:
        ent = syn.E[e]
        if e not in syn.copied:
            pi = np.asarray(ent.pi, float)
            th["entities"][e] = {"pi": _arr(pi / pi.sum())}
    for t, tm in syn.tm.items():
        if t in syn.copied or not hasattr(tm, "group_m"):
            continue
        tail = {}
        for cd in tm.codecs:
            for part in getattr(cd, "parts", None) or [cd]:
                w = getattr(part, "run_tail_w", None)
                if w is not None:
                    tail[part.name] = _arr(w)
        th["attributes"][t] = {
            "groups": {str(int(g)): [marginal_dict(m) for m in ms] for g, ms in tm.group_m.items()},
            "global": [marginal_dict(m) for m in tm.global_m],
            "copulas": {str(int(g)): _arr(c.current) for g, c in tm.copulas.items()},
            "pooled": _arr(tm.pooled.current), "tail_w": tail}
    for name, info in syn.edge_info.items():
        rel = info["rel"]
        if rel.kind == "tree":
            th["trees"][rel.e1.name] = {"lam_d": _arr(info["lam_d"]), "psi_d": _arr(info["psi_d"]),
                                       "frailty": _arr(info["frailty"], True), "lam": _arr(info["lam"]),
                                       "psi": _arr(info["psi"])}
        else:
            th["links"][name] = {"phi": _arr(info["phi"]), "p_null": _arr(info["p_null"]),
                                 "parent_frailty": _arr(info["parent_frailty"], True)}
    for f in syn.fact_order:
        info = syn.fact_info[f]
        rel = info["rel"]
        if info["nested"] or info.get("vdim"):
            continue
        d = {"frailty1": _arr(info["frailty1"], True)}
        if rel.kind == "single":
            d["lam"] = _arr(rel.lam)
        else:
            d["frailty2"] = _arr(info["frailty2"], True)
            if rel.kind == "occ":
                d["pi"], d["p"] = _arr(rel.pi), _arr(rel.p)
                d["mu"] = _arr(rel.mu)
            else:
                d["eta"] = _arr(rel.eta)
            d["extras"] = [{"P": _arr(xp["P"]), "frailty": _arr(xp["frailty"], True)} for xp in info.get("extra_p", [])]
        if "time" in info:
            d["time"] = {"P": _arr(info["time"]["P"]), "null_rate": float(info["time"]["null_rate"])}
        th["facts"][f] = d
    return th


def posterior_draws(syn, M=5):
    """(mean, [draw_0 .. draw_{M-1}]): draw m uses the parameter stream of synthetic
    dataset m (posterior.run_streams), so it is the theta of `relsynth run`'s m-th dataset."""
    from .posterior import run_streams
    cfg = syn.cfg
    ss = cfg.get("posterior", {}).get("sample_seed")
    draws = []
    for m in range(M):
        prng, _ = run_streams(cfg["seed"], ss, m)
        syn.redraw("draw", prng)
        draws.append(theta(syn))
    prng, _ = run_streams(cfg["seed"], ss, 0)
    syn.redraw("mean", prng)
    return theta(syn), draws


# ====================================================================== verification
def _simplex(x, tol=1e-6):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None]
    s = x.sum(-1)
    return bool((x >= -1e-12).all() and np.all((np.abs(s - 1) < tol) | (s == 0)))


def _check(out, name, ok, detail, ref, level="fail"):
    """level 'fail': a condition of a proposition (the model is invalid without it);
    'warn': a convergence diagnostic (the model is valid, its draws less certain)."""
    out.append({"check": name, "ok": bool(ok), "detail": detail, "reference": ref, "level": level})


def _geweke(trace):
    """Geweke (1992) z of a chain's first 10 % against its last 50 %, with
    batch-means variances (a stationarity diagnostic; |z| < 3 expected)."""
    x = np.asarray(trace, float)
    n = len(x)
    if n < 20:
        return None
    a, b = x[: max(n // 10, 2)], x[n // 2:]

    def bvar(y):
        k = max(int(np.sqrt(len(y))), 1)
        m = len(y) // k
        if m < 2:
            return y.var() / max(len(y), 1)
        bm = y[: m * k].reshape(m, k).mean(1)
        return bm.var(ddof=1) / m
    se = np.sqrt(bvar(a) + bvar(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def verify(S, mean, draws, syn=None):
    """Checks of the exported model against the conditions of the propositions
    (docs/layer2-proofs.pdf).  Each item: check, ok, detail, reference."""
    out = []
    allbad, worst, nbad = [], 1.0, 0
    for lab, th in [("mean", mean)] + [(f"draw {m}", d) for m, d in enumerate(draws)]:
        bad = []
        for e, d in th["entities"].items():
            if not _simplex(d["pi"]):
                bad.append(f"{e}.pi")
        for name, d in th["links"].items():
            if not _simplex(d["phi"]):
                bad.append(f"{name}.phi")
            if not all(0 <= x <= 1 for x in d["p_null"]):
                bad.append(f"{name}.p_null")
        for e, d in th["trees"].items():
            if not _simplex(np.asarray(d["psi_d"]).reshape(-1, np.asarray(d["psi_d"]).shape[-1])) or \
                    np.min(d["lam_d"]) < 0:
                bad.append(f"{e} tree")
        for f, d in th["facts"].items():
            for k in ("lam", "eta", "mu"):
                if k in d and np.min(d[k]) < 0:
                    bad.append(f"{f}.{k}")
            for k in ("pi", "p"):
                if k in d and not (np.min(d[k]) >= 0 and np.max(d[k]) <= 1):
                    bad.append(f"{f}.{k}")
            for x in d.get("extras", []):
                if not _simplex(x["P"]):
                    bad.append(f"{f} extra P")
            if "time" in d and not _simplex(d["time"]["P"]):
                bad.append(f"{f} time P")
        for t, a in th["attributes"].items():
            for g, ms in list(a["groups"].items()) + [("global", a["global"])]:
                for j, m in enumerate(ms):
                    if not 0 <= m["p_null"] <= 1:
                        bad.append(f"{t}[{g}] col {j} p_null")
                    if m["kind"] == "discrete" and not _simplex(m["p"]):
                        bad.append(f"{t}[{g}] col {j} p")
                    if m["kind"] == "continuous" and (np.any(np.diff(m["knots_v"]) < 0) or np.any(np.diff(m["knots_c"]) < 0)
                                                      or m["knots_c"][0] < 0 or m["knots_c"][-1] > 1):
                        bad.append(f"{t}[{g}] col {j} quantile knots")
        allbad += [f"{b} ({lab})" for b in bad]
        for a in th["attributes"].values():
            for C in list(a["copulas"].values()) + [a["pooled"]]:
                C = np.asarray(C, float)
                if C.size <= 1:
                    continue
                ev = np.linalg.eigvalsh((C + C.T) / 2).min()
                worst = min(worst, ev)
                if not (np.allclose(C, C.T, atol=1e-8) and np.allclose(np.diag(C), 1, atol=1e-6) and ev > -1e-10):
                    nbad += 1
    nth = 1 + len(draws)
    _check(out, f"parameter supports (mean and {len(draws)} draws)", not allbad,
           "every probability vector is on the simplex, every probability in [0, 1], every rate >= 0, "
           "every quantile function non-decreasing" + (f"; violations: {', '.join(allbad[:10])}" if allbad else ""),
           "proper conjugate posteriors: Props 'Block-rate posterior', 'Spike-and-slab Bernoulli', "
           "'Sparse Dirichlet, exact inclusion', 'Posterior mean of a segment marginal'")
    _check(out, f"copula correlation matrices ({nth} parameter sets)", nbad == 0,
           f"symmetric, unit diagonal, positive semi-definite; smallest eigenvalue {worst:.2e}"
           + (" (near-singular matrices are regularised to the nearest correlation matrix with eigenvalues "
              ">= 1e-8 before sampling)" if worst < 1e-6 else ""),
           "Sklar's theorem; Props 'Marginal preservation', 'The sampler targets the copula posterior'")
    # Galton-Watson trees: finite by construction (generation conditioned on the level counts)
    for e, tr in S["trees"].items():
        d = mean["trees"][e]
        M = np.asarray(d["psi"]) * np.asarray(d["lam"])[:, None]
        rho = float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0
        _check(out, f"tree {e}: size and depth", True,
               f"mean-matrix spectral radius {rho:.3f} ({'subcritical' if rho < 1 else 'critical or supercritical'}); "
               f"the forest is generated conditioned on its {len(tr['level_counts'])} level x segment counts, so its "
               "size and depth are those of the source (scaled) whatever the radius",
               "Props 'Generation sizes, finiteness and expected size', 'Hierarchy of a given size', "
               "'Hierarchies generated level by level'")
    # scale rule: a block with a positive rate needs partner rows to scale by
    for f, fk in S["facts"].items():
        if fk["model"] == "single":
            continue
        rate = np.asarray(mean["facts"][f].get("eta", mean["facts"][f].get("p")), float)
        mr = np.asarray(fk["m_real"], float)
        dead = int(((rate > 0) & (mr[None, :] == 0)).sum())
        _check(out, f"{f}: scale rule", dead == 0,
               f"every block with a positive rate has source partner rows (m_l > 0); {dead} blocks without",
               "Prop. 'Scale rule' (eta* = eta m_l / m*_l keeps rows per anchor under resizing)")
    if syn is not None:
        from .copula import split_rhat
        h = syn.rm.history
        tr = np.asarray(h.get("loglik_trace") or [], float)
        fin = tr[h.get("final_trace_start", 0):]
        fin = fin[len(fin) // 2:]             # the first half of the final sweeps is burn-in
        rh = split_rhat(fin) if len(fin) >= 8 else None
        z = _geweke(fin)
        tgt = syn.cfg["gibbs"].get("rhat_target", 1.1)
        ok = (rh is None or not np.isfinite(rh) or rh <= tgt) and (z is None or abs(z) < 3)
        _check(out, "latent-segment Gibbs sampler: stationarity", ok,
               f"last {len(fin)} of {h.get('final_sweeps', 2 * len(fin))} final sweeps (after the merge and a burn-in "
               f"of half the final sweeps; the partition is the last one): split R-hat of the "
               f"attribute log-likelihood {'n/a' if rh is None else round(float(rh), 3)}, Geweke z "
               f"{'n/a' if z is None else round(z, 2)} (stationary: R-hat <= {tgt}, abs(z) < 3)"
               + ("" if ok else "; raise synthesis.gibbs.max_final_iterations for a longer chain"),
               "blocked Gibbs sampler (layer 1 algorithms paper, Algorithm 3); Geweke (1992); Gelman et al. (2013)",
               level="warn")
        rh, bad, mz, where = [], 0, [], []
        for t, tm in syn.tm.items():
            if t in syn.copied or not hasattr(tm, "pooled"):
                continue
            for g, c in [("pooled", tm.pooled)] + list(tm.copulas.items()):
                inf = c.info or {}
                v = [inf.get(k) for k in ("rhat_max_entry", "rhat_logdet")]
                v = [x for x in v if x is not None and np.isfinite(x)]
                if v:
                    rh.append(max(v))
                    bad += inf.get("converged") is False
                    if inf.get("converged") is False:
                        names = _copula_columns(tm, c)
                        prs = ", ".join(f"{names[i]} ~ {names[j]}" for i, j in inf.get("unconverged_pairs", [])
                                        if i < len(names) and j < len(names))
                        where.append(f"{t} segment {g}" + (f" ({prs})" if prs else ""))
                    if inf.get("mcse_z_max") is not None:
                        mz.append(inf["mcse_z_max"])
        target = syn.cfg["copula"].get("rhat_target", 1.1)
        from .copula import MCSE_Z
        okc = bad == 0
        _check(out, "copula MCMC: convergence", okc,
               (f"{len(rh) - bad} of {len(rh)} chains pass: every correlation has split R-hat <= {target} or a "
                f"Monte Carlo standard error <= {MCSE_Z} on the Fisher-z scale (largest R-hat {max(rh):.3f}, "
                f"largest standard error {max(mz) if mz else float('nan'):.4f})" if rh else
                "no MCMC copula (one column, or the fast estimate)")
               + ("" if okc else "; not converged: " + "; ".join(where[:4])
                  + ". Usually a categorical column whose rarer levels sit on a handful of rows of the segment and "
                    "are separated by another column; its correlations are weakly identified and the chain moves "
                    "slowly. Raising synthesis.marginals.min_dependence_rows keeps such columns independent "
                    "within the segment; raising synthesis.copula.max_iterations runs longer chains"),
               "Prop. 'Rank likelihood with parameter expansion'; split R-hat (Gelman et al. 2013); fixed-width "
               "stopping (Jones et al. 2006; Flegal, Haran & Jones 2008)", level="warn")
    return out


def _copula_columns(tm, c):
    """Column names in the order of the chain's monitored matrix (the live columns
    when constant or sparse columns were kept out of the copula)."""
    p = len(tm.cols)
    D = np.asarray(c.draws[0]) if c.draws else np.eye(p)
    if (c.info or {}).get("dropped_constant") and D.shape[0] == p:
        off = np.abs(np.stack(c.draws) - np.eye(p)).max(axis=0)
        live = [j for j in range(p) if (np.delete(off[j], j) > 0).any()]
        return [tm.cols[j] for j in live]
    return list(tm.cols)


# ====================================================================== writers
def _md_matrix(A, rows=None, cols=None, fmt="{:.3f}", maxdim=12):
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[None]
    if A.shape[0] > maxdim or A.shape[1] > maxdim:
        nz = A[A > 0]
        return (f"{A.shape[0]} x {A.shape[1]} matrix; {int((A > 0).sum())} positive entries, "
                f"range {nz.min():.3g} to {nz.max():.3g}" if len(nz) else f"{A.shape[0]} x {A.shape[1]} zero matrix") + "\n"
    rows = rows or [str(i) for i in range(A.shape[0])]
    cols = cols or [str(j) for j in range(A.shape[1])]
    s = "| | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for r, row in zip(rows, A):
        s += f"| {r} | " + " | ".join(fmt.format(x) for x in row) + " |\n"
    return s


def _col_summary(cd, m):
    if cd["kind"] == "discrete":
        lv = cd["levels"] if not cd.get("parts") else ["(" + ", ".join(str(cd["parts"][q]["levels"][i]) if i >= 0 else "null"
                                                                      for q, i in enumerate(l)) + ")" for l in cd["levels"]]
        p = np.asarray(m["p"])
        top = np.argsort(-p)[:4]
        return f"discrete, {len(p)} levels; " + ", ".join(f"{_one(lv[i])}: {p[i]:.3f}" for i in top) + \
            (f"; null {m['p_null']:.3f}" if m["p_null"] else "")
    kv = np.asarray(m["knots_v"])

    def q(c):
        v = float(np.interp(c, m["knots_c"], kv))
        if cd["ctype"] == "date":
            return str((pd.Timestamp(0) + pd.Timedelta(days=round(v))).date())
        if cd["ctype"] == "datetime":
            return str(pd.Timestamp(0) + pd.Timedelta(seconds=round(v)))
        return f"{v:.4g}"
    return f"continuous ({cd['ctype']}); median {q(0.5)}, 5%-95% {q(0.05)} to {q(0.95)}" + \
        (f"; null {m['p_null']:.3f}" if m["p_null"] else "")


GENERAL_MD = r"""
## 1. The generative model

A synthetic database D* is a draw from the posterior predictive distribution

    theta ~ p(theta | D, z),   D* ~ p(D* | theta)

where z are the latent segments fitted by the blocked Gibbs sampler (one segment per
row of every entity table) and theta collects every parameter block below.  All
blocks are conjugate given z, so p(theta | D, z) is a product of the closed-form
posteriors listed here; draw m of model.json is the theta that `relsynth run` uses
for synthetic dataset m.  Tables are generated parents first (section 3).

**Entity table e** (n_e rows, K_e segments)

    pi_e ~ Dirichlet(1 / K_e + counts of rows per segment)               (segment shares)
    segment sizes: n*_e pi_e rounded by largest remainder, n*_e = scale x n_e
    row i of segment k: its modelled columns x_i ~ copula_k (below)

A self-referencing table (a parent-child tree) is a multi-type Galton-Watson
process typed by (segment, depth):

    lam_{k,d} ~ Gamma(c lam_k + children of depth-d nodes of segment k, c + those nodes)
    psi_{k,d} ~ Dirichlet(c psi_k + child-segment counts)
    the forest is drawn with round(scale x n_{k,d}) nodes of segment k at depth d; a
    node of depth d picks its parent at depth d-1 with probability
    proportional to w_i lam_{k_i,d-1} psi_{k_i,d-1}(k)   (w_i ~ Gamma(alpha_k, alpha_k) frailty)

**Columns of a table given a group g** (a segment for entities, a segment block
for facts): a Gaussian copula with correlation matrix R_g over the latent
normal scores of the modelled columns, and one marginal per column:

    R_g ~ p(R | ranks)  (LKJ(eta) prior, rank likelihood; MCMC draws kept)
    discrete column:   p_g ~ Dirichlet(a p_coarse + level counts of the group)
    continuous column: F_g ~ DP(a + n_g, (a F_coarse + sum_i delta_{x_i}) / (a + n_g))
                       (Bayesian bootstrap of the group mixed with its coarse group's law;
                        stored as 49 quantile knots)
    null share:        discrete q_g ~ Beta(a q_coarse + nulls, a (1 - q_coarse) + non-nulls),
                       continuous q_g ~ Beta(1 + nulls, 1 + non-nulls)
                       (with point masses at 0 and 1 under the structural prior)
    sampling: Z ~ N(0, R_g), U = Phi(Z), x_j = F_j^{-1}(U_j)  (null when U_j < q_j)

Columns joined into one categorical (functional dependencies, strongly
associated pairs) are one discrete column over their observed combinations.

**Many-to-one link child -> parent**

    eta_{kl} ~ Gamma(1/2 + links from segment k to segment l, b + n_k m_l),  b = 1 / (2 x overall link rate)
               (with a point mass at 0 under the structural prior)
    phi_{k.} = eta_{k.} m_. / sum_l eta_{kl} m_l           (parent segment law of a child of segment k)
    p_null,k ~ Beta(nulls of segment k + small prior, non-nulls + 1)
    parent row within segment l: by degree weight w_j (Gamma(alpha_l, alpha_l) frailty,
    alpha_l by the method of moments when the counts are over-dispersed), or by the
    DP posterior predictive of the counts per parent row (Bayesian bootstrap of the
    source counts with mass a on Poisson(mean)), conditioned on the segment's total

**Facts**

    single dimension:  rows of anchor row i ~ Poisson(lam_{z_i} w_i), lam_k ~ Gamma(a + rows, b + anchors)
                       (or 1 + Poisson, or the DP count law)
    Poisson block:     N_{kl} ~ Poisson(eta*_{kl} W1_k W2_l), eta_{kl} ~ Gamma(a + C_{kl}, b + E_{kl}),
                       E_{kl} = n_k m_l, eta* = eta m_l / m*_l (scale rule), rows by weight within the blocks
    occupancy:         anchor active ~ Bernoulli(pi_k), pair present ~ Bernoulli(p*_{kl}),
                       pi_k ~ Beta(a + active, b + inactive), p_{kl} ~ Beta(a + present, b + absent pairs at risk)
                       (+ point masses at 0 and 1), mu ~ Gamma(a_m + extra rows, b_m + present pairs),
                       p* = min(1, p m_l / m*_l); rows per present pair 1 + Poisson(mu)
    further keys:      segment of the dimension row ~ P(. | block) ~ sparse Dirichlet
                       backed off to the more informative block dimension
    time column:       period ~ theta_{z_i}, theta_k ~ Dirichlet(period counts + 2 x global profile),
                       uniform within the period; null with q ~ Beta(1 + nulls, 1 + rest)
    columns:           the block's copula and marginals; blocks with few rows use the
                       coarse group of the more informative dimension; parent attributes
                       (condition_on_parents) enter as observed copula coordinates
                       (conditional normal)

**Computed columns**: lookups (a parent's column through a key), derived columns
(expressions) and aggregates (a parent column computed from its children) are
deterministic functions of the generated columns, evaluated in dependency order.
"""


def _one(v):
    """Text from the data or the spec on one line (a newline in a level or a name
    must not start a new markdown line: an indented one is a verbatim block)."""
    return " ".join(str(v).split())


_NOT_SPACE_WS = re.compile(r"[^\S ]")         # whitespace other than a plain space


def _clean(o):
    """A copy with every string (and key) single-line, for model.md."""
    if isinstance(o, str):
        return _one(o) if _NOT_SPACE_WS.search(o) else o
    if isinstance(o, dict):
        return {(_clean(k) if isinstance(k, str) else k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_clean(v) for v in o)
    return o


def to_markdown(S, mean, draws, checks, scope, replica=None):
    S, checks, scope, replica = _clean(S), _clean(checks), _clean(scope), _clean(replica)
    L = [f"# Data model: {S.get('name') or 'specification'}", "",
         f"Exported by relsynth export-model (format {S['format_version']}); fit seed {S['seed']}; "
         f"{len(draws)} posterior draws plus the posterior mean in model.json.", ""]
    nok = sum(c["ok"] for c in checks)
    nfail = sum(not c["ok"] and c.get("level") != "warn" for c in checks)
    L += ["## Verification", "", f"{nok} of {len(checks)} checks pass"
          + (f"; {nfail} condition(s) of the propositions fail." if nfail else
             "; every condition of the propositions holds" + (" (the rest are convergence warnings)."
                                                               if nok < len(checks) else ".")), "",
          "FAIL: a condition under which a proposition holds is violated.  WARN: a convergence diagnostic "
          "of an MCMC block (the model is valid; a longer chain gives more certain draws).", "",
          "| check | result | detail | proved by |", "|---|---|---|---|"]
    for c in checks:
        res = "ok" if c["ok"] else ("WARN" if c.get("level") == "warn" else "FAIL")
        cell = [str(x).replace("|", "/") for x in (c["check"], res, c["detail"], c["reference"])]
        L.append("| " + " | ".join(cell) + " |")
    if replica:
        L += ["", "**Replica against the source** (scorecard of `relsynth` on the replica's output, "
              "posterior mean, same checks as `relsynth run`):", ""]
        for k, v in replica.items():
            L.append(f"- {k}: {v}")
    L += [GENERAL_MD, "## 2. This dataset's model (posterior means)", ""]
    start = len(L)
    for e in S["entity_order"]:
        E = S["entities"][e]
        L.append(f"### {e} (entity, {E['rows']:,} rows)")
        if E["copied"]:
            L.append(f"Copied reference table: its rows are kept as they are; {E['K']} latent segments "
                     "(one per row when small) carry its links.")
        else:
            pi = mean["entities"].get(e, {}).get("pi")
            if pi is not None:
                L.append(f"Segments: K = {len(pi)}; shares " + ", ".join(f"{x:.3f}" for x in pi[:20])
                         + (" ..." if len(pi) > 20 else ""))
        st = E.get("strata")
        if st:
            L.append(f"Strata: segments nested in {', '.join(st['by'] + ([st['attr']] if st.get('attr') else []))} "
                     f"({len(st['labels'])} strata); a row's stratum columns follow from its segment.")
        if e in S["trees"]:
            tr = S["trees"][e]
            cnt = np.asarray(tr["level_counts"])
            L.append(f"Tree {tr['name']}: {cnt.shape[0]} levels, nodes per level {', '.join(str(int(x)) for x in cnt.sum(1))}.")
        L += _attr_md(S, mean, e)
        for name, lk in S["links"].items():
            if lk["child"] != e:
                continue
            th = mean["links"][name]
            cm = "DP counts given the total" if lk["count_dp"] else ("every parent has a child (1 + Poisson)"
                                                                     if lk["min_one"] else "degree weights")
            L += [f"Link {lk['child_key']} -> {lk['parent']}: parent segment law phi ({cm}); "
                  f"null share by segment {', '.join(f'{x:.3f}' for x in th['p_null'][:12])}", "",
                  _md_matrix(th["phi"])]
        L.append("")
    for f in S["fact_order"]:
        fk = S["facts"][f]
        th = mean["facts"][f]
        L.append(f"### {f} (fact, {fk['model']} model, anchor {fk['anchor']}"
                 + (f", partner {fk['partner']})" if fk.get("partner") else ")"))
        if fk["model"] == "single":
            L.append("Rows per anchor row by segment: " + ", ".join(f"{x:.3f}" for x in th["lam"]))
        elif fk["model"] == "occupancy":
            L += ["Anchor activity pi by segment: " + ", ".join(f"{x:.3f}" for x in th["pi"]),
                  f"Presence probabilities p (anchor segment x partner segment); multiplicity by {fk['mult_by']}:", "",
                  _md_matrix(th["p"])]
        else:
            L += ["Block rates eta (anchor segment x partner segment):", "", _md_matrix(th["eta"], fmt="{:.3g}")]
        for x in fk.get("extras", []):
            L.append(f"Key {x['child_key']} -> {x['parent']}: segment law per block, backed off to the {x['backoff']}.")
        if fk.get("time"):
            L.append(f"Time column {fk['time']['col']}: {len(fk['time']['starts'])} {fk['time']['bucket']} periods "
                     f"from {fk['time']['tmin'][:10]} to {fk['time']['tmax'][:10]}; null share {th['time']['null_rate']:.3f}.")
        if fk.get("aux_map"):
            L.append("Conditioned on parent columns: " + ", ".join(f"{p}.{c}" for _, _, p, c in fk["aux_map"]) + ".")
        L += _attr_md(S, mean, f)
        L.append("")
    sec = []
    for ln in L[start:]:
        sec.append(ln)
        if ln and not ln.startswith(("|", "- ", "#", " ")):
            sec.append("")
    L = L[:start] + sec
    L += ["## 3. Generation order", "", " -> ".join(S["table_order"]), ""]
    if S["lookups"] or S["derive"] or S["aggregates"]:
        L += ["## 4. Computed columns", ""]
        for t, lk in S["lookups"].items():
            L += [f"- {t}.{x['column']} = {x['path']}" + (" (mapped through a fixed table)" if x["map"] else "")
                  for x in lk]
        for t, dv in S["derive"].items():
            L += [f"- {t}.{x['column']} = {x['expr']}" for x in dv]
        for a in S["aggregates"]:
            L.append(f"- {a['table']}.{a['column']} = {a['agg']}({a.get('of') or '*'}) over {a['from']} by {a['child_key']}")
        L.append("")
    L += ["## 5. Replicating the synthesis", "",
          "- Exactly, with relsynth: `python run_model.py` (the calls of `relsynth run`; same spec, same seed, "
          "bit-identical output).",
          "- On your own, without relsynth: `python replicate.py model.json --out replica/ [--draw 0|mean] "
          "[--seed 7] [--scale 1.0] [--reference DIR]` (numpy, pandas, scipy).  It draws the core process of "
          "section 1 from the exported posterior draws.",
          "", "What replicate.py leaves to the engine:", ""]
    L += [f"- {s}" for s in scope] or ["- nothing"]
    return "\n".join(L) + "\n"


def _attr_md(S, mean, t):
    a = S["attributes"].get(t)
    th = mean["attributes"].get(t)
    if not a or not th or not a["codecs"]:
        return []
    L = [f"Columns ({len(a['codecs'])} modelled; table-level marginals):", ""]
    for cd, m in zip(a["codecs"], th["global"]):
        tag = " (parent column, conditioning only)" if cd["name"] in a["aux"] else ""
        L.append(f"- {cd['name']}{tag}: {_col_summary(cd, m)}")
    C = np.asarray(th["pooled"])
    if C.ndim == 2 and C.shape[0] > 1:
        names = [c["name"] for c in a["codecs"]]
        iu = np.triu_indices(len(C), 1)
        top = np.argsort(-np.abs(C[iu]))[:5]
        L.append("Strongest latent correlations (pooled): " + ", ".join(
            f"{names[iu[0][i]]} ~ {names[iu[1][i]]} {C[iu][i]:+.2f}" for i in top))
    L.append(f"Groups with their own marginals: {len(th['groups'])}; own copulas: {len(th['copulas'])}.")
    L.append("")
    return L


def to_latex(md_text, title):
    """A LaTeX document of the markdown text (headings, lists, tables, code)."""

    def esc(s):
        s = s.replace("\\", r"\textbackslash{}")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                     ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
            s = s.replace(a, b)
        s = s.replace(r"\textbackslash\{\}", r"\textbackslash{}")
        s = s.replace("->", r"$\rightarrow$").replace(">=", r"$\geq$").replace("<=", r"$\leq$")
        s = s.replace("<", r"\textless{}").replace(">", r"\textgreater{}")
        return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", re.sub(r"`([^`]+)`", r"\\texttt{\1}", s))
    out = [r"\documentclass[10pt]{article}", r"\usepackage[margin=2cm,landscape=false]{geometry}",
           r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}", r"\usepackage{longtable}", r"\usepackage{array}",
           r"\usepackage{xcolor}", r"\usepackage[hidelinks]{hyperref}", r"\setlength{\parindent}{0pt}",
           r"\setlength{\parskip}{4pt}", r"\begin{document}", r"\title{" + esc(title) + "}", r"\date{}", r"\maketitle",
           r"\small"]
    # nothing inside a verbatim block can end it
    md_text = re.sub(r"\\end\s*\{\s*verbatim\s*\}", r"\\end {verbatim}", md_text)
    lines = md_text.splitlines()
    i, inlist = 0, False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            i += 1
            continue
        if inlist and not ln.startswith("- "):
            out.append(r"\end{itemize}")
            inlist = False
        if ln.startswith("## "):
            out.append(r"\section*{" + esc(ln[3:]) + "}")
        elif ln.startswith("### "):
            out.append(r"\subsection*{" + esc(ln[4:]) + "}")
        elif ln.startswith("    "):
            block = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                block.append(lines[i][4:])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            out.append(r"\begin{verbatim}" + "\n" + "\n".join(block) + "\n" + r"\end{verbatim}")
            continue
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|(-+\|)+$", lines[i].replace(" ", "")):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            n = max(len(r) for r in rows)
            wide = n <= 4 and any(len(c) > 40 for r in rows for c in r)
            spec = ("p{3.2cm}p{1cm}p{7.2cm}p{4.2cm}" if wide and n == 4 else "l" * n)
            out.append(r"{\footnotesize\begin{longtable}{" + spec + "}")
            for j, r in enumerate(rows):
                out.append(" & ".join(esc(c) for c in r + [""] * (n - len(r))) + r" \\" + (r" \hline" if j == 0 else ""))
            out.append(r"\end{longtable}}")
            continue
        elif ln.startswith("- "):
            if not inlist:
                out.append(r"\begin{itemize}")
                inlist = True
            out.append(r"\item " + esc(ln[2:]))
        elif ln.strip():
            out.append(esc(ln))
        else:
            out.append("")
        i += 1
    if inlist:
        out.append(r"\end{itemize}")
    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def _code_text(v, limit=120):
    """Text safe inside the generated script's docstring: one line, no quotes or
    backslashes (nothing from the spec is ever formatted into its code)."""
    return re.sub(r"[^A-Za-z0-9 _.,:;()+\-/]", "_", _one(v))[:limit]


RUN_MODEL = '''#!/usr/bin/env python3
"""Run the data model of {spec} exactly as `relsynth run` does (bit-identical output).

    python run_model.py [--out synthetic] [--datasets {datasets}] [--scale 1.0]

Needs relsynth (the package this model was exported from) and access to the source
named in the specification (for the fit).  Every step is a public relsynth call:
load the source, fit the Bayesian model (latent segments by blocked Gibbs, then
the conjugate posteriors of every block), then for each dataset draw theta from
the posterior (parameter stream m) and the tables given theta (data stream m).
"""
import argparse
from pathlib import Path

from relsynth.cli import _dataset_target
from relsynth.data import coerce, write_tables
from relsynth.pushdown import load_for_fit
from relsynth.engine import Synthesizer
from relsynth.spec import load_spec

SPEC = Path(__file__).with_name("spec.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output folder (default: the spec's target)")
    ap.add_argument("--datasets", type=int, default=None, help="number of synthetic datasets")
    ap.add_argument("--scale", type=float, default=None)
    a = ap.parse_args()
    out = str(Path(a.out).resolve()) if a.out else None   # --out is relative to the current folder
    spec = load_spec(str(SPEC))
    if a.scale is not None:
        spec["synthesis"]["scale"]["default"] = a.scale
    n = a.datasets or int(spec["synthesis"]["posterior"].get("datasets", 1))
    data = load_for_fit(spec)                         # the source (large facts: sample + SQL statistics)
    syn = Synthesizer(spec).fit(data.tables, stats=data.stats)   # posterior p(theta, z | data)
    for i in range(n):
        fake = syn.sample(index=i)                    # theta^(i) ~ posterior, D*^(i) ~ p(D* | theta^(i))
        fake = {{t: coerce(df, spec["tables"][t]) for t, df in fake.items()}}
        write_tables(spec, fake, _dataset_target(spec, out, i))
        print(f"dataset {{i + 1}}: " + ", ".join(f"{{t}} {{len(df):,}}" for t, df in fake.items()))


if __name__ == "__main__":
    main()
'''


def write_pdf(tex_path):
    """pdflatex twice (longtable widths); returns the pdf path or None.  pdflatex
    runs without shell escape, reads and writes only in a private temporary folder
    (openin_any / openout_any paranoid), and only model.pdf is copied back: it never
    has write access to the folder of replicate.py."""
    exe = shutil.which("pdflatex")
    if not exe:
        return None
    tex_path = Path(tex_path)
    env = {**os.environ, "openin_any": "p", "openout_any": "p", "shell_escape": "f"}
    with tempfile.TemporaryDirectory(prefix="relsynth-tex-") as d:
        shutil.copyfile(tex_path, Path(d) / "model.tex")
        for _ in range(2):
            # fixed program found on PATH, fixed arguments, no shell, sandboxed TeX settings
            r = subprocess.run([exe, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",  # noqa: S603  # nosec B603
                                "model.tex"], cwd=d, capture_output=True, text=True, timeout=300, env=env)
            if r.returncode != 0:
                return None
        if not (Path(d) / "model.pdf").exists():
            return None
        pdf = tex_path.with_suffix(".pdf")
        shutil.copyfile(Path(d) / "model.pdf", pdf)
    return pdf


def replica_check(model, spec, real, seed=7, reference=None, real_stats=None):
    """Scorecard of the standalone replica's output against the source (posterior
    mean), with relsynth's own evaluation."""
    from .data import coerce
    from .evaluate import evaluate
    from . import replica
    t0 = time.time()
    if any(c not in real[t].columns for t in real if t in spec["tables"] for c in spec["tables"][t]["columns"]):
        # in-memory source tables get the preparation of loaded ones (computed columns filled)
        from .prepare import computed_columns, prepare_source
        comp = computed_columns(spec)
        real = prepare_source(spec, {t: coerce(df, spec["tables"][t], optional=set(comp[t]))
                                     for t, df in real.items() if t in spec["tables"]})[0]
    out = replica.generate(model, "mean", seed, 1.0, reference)
    fake = {}
    for t, df in out.items():
        if t in spec["tables"]:
            fake[t] = coerce(df.reset_index(drop=True), spec["tables"][t])
    # score the tables the replica generates (the rest are engine only, listed in
    # replica_scope): relationships and rules among them
    have = set(fake)
    sub = dict(spec, tables={t: ts for t, ts in spec["tables"].items() if t in have},
               relationships=[r for r in spec["relationships"] if r["child"] in have and r["parent"] in have],
               rules=[r for r in spec.get("rules", []) if r.get("table") in have
                      and r.get("from", r.get("table")) in have])
    rep = evaluate(sub, {t: real[t] for t in have if t in real}, fake, privacy=False,
                   real_stats={t: v for t, v in (real_stats or {}).items() if t in have})
    sc = rep.get("scorecard", {})
    missing = sorted(t for t in real if t not in have)
    return {"replica scorecard": f"{sc.get('passed')}/{sc.get('total')} checks pass"
            + (f" (tables not generated: {', '.join(missing)})" if missing else ""),
            "replica rows": ", ".join(f"{t} {len(df):,}" for t, df in out.items()),
            "seconds": round(time.time() - t0, 1),
            "failed": [f"{f['area']}: {f['metric']} = {f['value']} (limit {f['limit']})" for f in sc.get("failed", [])][:40]}


def export_model(syn, out_dir, spec_path=None, draws=5, formats=FORMATS, real=None, check_replica=True, stan=False,
                 real_stats=None, spec_raw=None):
    """Write the model files (spec_raw: the loaded spec, used for the bundle's
    spec.yaml when there is no spec file); returns {"files": [...], "checks": [...], "replica": ...}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S, scope = structure(syn)
    mean, D = posterior_draws(syn, draws)
    checks = verify(S, mean, D, syn)
    if stan:
        from .stancheck import verify as sv
        r = sv()
        if r.get("available"):
            _check(checks, "independent check with Stan (conjugate blocks and copula)", r["all_ok"],
                   f"{sum(b['ok'] for b in r['blocks'])}/{len(r['blocks'])} parameters agree "
                   f"(largest z {max(b['z_mean'] for b in r['blocks']):.2f})",
                   "posterior draws of the engine against Stan's NUTS on the textbook model")
        else:
            _check(checks, "independent check with Stan", True, r.get("reason", "skipped"), "optional")
    model = {"structure": S, "mean": mean, "draws": D, "replica_scope": scope, "verification": checks}
    files = []
    rep = None
    if check_replica and real is not None:
        try:
            rep = replica_check(model, syn.user_spec, real, seed=syn.cfg["seed"],
                                reference={e: syn._copy_rows[e] for e in syn.copied}, real_stats=real_stats)
        except Exception as e:  # noqa: BLE001
            rep = {"replica scorecard": f"not computed ({e!r})"}
        model["replica_check"] = rep
    if "json" in formats:
        p = out / "model.json"
        p.write_text(json.dumps(model, indent=1, allow_nan=False))
        files.append(p)
    md = to_markdown(S, mean, D, checks, scope, {k: v for k, v in (rep or {}).items() if k != "failed"} or None)
    if "md" in formats or "pdf" in formats:
        p = out / "model.md"
        p.write_text(md)
        files.append(p)
    if "pdf" in formats:
        tex = out / "model.tex"
        tex.write_text(to_latex(md, f"Data model: {S.get('name') or 'specification'}"))
        files.append(tex)
        pdf = write_pdf(tex)
        if pdf:
            files.append(pdf)
    if "py" in formats:
        src = Path(__file__).with_name("replica.py")
        p = out / "replicate.py"
        shutil.copyfile(src, p)
        files.append(p)
        p = out / "relsynth_expr.py"          # the engine's rule language, used by replicate.py
        shutil.copyfile(Path(__file__).with_name("expr.py"), p)
        files.append(p)
        p = out / "run_model.py"
        try:
            n_ds = int(syn.cfg.get("posterior", {}).get("datasets", 1))
        except (TypeError, ValueError):
            n_ds = 1
        p.write_text(RUN_MODEL.format(spec=_code_text(Path(spec_path).name) if spec_path else "spec.yaml",
                                      datasets=n_ds))
        files.append(p)
        if spec_path or spec_raw:
            import copy

            import yaml
            from .data import mask_spec_urls
            from .spec import read_yaml
            if spec_path:
                raw, base = read_yaml(spec_path), Path(spec_path).parent
            else:
                raw = {k: copy.deepcopy(v) for k, v in spec_raw.items() if not k.startswith("_")}
                base = Path(spec_raw.get("_base_dir", "."))
            src_cfg = raw.get("source") or {}
            for k in ("path", "dir"):
                if isinstance(src_cfg.get(k), str) and not Path(src_cfg[k]).is_absolute():
                    src_cfg[k] = str((base / src_cfg[k]).resolve())
            # credentials stay out of the bundle: the URL's password is ***, url_env names
            # the environment variable that supplies the real URL to run_model.py
            (out / "spec.yaml").write_text(yaml.safe_dump(mask_spec_urls(raw), sort_keys=False, width=110),
                                           encoding="utf-8")
            files.append(out / "spec.yaml")
    return {"files": [str(f) for f in files], "checks": checks, "replica": rep}
