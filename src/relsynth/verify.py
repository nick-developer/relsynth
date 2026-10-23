"""Model checks on real data, where no ground truth exists (R1-R7).

On fake data the generating model is known and the checks compare with it
(tests/science).  On real data these checks stand in for it (proofs, Section
"Checks without ground truth"):

  R1 scorecard: every fidelity, rule and privacy check passes
  R2 copula convergence: Metropolis acceptance and split R-hat of the chains
  R3 posterior predictive checks: each real key statistic (rows per parent,
     parents without rows, dispersion, tree depth, seasonality, fact means) lies
     within the spread of D synthetic databases, each from its own posterior draw
  R4 scale rule: doubling an anchor doubles its fact; doubling a partner keeps
     the rows per anchor (Prop. "Scale rule")
  R5 structural validity: foreign keys resolve, hierarchies have no cycles
  R6 stability: a second fit with another seed finds the same segments (ARI)
  R7 reproducibility: the same seed gives the same synthetic database

    report = relsynth.verify("spec.yaml", draws=5)
    report.passed, report.total, report.to_markdown()

The result holds aggregates only (no rows, no category values) and can be
shared; `relsynth verify spec.yaml --bundle b.json` adds it to a diagnostics
bundle for `relsynth doctor`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("relsynth.verify")


def ari(a, b) -> float:
    """Adjusted Rand index of two labelings."""
    from scipy.special import comb
    ct = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy()
    s = comb(ct, 2).sum()
    sa, sb = comb(ct.sum(1), 2).sum(), comb(ct.sum(0), 2).sum()
    e = sa * sb / comb(ct.sum(), 2)
    d = 0.5 * (sa + sb) - e
    return float((s - e) / d) if d else 1.0


def key_stats(spec: dict, db: dict, skip=()) -> dict[str, float]:
    """Scalar statistics every synthetic copy should reproduce (tables in skip
    are samples, whose per-parent counts would be wrong, and are left out)."""
    from .evaluate import keystr
    from .relational import tree_depth
    T, out = spec["tables"], {}
    for r in spec["relationships"]:
        c, p = r["child"], r["parent"]
        if r.get("composite") or c in skip or c not in db or p not in db:
            continue                      # nested references: covered by the scorecard
        if r["kind"] == "self_reference":
            pk = T[c]["primary_key"]
            par = pd.Index(keystr(db[c][pk])).get_indexer(keystr(db[c][r["child_key"]]).fillna("__none__"))
            d = tree_depth(par.astype(np.int64).copy())
            out[f"{r['name']}: mean depth"] = float(d.mean()) if len(d) else 0.0
            out[f"{r['name']}: share of leaves"] = float((np.bincount(par[par >= 0], minlength=len(par)) == 0).mean())
            continue
        cnt = keystr(db[c][r["child_key"]]).value_counts().reindex(keystr(db[p][T[p]["primary_key"]]), fill_value=0)
        out[f"{r['name']}: rows per parent"] = float(cnt.mean())
        out[f"{r['name']}: share of parents without rows"] = float((cnt == 0).mean())
        out[f"{r['name']}: dispersion"] = float(cnt.var() / max(cnt.mean(), 1e-9))
    for t, ts in T.items():
        if t not in db or t in skip:
            continue
        if ts["role"] == "fact" and ts.get("fact", {}).get("time_column"):
            s = pd.to_datetime(db[t][ts["fact"]["time_column"]], errors="coerce")
            if s.notna().any():
                out[f"{t}: share of rows in the busiest month"] = float(
                    s.dt.to_period("M").value_counts(normalize=True).max())
        for c, cs in ts["columns"].items():
            if cs["type"] in ("float", "integer") and ts["role"] == "fact" and c in db[t]:
                out[f"{t}.{c}: mean"] = float(pd.to_numeric(db[t][c], errors="coerce").mean())
    return out


def _digest(db: dict) -> str:
    dig = hashlib.sha256()
    for t in sorted(db):
        dig.update(t.encode())
        dig.update(pd.util.hash_pandas_object(db[t].astype(str), index=False).to_numpy().tobytes())
    return dig.hexdigest()


@dataclass
class VerifyReport:
    """The R1-R7 checks of one spec: `checks` (id, what, value, target, pass),
    `errors` of checks that could not run, and the run's diagnostics bundle."""
    name: str
    checks: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    seconds: float = 0.0
    bundle: dict = field(default_factory=dict, repr=False)

    @property
    def passed(self) -> int:
        return sum(bool(c["pass"]) for c in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def ok(self) -> bool:
        return self.passed == self.total and not self.errors

    def failed(self) -> list[dict]:
        return [c for c in self.checks if not c["pass"]]

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "total": self.total, "seconds": self.seconds,
                "checks": self.checks, "errors": [{k: e[k] for k in ("step", "error")} for e in self.errors]}

    def to_json(self, path: str | Path | None = None) -> str:
        txt = json.dumps(self.to_dict(), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        if path is not None:
            Path(path).write_text(txt, encoding="utf-8")
        return txt

    def to_markdown(self) -> str:
        md = [f"# Model checks on real data: {self.name}", "",
              f"{self.passed} of {self.total} checks passed in {self.seconds}s.", "",
              "| id | check | value | target | ok |", "|---|---|---|---|---|"]
        md += [f"| {c['id']} | {c['what']} | {c['value']} | {c['target']} | {'yes' if c['pass'] else 'NO'} |"
               for c in self.checks]
        if self.errors:
            md += ["", f"{len(self.errors)} check(s) could not run: " +
                   "; ".join(f"{e['step']}: {e['error']}" for e in self.errors)]
        return "\n".join(md) + "\n"

    def save_bundle(self, path: str | Path) -> Path:
        """The diagnostics bundle with the checks added (aggregates only)."""
        from .diagnostics import _jsonable, write_bundle
        b = dict(self.bundle)
        b["replication"] = self.to_dict()
        write_bundle(_jsonable(b), path)
        return Path(path)


def verify(spec, *, draws: int = 5, stability: bool = True, out=None, privacy: bool = True,
           pushdown: str = "off", seed: int | None = None) -> VerifyReport:
    """Run R1-R7 on the spec's source (see the module docstring).  draws: synthetic
    copies for R3; stability=False skips the two extra fits (R6, R7) on very large
    data; out: also write the first synthetic database there.  Pushed-down facts
    are samples, so their per-parent statistics are left out of R3; the default
    loads every table whole."""
    from .api import _resolved
    from .data import coerce
    from .engine import Synthesizer
    from .pipeline import execute
    from .pushdown import load_for_fit
    t0 = time.time()
    s = _resolved(spec)
    if seed is not None:
        s["synthesis"]["seed"] = int(seed)
    run = execute(s, out=out if out else False, write=bool(out), privacy=privacy, pushdown=pushdown)
    rep = VerifyReport(name=str(s.get("name") or "spec"), bundle=run.bundle)
    rep.errors = list(run.errors)

    def add(cid, what, value, target, ok):
        rep.checks.append({"id": cid, "what": what, "value": value, "target": target, "pass": bool(ok)})

    def guarded(cid, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a failed check is reported, the others still run
            rep.errors.append({"step": cid, "error": repr(e), "traceback": traceback.format_exc()})
            add(cid, "check failed to run", repr(e), "-", False)

    if run.synthesizer is None or run.data is None or run.report is None:
        rep.seconds = round(time.time() - t0, 1)
        return rep
    syn, fake, real = run.synthesizer, run.data, run.fit_data.tables
    sampled = set(run.fit_data.stats)
    sc = run.report["scorecard"]
    add("R1", "scorecard checks passed", f"{sc['passed']}/{sc['total']}", "all", sc["passed"] == sc["total"])

    def r2():
        for t, cops in (run.bundle.get("model") or {}).get("copulas", {}).items():
            for g, info in cops.items():
                if info.get("mode") not in ("bayes", "rank") or not info.get("kept_draws", 9):
                    continue
                if info.get("mode") == "bayes":
                    acc = info.get("mh_acceptance") or 0
                    add("R2", f"{t}/{g} Metropolis acceptance", acc, "0.1-0.6", 0.1 <= acc <= 0.6)
                if info.get("rhat_logdet") is None and info.get("rhat_max_entry") is None:
                    continue
                rh, dr = info.get("rhat_max_entry") or info.get("rhat_logdet") or 9, info.get("drift_max")
                add("R2", f"{t}/{g} split R-hat {rh:.2f}, largest drift of a correlation between chain halves",
                    dr, "R-hat <= 1.1 or drift <= 0.02", rh <= 1.1 or (dr is not None and dr <= 0.02))
    guarded("R2", r2)

    def r3():
        T = s["tables"]
        rs = key_stats(s, real, skip=sampled)
        sims = []
        for i in range(1, draws + 1):
            log.info("verify R3: posterior predictive copy %d of %d", i, draws)
            d = syn.sample(index=i)              # its own parameter draw, then new data
            sims.append(key_stats(s, {t: coerce(df, T[t]) for t, df in d.items() if t in T}, skip=sampled))
        for k, v in rs.items():
            col = np.array([x.get(k, np.nan) for x in sims], dtype=float)
            col = col[np.isfinite(col)]
            if len(col) < 2 or not np.isfinite(v):
                continue
            sd = max(col.std(ddof=1), 1e-3 * max(abs(col.mean()), 1e-9))
            z = abs(v - col.mean()) / sd
            rel = abs(v - col.mean()) / max(abs(v), 1e-9)
            # the spread covers sampling noise given one posterior draw, not all of the
            # parameter uncertainty, so differences under 3% also count as reproduced
            add("R3", f"{k}: real {v:.4g}, synthetic {col.mean():.4g} +- {col.std(ddof=1):.2g}", round(float(z), 2),
                "|z| <= 3 or within 3%", z <= 3 or rel <= 0.03)
            rep.checks[-1]["sign"] = 1 if v >= col.mean() else -1
    if draws >= 2:
        guarded("R3", r3)

    def r4():
        T = s["tables"]
        n_real = run.fit_data.source_rows
        for f, ts in T.items():
            if ts["role"] != "fact" or f not in syn.fact_info:
                continue
            info = syn.fact_info[f]
            anchor = info["anchor"]
            hdr = syn.plan.headers.get(f)
            if hdr:                   # lines of generated orders: rows per order-placing parent
                anchor = hdr["scale_from"]
                if anchor is None:
                    continue
            if info.get("nested") or T[anchor].get("mode") == "copy" or anchor in getattr(syn, "grain", {}):
                continue              # nested facts, copied or grain anchors: not sized by a scale factor
            base = n_real[f] / max(n_real[anchor], 1)
            tests = [("anchor x2", {anchor: 2.0}, 1.0)]
            partner = info["fks"][1]["parent"] if len(info["fks"]) > 1 else None
            if partner in T and T[partner].get("mode") != "copy":
                rel = info.get("rel")
                want = 1.0
                if rel is not None and rel.kind == "occ":
                    # occupancy: blocks where every pair is present (a full grid) gain
                    # rows with the partner; the others keep rows per anchor
                    blk = rel.e1.z[rel.u] * rel.e2.K + rel.e2.z[rel.v]
                    full = rel.p.reshape(-1)[blk] >= 1
                    want = 1.0 + float(rel.c[full].sum() / max(rel.c.sum(), 1))
                tests.append((f"{partner} x2", {partner: 2.0}, want))
            for label, sc_, want in tests:
                log.info("verify R4: %s, %s", f, label)
                o = syn.sample({"default": 1.0, "tables": sc_}, index=0)
                per = len(o[f]) / max(len(o[anchor]), 1)
                add("R4", f"{f}, {label}: rows per {anchor} {per:.3f} (real {base:.3f}, expected ratio {want:.3f})",
                    round(per / base, 4) if base else None, f"{want:.3g} +- 0.05",
                    bool(base) and abs(per / base - want) <= 0.05 * want)
    guarded("R4", r4)

    def r5():
        ev = run.report
        for n, d in ev.get("relations", {}).items():
            if "fk_integrity" in d:
                add("R5", f"{n} FK integrity", d["fk_integrity"], 1.0, d["fk_integrity"] == 1.0)
        for n, d in ev.get("trees", {}).items():
            add("R5", f"{n} cycles", d["cycles_syn"], 0, d["cycles_syn"] == 0)
    guarded("R5", r5)

    def r6():
        log.info("verify R6: second fit with another seed")
        s2 = json.loads(json.dumps(s, default=str))
        s2["synthesis"]["seed"] = int(s["synthesis"]["seed"]) + 1000
        fd = load_for_fit(s2, mode=pushdown)
        syn2 = Synthesizer(s2).fit(fd.tables, stats=fd.stats)
        for e in syn.E:
            if syn.E[e].n < 50 or syn.E[e].K < 2 or getattr(syn.E[e], "fixed", False) or e not in syn2.E:
                continue          # copied tables have fixed segments (one per row)
            v = ari(syn.E[e].z, syn2.E[e].z)
            add("R6", f"{e}: segment agreement between two seeds (ARI), K {syn.E[e].K} vs {syn2.E[e].K}",
                round(v, 3), ">= 0.7", v >= 0.7)

    def r7():
        log.info("verify R7: refit with the same seed")
        fd = load_for_fit(s, mode=pushdown)
        o = Synthesizer(s).fit(fd.tables, stats=fd.stats).sample(index=0)
        T = s["tables"]
        a = {t: coerce(df, T[t]) for t, df in o.items() if t in T}
        b = {t: coerce(df, T[t]) for t, df in fake.items() if t in T}
        same = _digest(a) == _digest(b)
        add("R7", "same seed, same synthetic database", same, True, same)
    if stability:
        guarded("R6", r6)
        guarded("R7", r7)
    rep.seconds = round(time.time() - t0, 1)
    return rep
