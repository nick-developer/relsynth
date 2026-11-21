# SPDX-License-Identifier: Apache-2.0
"""Generator selection: which generator plug-in models each table's attribute columns
(synthesis.generator_selection; docs/generators.md, "Choosing a generator").

    off    the spec's choice (``synthesis.generator`` and each table's ``generator``); the default
    auto   rules on each table's traits (rows, column kinds, compute), from published comparisons
    trial  fit, sample and score every candidate with ``seeds`` seeds on the real data, keep the best

A table whose spec names a generator keeps it in every mode: the user's choice wins.  Copied
tables and views are not generated and take no part.  Selection reads the real data, so it is
refused in differential privacy mode, where a data-dependent choice would need its own budget.

Trial ranks the candidates by privacy first, then fidelity (Prop. 1845): fewer failed privacy
items of the scorecard (distance to closest record, exact matches, rare-text copies), then the
higher mean share of passed scorecard items over the seeds, then the spec's own generator, then
the registry's order.  Each candidate is fitted on the same rows with the same seeds, so the
comparison is paired.  Every candidate's result is kept in the decision record.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

from ..marginals import MODELLED
from .contract import GeneratorError
from .registry import DEFAULT, check, default_registry

log = logging.getLogger("relsynth")

MODES = ("off", "auto", "trial")
MAX_SEEDS = 10
# scorecard metrics that measure disclosure rather than fidelity (relsynth.evaluate)
PRIVACY_METRICS = frozenset({"dcr_ratio", "exact_match_rate", "text_rare_exact_copies", "text_rare_ngram_copies",
                             "text_rare_token_leaks"})
# auto: thresholds of the rules (rows of the table being modelled)
SMALL_ROWS = 1_000
NEURAL_MIN_ROWS = 10_000
DIFFUSION_MIN_ROWS = 50_000
CATEGORICAL_SHARE = 0.5


@dataclass
class Selection:
    """A selection decision: the generator chosen per table, and how."""
    mode: str
    chosen: dict = field(default_factory=dict)          # table -> generator name
    reasons: dict = field(default_factory=dict)         # table -> one line
    candidates: list = field(default_factory=list)      # trial: one result per candidate
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"mode": self.mode, "chosen": dict(self.chosen), "reasons": dict(self.reasons),
                "candidates": [dict(c) for c in self.candidates], "seconds": round(self.seconds, 1)}


def config(spec: dict) -> dict:
    """synthesis.generator_selection with its defaults."""
    g = (spec.get("synthesis") or {}).get("generator_selection") or {}
    return {"mode": g.get("mode", "off"), "seeds": g.get("seeds", 1), "candidates": g.get("candidates")}


def validate(spec: dict) -> list[str]:
    """Errors of synthesis.generator_selection."""
    raw = (spec.get("synthesis") or {}).get("generator_selection")
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return ["synthesis.generator_selection must be a mapping {mode, seeds, candidates}"]
    errs = []
    unknown = set(raw) - {"mode", "seeds", "candidates"}
    if unknown:
        errs.append(f"synthesis.generator_selection: unknown key(s) {', '.join(sorted(unknown))}")
    g = config(spec)
    if g["mode"] not in MODES:
        errs.append(f"synthesis.generator_selection.mode must be one of {', '.join(MODES)}")
    s = g["seeds"]
    if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= MAX_SEEDS:
        errs.append(f"synthesis.generator_selection.seeds must be an integer from 1 to {MAX_SEEDS}")
    c = g["candidates"]
    if c is not None:
        if not isinstance(c, list) or not c or not all(isinstance(x, str) and x for x in c):
            errs.append("synthesis.generator_selection.candidates must be a non-empty list of generator names")
        else:
            known = set(default_registry().names())
            missing = sorted(set(c) - known)
            if missing:
                errs.append(f"synthesis.generator_selection.candidates: unknown generator(s) {', '.join(missing)}"
                            f" (installed: {', '.join(sorted(known))})")
    return errs


def selectable(spec: dict) -> list[str]:
    """Tables whose generator the selection decides: generated tables with modelled columns and no
    generator of their own in the spec."""
    out = []
    for t, T in spec["tables"].items():
        if T.get("role") == "view" or T.get("mode") == "copy" or T.get("generator") is not None:
            continue
        if any(c.get("type") in MODELLED for c in T.get("columns", {}).values()):
            out.append(t)
    return out


def _rows(t, real, stats) -> int:
    if stats and t in stats:
        return int(stats[t].n_rows)
    return len(real[t]) if real is not None and t in real else 0


def _compatible(spec, t, names, pushed) -> list[str]:
    reg = default_registry()
    ok = []
    for n in names:
        try:
            e = reg.entry(n)
        except GeneratorError:
            continue
        if getattr(e.cls, "unavailable", None) and e.cls.unavailable():
            continue
        if not check(e.cls, spec, t, pushed=t in pushed)[0]:
            ok.append(n)
    return ok


def _device(spec, name) -> str:
    """The device the generator would get under the spec's compute settings ('cpu' when unknown)."""
    from ..compute import resolve
    try:
        cls = default_registry().entry(name).cls
        return resolve((spec.get("synthesis") or {}).get("compute"), cls.capabilities.gpu, who=name,
                       base_dir=spec.get("_base_dir")).device
    except Exception:  # noqa: BLE001 - no device information is the CPU case
        return "cpu"


def auto_rule(spec, t, n_rows, available, device=None) -> tuple[str, str]:
    """(generator, reason) for table t by the documented rules (docs/generators.md):

    1. fewer than 1,000 rows: copula (a parametric Bayesian model; flexible learners overfit small
       samples and copy rare values);
    2. 50,000 rows or more, a diffusion generator installed and a GPU: ddpm (TabDDPM, Kotelnikov et
       al. 2023);
    3. at least half the modelled columns categorical or boolean: cart (sequential CART, Reiter 2005;
       synthpop, Nowok et al. 2016; best utility across datasets in Dankar and Ibrahim 2021);
    4. 10,000 rows or more and a variational autoencoder installed: tvae (Xu et al. 2019);
    5. otherwise copula (Gaussian copula, Patki et al. 2016): continuous columns, monotone dependence.

    A rule whose generator is not available (not installed or not able to model the table) is skipped.
    """
    device = device or (lambda name: "cpu")
    cols = [c for c in spec["tables"][t].get("columns", {}).values() if c.get("type") in MODELLED]
    cat = sum(c["type"] in ("categorical", "boolean") for c in cols) / len(cols) if cols else 0.0
    rules = [
        (n_rows < SMALL_ROWS, "copula", f"{n_rows:,} rows (< {SMALL_ROWS:,}): a parametric Bayesian model"),
        (n_rows >= DIFFUSION_MIN_ROWS and device("ddpm") != "cpu", "ddpm",
         f"{n_rows:,} rows and a GPU: tabular diffusion"),
        (cat >= CATEGORICAL_SHARE, "cart", f"{cat:.0%} of modelled columns categorical or boolean: sequential CART"),
        (n_rows >= NEURAL_MIN_ROWS, "tvae", f"{n_rows:,} rows (>= {NEURAL_MIN_ROWS:,}): variational autoencoder"),
        (True, "copula", "mostly numeric columns: Gaussian copula"),
    ]
    for cond, name, why in rules:
        if cond and name in available:
            return name, why
    # nothing the rules name can model the table: keep the spec's default if it can, else the first available
    fallback = DEFAULT if DEFAULT in available else (available[0] if available else DEFAULT)
    return fallback, "no rule's generator can model this table"


def auto(spec, real, stats=None, pushed=()) -> Selection:
    g = config(spec)
    names = g["candidates"] or default_registry().names()
    sel = Selection("auto")
    for t in selectable(spec):
        avail = _compatible(spec, t, names, set(pushed))
        name, why = auto_rule(spec, t, _rows(t, real, stats), avail, lambda n: _device(spec, n))
        sel.chosen[t], sel.reasons[t] = name, why
    return sel


def with_choice(spec: dict, chosen: dict) -> dict:
    """A copy of the spec with each table of `chosen` given that generator."""
    s = copy.deepcopy(spec)
    for t, name in chosen.items():
        s["tables"][t]["generator"] = name
    return s


def _score(spec, real, stats, seeds, fit_sample_evaluate) -> dict:
    """Mean scorecard share and failed privacy items of one candidate over the seeds."""
    shares, priv, failed = [], [], []
    base = int(spec["synthesis"].get("seed", 0))
    for k in range(seeds):
        s = copy.deepcopy(spec)
        s["synthesis"]["seed"] = base + k
        rep = fit_sample_evaluate(s, real, stats)
        sc = rep.get("scorecard") or {}
        items = sc.get("items") or []
        total = sc.get("total") or len(items)
        shares.append((sc.get("passed", 0) / total) if total else 1.0)
        priv.append(sum(1 for i in items if i.get("metric") in PRIVACY_METRICS and not i.get("pass")))
        failed.append(sc.get("total", 0) - sc.get("passed", 0))
    return {"score": round(sum(shares) / len(shares), 6), "privacy_failed": max(priv),
            "failed": round(sum(failed) / len(failed), 2)}


def trial(spec, real, stats=None, pushed=(), fit_sample_evaluate=None) -> Selection:
    """Fit, sample and score each candidate (one generator for every selectable table) and keep the
    best by the ranking of the module docstring.  fit_sample_evaluate(spec, real, stats) -> the raw
    evaluation report (relsynth.evaluate); the pipeline passes its own."""
    g = config(spec)
    tables = selectable(spec)
    sel = Selection("trial")
    if not tables:
        return sel
    names = g["candidates"] or default_registry().names()
    own = (spec.get("synthesis") or {}).get("generator", DEFAULT)
    order = {n: i for i, n in enumerate(names)}
    results = []
    for name in names:
        if any(name not in _compatible(spec, t, [name], set(pushed)) for t in tables):
            results.append({"generator": name, "error": "cannot model every table"})
            continue
        t0 = time.time()
        log.info("generator selection: trying %s with %d seed(s)", name, g["seeds"])
        try:
            r = _score(with_choice(spec, dict.fromkeys(tables, name)), real, stats, g["seeds"], fit_sample_evaluate)
        except Exception as e:  # noqa: BLE001 - a candidate that fails is a result, not a crash
            results.append({"generator": name, "error": type(e).__name__ + ": " + str(e)[:300]})
            continue
        results.append({"generator": name, **r, "seconds": round(time.time() - t0, 1)})
    ok = [r for r in results if "error" not in r]
    if not ok:
        raise GeneratorError("generator selection: no candidate could be fitted: "
                             + "; ".join(f"{r['generator']}: {r['error']}" for r in results))
    best = min(ok, key=lambda r: (r["privacy_failed"], -r["score"], r["generator"] != own, order[r["generator"]]))
    sel.candidates = results
    for t in tables:
        sel.chosen[t] = best["generator"]
        sel.reasons[t] = (f"trial: {best['privacy_failed']} privacy item(s) failed, scorecard share "
                          f"{best['score']:.3f} over {g['seeds']} seed(s)")
    return sel


def select(spec, real, stats=None, pushed=(), fit_sample_evaluate=None) -> Selection | None:
    """The selection the spec asks for, or None when its mode is off."""
    mode = config(spec)["mode"]
    if mode == "off":
        return None
    t0 = time.time()
    sel = auto(spec, real, stats, pushed) if mode == "auto" else trial(spec, real, stats, pushed, fit_sample_evaluate)
    sel.seconds = time.time() - t0
    for t, name in sel.chosen.items():
        log.info("generator selection (%s): %s -> %s (%s)", mode, t, name, sel.reasons[t])
    return sel
