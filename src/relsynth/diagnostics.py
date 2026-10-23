"""Diagnostics bundle: everything needed to find model-layer problems, and
nothing that reveals source rows.

The bundle holds: the spec (credentials masked), a profile of every source
column (types, null rates, distinct ratios: no values, no level names), the
fitted model's summary (segments, merges, copula convergence, dispersion),
the evaluation report and scorecard, warnings, errors with tracebacks, and
timings.  Counts below `privacy.suppress_small_counts` are blanked.
"""
from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd

from .data import is_stringlike, mask_spec_urls, mask_url

from .spec import spec_hash

BUNDLE_VERSION = 1


def _mask(url):
    return mask_url(url)


def clean_spec(spec):
    return mask_spec_urls({k: v for k, v in spec.items() if not k.startswith("_")})


def profile(spec, tables, k_small=5):
    out = {}
    for t, df in tables.items():
        cols = {}
        for c, cs in spec["tables"][t]["columns"].items():
            s = df[c]
            nn = int(s.notna().sum())
            d = {"declared": cs["type"], "null_rate": round(float(s.isna().mean()), 4),
                 "distinct": int(s.nunique()), "distinct_ratio": round(float(s.nunique() / max(nn, 1)), 4)}
            if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
                v = pd.to_numeric(s, errors="coerce").dropna()
                d["integer_valued"] = bool(len(v) and np.allclose(v, np.round(v)))
                d["monotone"] = bool(len(v) > 2 and (v.is_monotonic_increasing or v.is_monotonic_decreasing))
                if len(v):
                    d["skew"] = round(float(v.skew()), 3) if len(v) > 2 else 0.0
            elif cs["type"] == "binary":
                pass
            elif cs["type"] in ("categorical", "text", "pii") or is_stringlike(s):
                sv = s.dropna().astype(str)
                if len(sv):
                    d["mean_len"] = round(float(sv.str.len().mean()), 1)
                    d["mean_tokens"] = round(float(sv.str.split().str.len().mean()), 1)
                    vc = sv.value_counts()
                    d["singleton_share"] = round(float((vc == 1).sum() / max(len(vc), 1)), 4)
                    d["top_level_share"] = round(float(vc.iloc[0] / len(sv)), 4) if vc.iloc[0] >= k_small else None
                    d["looks_like_email"] = round(float(sv.str.contains(r"@\w").mean()), 3)
            cols[c] = d
        out[t] = {"rows": len(df), "columns": cols, "key_candidates": key_candidates(spec, t, df)}
    return out


def key_candidates(spec, t, df, max_levels=50):
    """Duplicate rates of column pairs that could be a unique key (grain): two
    foreign keys, or a foreign key and a low-cardinality categorical column.
    Rates only, no values.  A rate of 0 on many rows suggests declaring the key."""
    ts = spec["tables"][t]
    fks = [r["child_key"] for r in spec["relationships"]
           if r["child"] == t and not r.get("composite") and r["kind"] == "many_to_one"]
    cats = [c for c, cs in ts["columns"].items() if cs["type"] in ("categorical", "boolean", "integer")
            and c in df.columns and c not in fks and df[c].nunique() <= max_levels]
    pairs = [(a, b) for i, a in enumerate(fks) for b in fks[i + 1:]] + [(a, c) for a in fks for c in cats]
    out = []
    declared = [sorted(u) for u in ts.get("unique", [])]
    for a, b in pairs:
        if a not in df.columns or b not in df.columns or len(df) < 50:
            continue
        sub = df[[a, b]].dropna()
        if not len(sub):
            continue
        # pairs expected to repeat if the two columns were independent:
        # C(n,2) * sum p_a^2 * sum p_b^2, with the unbiased sum c(c-1)/(n(n-1)) for each
        # sum p^2; P(no repeat) is about exp(-that), the evidence a 0 rate carries
        n = len(sub)
        ca, cb = sub[a].value_counts().to_numpy(float), sub[b].value_counts().to_numpy(float)
        qa = float((ca * (ca - 1)).sum()) / max(n * (n - 1), 1)
        qb = float((cb * (cb - 1)).sum()) / max(n * (n - 1), 1)
        out.append({"columns": [a, b], "duplicate_rate": round(float(sub.duplicated().mean()), 6),
                    "rows": int(n), "declared": sorted([a, b]) in declared,
                    "expected_repeats_independent": round(n * (n - 1) / 2 * qa * qb, 3)})
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def build_bundle(spec, tables=None, synthesizer=None, report=None, errors=None, extra=None, source_rows=None):
    """The diagnostics bundle.  source_rows gives exact row counts of tables that
    `tables` holds only a sample of (pushed-down facts)."""
    b = {"bundle_version": BUNDLE_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "spec_hash": spec_hash(spec), "spec": clean_spec(spec),
         "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
         "errors": errors or []}
    if tables is not None:
        b["profile"] = profile(spec, tables, spec["synthesis"]["privacy"]["suppress_small_counts"])
        for t, n in (source_rows or {}).items():
            if t in b["profile"] and b["profile"][t]["rows"] != n:
                b["profile"][t]["rows_profiled"], b["profile"][t]["rows"] = b["profile"][t]["rows"], int(n)
    if synthesizer is not None:
        try:
            b["model"] = synthesizer.model_summary()
            b["sampling"] = {k: v for k, v in getattr(synthesizer, "sample_log", {}).items() if k != "tree_ppc"}
        except Exception as e:  # noqa: BLE001
            b["errors"].append({"step": "model_summary", "error": repr(e)})
    if report is not None:
        b["evaluation"] = report
    if extra:
        b.update(extra)
    return _jsonable(b)


def write_bundle(bundle, path):
    """Write a bundle as JSON; its spec is masked again here, whoever built it."""
    if isinstance(bundle, dict) and isinstance(bundle.get("spec"), dict):
        bundle = {**bundle, "spec": clean_spec(bundle["spec"])}
    with open(path, "w") as f:
        json.dump(bundle, f, indent=1)
    return path


def read_bundle(path):
    with open(path) as f:
        return json.load(f)
