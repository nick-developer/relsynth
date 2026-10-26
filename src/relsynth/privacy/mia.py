"""Membership inference: can an attacker tell whether a person was in the
training data of a release?

Three attacks, all on one profile row per person:

* Shadow modelling (Stadler, Oprisanu and Troncoso, USENIX Security 2022;
  Houssiau et al., TAPAS 2022): the attacker knows the generator and has data
  from the population.  It fits the generator on many shadow databases, each
  of which holds every target with probability 1/2 independently (Prop. 602),
  and learns to read a target's membership from target-specific features of
  the synthetic data: counting queries on the target's values (TAPAS) and
  distances to the target.  Two read-outs share the same shadow fits: a
  logistic classifier on the features, and a parametric likelihood-ratio test
  on one statistic in the style of LiRA (Carlini et al., IEEE S&P 2022).
* DOMIAS (van Breugel et al., AISTATS 2023): the density of the synthetic data
  over the density of population data at the person, p_syn(x) / p_ref(x), on
  the actual release; no shadow fits.
* Vulnerable-record selection (Meeus, Guépin, Cretu and de Montjoye, ESORICS
  2023): targets are the people with the largest mean cosine distance to
  their 5 nearest neighbours, plus random people for comparison.

Results are TPR at low FPR with exact intervals and AUC with DeLong intervals,
never average accuracy alone (Carlini et al. 2022, sec. 3).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import gaussian_kde, norm

from .encode import Encoder, column_kinds, kneighbors
from .stats import auc_delong, auc_interval, tpr_at_fpr

FPRS = (0.1, 0.01, 0.001)


# -------------------------------------------------------- vector encoding
class Vectoriser:
    """Profiles -> real vectors: numbers min-max scaled on the reference,
    categories one-hot (the 30 most frequent levels).  Used for cosine
    distances (vulnerability) and densities (DOMIAS)."""

    def __init__(self, ref: pd.DataFrame, max_levels: int = 30):
        self.cols = list(ref.columns)
        self.num, self.cat = [], {}
        for c in self.cols:
            s = ref[c]
            if s.dtype.kind in "biuf" and s.nunique() > 2:
                v = s.astype(float)
                self.num.append((c, float(np.nanmin(v)), float(np.nanmax(v) - np.nanmin(v)) or 1.0))
            else:
                self.cat[c] = s.astype(str).value_counts().index[:max_levels].tolist()

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for c, lo, sp in self.num:
            x = (pd.to_numeric(df[c], errors="coerce").to_numpy(float) - lo) / sp
            parts.append(np.nan_to_num(x, nan=-0.5)[:, None])
        for c, lv in self.cat.items():
            v = df[c].astype(str).to_numpy()
            parts.append(np.column_stack([(v == x).astype(float) for x in lv]) if lv else np.zeros((len(df), 0)))
        return np.column_stack(parts) if parts else np.zeros((len(df), 0))


def vulnerability(profiles: pd.DataFrame, k: int = 5, vec: Vectoriser | None = None) -> pd.Series:
    """Meeus et al. (2023) score: mean cosine distance of each person to their
    k nearest other people.  Large = isolated = vulnerable."""
    vec = vec or Vectoriser(profiles)
    X = vec(profiles)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    out = np.empty(len(X))
    for i in range(0, len(X), 2048):
        S = X[i:i + 2048] @ X.T
        S[np.arange(len(S)), i + np.arange(len(S))] = -np.inf
        top = -np.partition(-S, k - 1, axis=1)[:, :k]
        out[i:i + 2048] = 1 - top.mean(1)
    return pd.Series(out, index=profiles.index, name="vulnerability")


# ------------------------------------------------------------ features
class TargetFeatures:
    """Target-specific features of a synthetic profile table (TAPAS-style
    counting queries plus distances).  Fixed per target before any shadow is
    fitted, so features from different shadows are comparable."""

    def __init__(self, target: pd.Series, ref: pd.DataFrame, n_queries: int = 40, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.target = target
        self.cols = list(ref.columns)
        self.bins = {}
        for c in self.cols:
            s = ref[c]
            if s.dtype.kind in "biuf" and s.nunique() > 10:
                q = np.unique(np.nanquantile(s.astype(float), np.linspace(0, 1, 11)))
                self.bins[c] = q
        sizes = rng.integers(1, min(4, len(self.cols)) + 1, n_queries)
        self.queries = [sorted(rng.choice(self.cols, size=k, replace=False).tolist()) for k in sizes]
        self.queries.append(self.cols)                      # the full record
        self.enc = Encoder([ref], self.cols, column_kinds([ref], self.cols))
        self.frame = pd.DataFrame([target.to_numpy()], columns=target.index, index=[target.name])[self.cols]

    def _match(self, syn: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for c in self.cols:
            v = self.target[c]
            s = syn[c]
            if c in self.bins:
                b = self.bins[c]
                tv = np.searchsorted(b, float(v), "right") if pd.notna(v) else -1
                sv = np.where(s.isna(), -1, np.searchsorted(b, pd.to_numeric(s, errors="coerce").fillna(0), "right"))
                out[c] = sv == tv
            elif pd.isna(v):
                out[c] = s.isna().to_numpy()
            else:
                out[c] = (s.astype(str) == str(v)).to_numpy()
        return out

    def __call__(self, syn: pd.DataFrame) -> np.ndarray:
        m = self._match(syn)
        n = max(len(syn), 1)
        f = [np.logical_and.reduce([m[c] for c in q]).sum() / n for q in self.queries]
        _, d = kneighbors(self.enc, self.frame, syn, k=min(5, len(syn)))
        d = d[0]
        f += [d[0], d.mean(), float(np.log(n))]
        return np.asarray(f, float)

    @staticmethod
    def lira_stat(feat: np.ndarray) -> float:
        """The LiRA statistic: distance from the target to its closest synthetic
        profile (feature -3)."""
        return -float(feat[-3])


def _fit_logistic(X, y, lam=1.0):
    from scipy.optimize import minimize
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])

    def f(w):
        z = Z @ w
        return (np.logaddexp(0, z) - y * z).sum() + lam * (w[:-1] ** 2).sum(), \
            Z.T @ (expit(z) - y) + 2 * lam * np.r_[w[:-1], 0]
    w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return lambda Xn: np.column_stack([(Xn - mu) / sd, np.ones(len(Xn))]) @ w


def _lira(s_tr, y_tr):
    s_in, s_out = s_tr[y_tr == 1], s_tr[y_tr == 0]
    mi, si = s_in.mean(), s_in.std() + 1e-9
    mo, so = s_out.mean(), s_out.std() + 1e-9
    return lambda s: norm.logpdf(s, mi, si) - norm.logpdf(s, mo, so)


# --------------------------------------------------------------- shadows
_JOB = None


def _one_fit(arg):
    ids, sd = arg
    return _JOB(ids, sd)


def _run_fits(fn, plans, index, n_jobs, progress, total, done):
    """Fit the shadows, in forked worker processes when n_jobs > 1 (the fits
    are independent; each has its own seed, so the result does not depend on
    n_jobs)."""
    import multiprocessing as mp
    global _JOB
    if n_jobs == 1 or len(plans) < 2 or "fork" not in mp.get_all_start_methods():
        for k, (i, p) in enumerate(zip(index, plans)):
            yield i, fn(*p)
            if progress:
                progress(done + k + 1, total)
        return
    _JOB = fn
    try:
        with mp.get_context("fork").Pool(min(n_jobs, len(plans))) as pool:
            for k, (i, syn) in enumerate(zip(index, pool.imap(_one_fit, plans))):
                yield i, syn
                if progress:
                    progress(done + k + 1, total)
    finally:
        _JOB = None
@dataclass
class ShadowResult:
    targets: pd.DataFrame          # per target: kind, vulnerability, auc (+ci) for each read-out
    pooled: dict                   # read-out -> {auc, tpr_at_fpr: [...]}
    n_shadows: tuple
    shadow_size: int
    seconds: float
    fits_cached: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        t = self.targets.reset_index().rename(columns={"index": "target"})
        return {"attack": "shadow_modelling", "pooled": self.pooled, "n_shadows_train": self.n_shadows[0],
                "n_shadows_test": self.n_shadows[1], "shadow_size": self.shadow_size, "seconds": self.seconds,
                "fits_cached": self.fits_cached, "notes": self.notes,
                "targets": json.loads(t.to_json(orient="records"))}


class ShadowCache:
    """Synthetic profiles of shadow fits, on disk, keyed by the generator's
    identity, the people in the shadow and the seed (JSON, no pickle)."""

    def __init__(self, folder: str | Path | None, ident: str):
        self.folder = Path(folder) if folder else None
        self.ident = ident
        if self.folder:
            self.folder.mkdir(parents=True, exist_ok=True)

    def key(self, ids, seed) -> str:
        h = hashlib.sha256(self.ident.encode())
        h.update(json.dumps(sorted(map(str, ids))).encode())
        h.update(str(seed).encode())
        return h.hexdigest()[:32]

    def get(self, key):
        if not self.folder:
            return None
        p = self.folder / f"{key}.json"
        return pd.read_json(p, orient="split", dtype=False) if p.exists() else None

    def put(self, key, df):
        if self.folder:
            df.to_json(self.folder / f"{key}.json", orient="split", date_format="iso")


def shadow_attack(population_ids, targets: pd.DataFrame, target_kind: pd.Series, synth_profiles, *,
                  shadow_size: int, n_train: int = 32, n_test: int = 32, seed: int = 0,
                  ref_profiles: pd.DataFrame, cache: ShadowCache | None = None,
                  vuln: pd.Series | None = None, level: float = 0.95, progress=None,
                  n_jobs: int = 1) -> ShadowResult:
    """Generator-level membership inference with shared shadow fits.

    population_ids: people the attacker can draw shadow data from (no target).
    targets: target profiles (index: person id).  synth_profiles(ids, seed) fits
    the generator on the people `ids` and returns the synthetic profiles."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pop = np.asarray(population_ids)
    tids = list(targets.index)
    n_other = max(shadow_size - len(tids) // 2, 1)
    feats = {t: TargetFeatures(targets.loc[t], ref_profiles, rng=np.random.default_rng(seed + 7 + i))
             for i, t in enumerate(tids)}
    X = {t: [] for t in tids}
    Y = np.zeros((n_train + n_test, len(tids)), int)
    plans = []
    for s_ in range(n_train + n_test):
        member = rng.random(len(tids)) < 0.5
        Y[s_] = member
        others = rng.choice(pop, size=min(n_other, len(pop)), replace=False)
        plans.append((list(others) + [t for t, m in zip(tids, member) if m], seed * 100003 + s_))
    keys = [cache.key(ids, sd) for ids, sd in plans] if cache else [None] * len(plans)
    syns = [cache.get(k) if cache else None for k in keys]
    todo = [i for i, x in enumerate(syns) if x is None]
    cached = len(plans) - len(todo)
    for i, syn in _run_fits(synth_profiles, [plans[i] for i in todo], todo, n_jobs, progress, len(plans), cached):
        syns[i] = syn
        if cache:
            cache.put(keys[i], syn)
    for syn in syns:
        syn = syn[[c for c in ref_profiles.columns if c in syn.columns]]
        for t in tids:
            X[t].append(feats[t](syn))
    rows, pooled_scores = [], {"classifier": ([], [], [], []), "lira": ([], [], [], [])}
    tr, te = slice(0, n_train), slice(n_train, None)
    for j, t in enumerate(tids):
        F = np.vstack(X[t])
        y = Y[:, j]
        row = {"target": t, "kind": target_kind.get(t, "random"),
               "vulnerability": float(vuln.get(t, np.nan)) if vuln is not None else np.nan}
        if y[tr].min() == y[tr].max():
            row["note"] = "target never (or always) in the training shadows"
            rows.append(row)
            continue
        clf = _fit_logistic(F[tr], y[tr].astype(float))
        stat = np.array([TargetFeatures.lira_stat(f) for f in F])
        lr = _lira(stat[tr], y[tr])
        for name, sc_te in (("classifier", clf(F[te])), ("lira", lr(stat[te]))):
            a, se = auc_delong(sc_te, y[te])
            row[f"{name}_auc"] = a
            row[f"{name}_auc_ci"] = auc_interval(a, se, level)
            row[f"{name}_auc_se"] = se
            # pooled scores: calibration half of the test shadows sets thresholds, the other half counts
            P = pooled_scores[name]
            half = len(sc_te) // 2
            for part, vals in zip(P, (sc_te[:half], y[te][:half], sc_te[half:], y[te][half:])):
                part.extend(vals)
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("target")
    pooled = {}
    for name, (sc, yc, st, yt) in pooled_scores.items():
        if not st:
            continue
        a, se = auc_delong(np.r_[sc, st], np.r_[yc, yt])
        pooled[name] = {"auc": a, "auc_ci": auc_interval(a, se, level),
                        "tpr_at_fpr": [tpr_at_fpr(sc, yc, st, yt, f, level) for f in FPRS]}
        for kind in ("vulnerable", "random"):
            col = f"{name}_auc"
            if col in tab and (tab["kind"] == kind).any():
                v = tab.loc[tab["kind"] == kind, col].dropna()
                pooled[name][f"mean_auc_{kind}"] = float(v.mean()) if len(v) else np.nan
    return ShadowResult(tab, pooled, (n_train, n_test), int(n_other + len(tids) / 2), time.time() - t0, cached)


# ---------------------------------------------------------------- DOMIAS
def domias(members: pd.DataFrame, non_members: pd.DataFrame, syn: pd.DataFrame, ref: pd.DataFrame, *,
           dims: int = 8, level: float = 0.95, seed: int = 0) -> dict:
    """DOMIAS score p_syn(x) / p_ref(x) with Gaussian kernel densities in a
    principal-component embedding of the profiles (van Breugel et al. 2023 use
    a normalising flow or a KDE; the KDE is their lighter variant).  Members are
    training people, non-members are control people; half of each calibrates
    the threshold for TPR at FPR, the other half counts."""
    vec = Vectoriser(ref)
    R = vec(ref)
    mu = R.mean(0)
    U, S, Vt = np.linalg.svd(R - mu, full_matrices=False)
    k = int(min(dims, (S > 1e-9).sum()))
    P = Vt[:k].T
    emb = lambda df: (vec(df) - mu) @ P  # noqa: E731
    try:
        k_syn, k_ref = gaussian_kde(emb(syn).T), gaussian_kde(emb(ref).T)
    except (np.linalg.LinAlgError, ValueError) as e:
        return {"attack": "domias", "identified": False, "reason": f"density not estimable: {e}"}
    xm, xn = emb(members).T, emb(non_members).T
    s_m = k_syn.logpdf(xm) - k_ref.logpdf(xm)
    s_n = k_syn.logpdf(xn) - k_ref.logpdf(xn)
    rng = np.random.default_rng(seed)
    im, inn = rng.permutation(len(s_m)), rng.permutation(len(s_n))
    hm, hn = len(im) // 2, len(inn) // 2
    sc = np.r_[s_m[im[:hm]], s_n[inn[:hn]]]
    yc = np.r_[np.ones(hm), np.zeros(hn)]
    st = np.r_[s_m[im[hm:]], s_n[inn[hn:]]]
    yt = np.r_[np.ones(len(im) - hm), np.zeros(len(inn) - hn)]
    a, se = auc_delong(np.r_[s_m, s_n], np.r_[np.ones(len(s_m)), np.zeros(len(s_n))])
    score = pd.Series(np.r_[s_m, s_n], index=list(members.index) + list(non_members.index))
    return {"attack": "domias", "identified": True, "auc": a, "auc_ci": auc_interval(a, se, level),
            "tpr_at_fpr": [tpr_at_fpr(sc, yc, st, yt, f, level) for f in FPRS],
            "n_members": int(len(s_m)), "n_non_members": int(len(s_n)), "dims": k,
            "_member_scores": score.iloc[:len(s_m)]}
