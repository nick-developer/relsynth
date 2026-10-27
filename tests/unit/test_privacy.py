"""Privacy attacks and the privacy report (relsynth.privacy, Props. 600-606).

Fast checks of each building block on fake data with known answers; the
statistical coverage checks are in tests/science/test_privacy_checks.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))
from conftest import scenario  # noqa: E402

from relsynth.privacy import assess, criteria, mia, report, stats, unit as U  # noqa: E402
from relsynth.privacy.attribute import attribute_inference  # noqa: E402
from relsynth.privacy.config import load as load_config  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402


@pytest.fixture(scope="module")
def star():
    t, s, _ = scenario("S2_star")
    return t, load_spec(s)


def _people(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.integers(18, 90, n), "income": np.round(rng.lognormal(10, 1, n), 2),
                         "region": rng.choice(list("ABCDE"), n), "smoker": rng.choice(["y", "n"], n, p=[.2, .8]),
                         "score": rng.normal(0, 1, n).round(3)})


# ------------------------------------------------------------------ stats
def test_wilson_and_residual_match_anonymeter_formulas():
    r = stats.wilson(500, 18)
    # anonymeter.stats.confidence.success_rate(500, 18, 0.95)
    z = 1.959963984540054
    assert r.value == pytest.approx((18 + z * z / 2) / (500 + z * z))
    assert r.error == pytest.approx(z / (500 + z * z) * np.sqrt(18 * 482 / 500 + z * z / 4))
    k = stats.risk(500, 18, 500, 11)
    v, e = stats.residual(stats.wilson(500, 18), stats.wilson(500, 11))
    assert k.value == pytest.approx(max(v, 0))
    assert k.ci[1] == pytest.approx(min(v + e, 1))


def test_clopper_pearson_edges():
    assert stats.clopper_pearson(0, 10)[0] == 0.0
    assert stats.clopper_pearson(10, 10)[1] == 1.0
    lo, hi = stats.clopper_pearson(5, 10)
    assert lo < 0.5 < hi


def test_tpr_at_fpr_threshold_from_calibration_only():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.5
    s = rng.normal(size=4000) + 1.0 * y
    e = stats.tpr_at_fpr(s[:2000], y[:2000], s[2000:], y[2000:], 0.1)
    assert e["identified"]
    assert e["fpr_ci"][0] <= 0.1 <= e["fpr_ci"][1] + 0.02
    assert e["tpr"] > 0.3
    assert not stats.tpr_at_fpr(s[:50], y[:50], s, y, 0.001)["identified"]


def test_auc_interval_never_zero_width():
    a, se = stats.auc_delong([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    assert a == 1.0 and se > 0


def test_dp_bound():
    assert stats.dp_tpr_bound(0.01, 0.0) == pytest.approx(0.01)
    assert stats.dp_tpr_bound(0.01, 1.0) == pytest.approx(np.e * 0.01)
    assert stats.dp_tpr_bound(0.5, 10.0) == pytest.approx(1 - 0.5 * np.exp(-10))


# ------------------------------------------------------------ privacy unit
def test_unit_paths_and_split_keep_people_whole(star):
    t, s = star
    u = U.resolve(s, "customer")
    assert u.owned == ["sales"]
    tr, ct = U.split(u, t, 0.3, seed=1)
    a, b = U.subset(u, t, tr), U.subset(u, t, ct)
    assert set(a["sales"]["customer_id"]) <= set(tr) and set(b["sales"]["customer_id"]) <= set(ct)
    assert len(a["sales"]) + len(b["sales"]) == len(t["sales"])            # no row lost or shared
    assert len(a["product"]) == len(t["product"])                            # shared tables stay whole


def test_unit_chain_stops_at_other_person_tables():
    spec = {"tables": {"customer": {"role": "entity", "primary_key": "id", "columns": {
        "id": {"type": "id"}, "name": {"type": "pii"}, "rep_id": {"type": "id"}}},
        "employee": {"role": "entity", "primary_key": "id", "columns": {"id": {"type": "id"}, "email": {"type": "pii"}}},
        "invoice": {"role": "fact", "primary_key": "id", "columns": {"id": {"type": "id"}, "cid": {"type": "id"}}}},
        "relationships": [{"child": "customer", "child_key": "rep_id", "parent": "employee", "parent_key": "id"},
                          {"child": "invoice", "child_key": "cid", "parent": "customer", "parent_key": "id"}]}
    assert U.resolve(spec, "employee").owned == []
    assert U.resolve(spec, "customer").owned == ["invoice"]
    assert U.default_unit(spec) == "customer"


def test_profiles_ignore_keys_and_row_order(star):
    t, s = star
    u = U.resolve(s, "customer")
    p1 = U.profiles(s, u, t)
    # relabel every key and shuffle every table: the profiles must not change (Prop. 605)
    m = {k: f"X{i}" for i, k in enumerate(t["customer"]["customer_id"])}
    pm = {k: f"Q{i}" for i, k in enumerate(t["product"]["product_id"])}
    t2 = {"customer": t["customer"].assign(customer_id=t["customer"]["customer_id"].map(m)).sample(frac=1, random_state=3),
          "product": t["product"].assign(product_id=t["product"]["product_id"].map(pm)).sample(frac=1, random_state=4),
          "sales": t["sales"].assign(customer_id=t["sales"]["customer_id"].map(m),
                                     product_id=t["sales"]["product_id"].map(pm)).sample(frac=1, random_state=5)}
    p2 = U.profiles(s, u, t2)
    p2.index = [next(k for k, v in m.items() if v == i) for i in p2.index]
    p2 = p2.loc[p1.index]
    pd.testing.assert_frame_equal(p1.reset_index(drop=True), p2.reset_index(drop=True), check_dtype=False, atol=1e-9)
    assert "sales.n_rows" in p1 and not any(c.endswith("_id") for c in p1.columns)


def test_profiles_follow_the_real_column_kinds(star):
    t, s = star
    u = U.resolve(s, "customer")
    p1 = U.profiles(s, u, t)
    num = next(c for c in t["sales"].columns if c.endswith("_id") is False and t["sales"][c].dtype.kind in "if")
    # a release that writes a numeric column as text still gets the real profile columns
    t2 = dict(t, sales=t["sales"].assign(**{num: t["sales"][num].astype(str)}))
    p2 = U.profiles(s, u, t2, kinds=p1.attrs["kinds"])
    assert list(p2.columns) == list(p1.columns)
    assert f"sales.{num}.mean" in p2 and f"sales.{num}.mode" not in U.profiles(s, u, t)


# ---------------------------------------------------------------- criteria
def test_copy_release_is_caught_and_fresh_release_is_not():
    pop = _people(6000)
    tr, ct, other = pop.iloc[:2500], pop.iloc[2500:3500], pop.iloc[3500:6000]
    same_size = pop.iloc[3500:6000]               # a control as large as the training set: no subsampling
    leak = criteria.record_isolation(tr, tr, same_size, mode="multivariate", n_attacks=200, seed=1)
    safe = criteria.record_isolation(tr, other, ct, mode="multivariate", n_attacks=200, seed=1)
    assert leak.risk.ci[0] > 0.3
    assert safe.risk.ci[0] == 0.0
    # Anonymeter's fitted size correction is not exact under the null (see record_isolation)
    biased = [criteria.record_isolation(tr, other, ct, mode="multivariate", n_attacks=200, seed=s,
                                        size_correction="anonymeter").risk.value for s in range(2)]
    assert min(biased) > 0.05
    lk = criteria.linkage(tr, tr, ct, aux_cols=(["age", "region"], ["income", "score"]), n_attacks=300, n_neighbors=10,
                          seed=1)
    assert lk.risk.value > 0.5
    inf = criteria.inference(tr, tr, ct, attribute="smoker", n_attacks=300, seed=1)
    assert inf.risk.ci[0] > 0.5
    inf0 = criteria.inference(tr, other, ct, attribute="smoker", n_attacks=300, seed=1)
    assert inf0.risk.ci[0] == 0.0


def test_attribute_inference_against_population():
    pop = _people(8000)
    pop["smoker"] = np.where(pop["age"] > 60, "y", pop["smoker"])        # learnable from the population
    tr, ct, ref = pop.iloc[:3000], pop.iloc[3000:4000], pop.iloc[4000:]
    fresh = attribute_inference(tr, ct, ref.sample(3000, random_state=1), ref, attribute="smoker",
                                known=["age", "region", "income", "score"])
    lo, hi = fresh["specific_leakage"]["ci"]
    assert lo < 0.02 and hi > -0.02                 # only population knowledge: no specific leakage
    leak = attribute_inference(tr, ct, tr, ref, attribute="smoker", known=["age", "region", "income", "score"])
    assert leak["advantage_members"]["value"] >= fresh["advantage_members"]["value"] - 0.02


# -------------------------------------------------------------- membership
def test_vulnerability_ranks_outliers_first():
    pop = _people(500)
    pop.loc[pop.index[7], ["age", "income", "region", "smoker", "score"]] = [120, 1e9, "Z", "y", 40.0]
    v = mia.vulnerability(pop)
    assert v.idxmax() == pop.index[7]


def test_shadow_attack_raw_release_vs_population_generator():
    pop = _people(4000, seed=2)
    pop.index = [f"p{i}" for i in range(len(pop))]
    vul = mia.vulnerability(pop.iloc[:1000])
    tids = list(vul.sort_values(ascending=False).index[:3]) + ["p500", "p501", "p502"]
    kinds = pd.Series({t: ("vulnerable" if i < 3 else "random") for i, t in enumerate(tids)})
    pool = [i for i in pop.index[1000:] if i not in tids]
    raw = mia.shadow_attack(pool, pop.loc[tids], kinds, lambda ids, sd: pop.loc[ids].reset_index(drop=True),
                            shadow_size=200, n_train=16, n_test=16, seed=0, ref_profiles=pop.iloc[:1000])
    assert raw.pooled["lira"]["auc"] > 0.9
    ref = pop.iloc[:1000]
    noinfo = mia.shadow_attack(pool, pop.loc[tids], kinds,
                               lambda ids, sd: ref.sample(len(ids), random_state=sd).reset_index(drop=True),
                               shadow_size=200, n_train=16, n_test=16, seed=0, ref_profiles=ref)
    lo, hi = noinfo.pooled["lira"]["auc_ci"]
    assert lo < 0.5 < hi or abs(noinfo.pooled["lira"]["auc"] - 0.5) < 0.12


def test_shadow_cache_reuses_fits(tmp_path):
    pop = _people(1500, seed=3)
    pop.index = [f"p{i}" for i in range(len(pop))]
    calls = []

    def gen(ids, sd):
        calls.append(sd)
        return pop.loc[ids].reset_index(drop=True)
    tids = ["p1", "p2"]
    kinds = pd.Series({"p1": "random", "p2": "random"})
    cache = mia.ShadowCache(tmp_path, "g")
    args = dict(shadow_size=100, n_train=4, n_test=4, seed=0, ref_profiles=pop.iloc[500:], cache=cache)
    a = mia.shadow_attack(list(pop.index[3:500]), pop.loc[tids], kinds, gen, **args)
    n = len(calls)
    b = mia.shadow_attack(list(pop.index[3:500]), pop.loc[tids], kinds, gen, **args)
    assert n == 8 and len(calls) == 8 and b.fits_cached == 8
    assert a.targets["lira_auc"].tolist() == b.targets["lira_auc"].tolist()


def test_domias_detects_overfitting():
    pop = _people(6000, seed=4)
    tr, ct, ref = pop.iloc[:2000], pop.iloc[2000:3000], pop.iloc[3000:]
    leak = mia.domias(tr, ct, pd.concat([tr] * 2), ref)
    fresh = mia.domias(tr, ct, ref.sample(2000, random_state=0), ref)
    assert leak["auc"] > fresh["auc"] + 0.1


# ------------------------------------------------------------- end to end
def test_assess_report_and_contexts(star, tmp_path):
    t, s = star
    cfg = {"unit": "customer", "sensitive": ["income"], "quasi_identifiers": ["age", "region"],
           "attacks": {"record_isolation": {"n_attacks": 100}, "linkage": {"n_attacks": 100},
                       "inference": {"n_attacks": 200},
                       "membership": {"shadow": {"n_train": 4, "n_test": 4, "shadow_size": 60,
                                                 "targets_vulnerable": 2, "targets_random": 2}}}}
    # a generator that returns its training data: the worst release possible
    rep = assess(s, cfg, tables=t, generator=lambda db, seed: {k: v.copy() for k, v in db.items()})
    assert rep["people"]["training"] + rep["people"]["control"] + rep["people"]["reference"] == 1500
    assert set(rep["contexts"]) == {"internal", "partner", "public"}
    assert rep["contexts"]["public"]["verdict"] == "blocked"          # no DP: external releases are blocked
    assert rep["contexts"]["internal"]["verdict"] == "fail"
    assert rep["attacks"]["record_isolation"][1]["value"] > 0.3
    assert len(rep["vulnerable_records"]) == 4
    assert all(r["person"].startswith("P-") for r in rep["vulnerable_records"])
    # simultaneous intervals over the 4 targets x 2 attacks
    assert all(r["ci_level"] == pytest.approx(1 - 0.05 / 8) for r in rep["vulnerable_records"])
    paths = report.write_report(rep, tmp_path)
    md = Path(paths["md"]).read_text()
    assert md.count("# Privacy report:") == 3 and "BLOCKED" in md
    assert "C000" not in md and "C000" not in Path(paths["html"]).read_text()     # no real keys in the report
    j = json.loads(Path(paths["json"]).read_text())
    _validate(j)
    # labels are keyed by a secret salt, never by the split seed, and the salt is never written
    assert rep["pseudonyms"]["salt_id"] and rep["_salt"] not in Path(paths["json"]).read_text()
    assert rep["_salt"] not in md and rep["_salt"] not in Path(paths["html"]).read_text()
    rep2 = assess(s, cfg, tables=t, generator=lambda db, seed: {k: v.copy() for k, v in db.items()},
                  salt=bytes.fromhex(rep["_salt"]))
    assert [r["person"] for r in rep2["vulnerable_records"]] == [r["person"] for r in rep["vulnerable_records"]]
    ref = report.reference(paths["json"])
    assert len(ref["sha256"]) == 64 and ref["verdicts"]["public"] == "blocked"


def test_assess_with_dp_is_not_blocked(star):
    t, s = star
    cfg = {"unit": "customer", "release": {"dp": {"epsilon": 1.0, "delta": 1e-6, "cumulative_epsilon": 2.0}},
           "attacks": {"record_isolation": {"modes": ["univariate"], "n_attacks": 50}, "linkage": {"n_attacks": 50},
                       "inference": {"n_attacks": 50}, "attribute_inference": False,
                       "membership": {"domias": False, "shadow": {"enabled": False}}}}
    rng = np.random.default_rng(0)
    rep = assess(s, cfg, tables=t, generator=lambda db, seed: {
        k: (v.sample(frac=1, random_state=1).reset_index(drop=True) if k != "customer" else
            v.assign(age=rng.permutation(v["age"].to_numpy()))) for k, v in db.items()})
    assert rep["contexts"]["public"]["verdict"] != "blocked"
    assert rep["contexts"]["public"]["dp_bound_tpr_at_fpr_0.01"] == pytest.approx(np.e * 0.01 + 1e-6)
    # the domain's cumulative epsilon gives the looser bound for everything released so far
    assert rep["contexts"]["public"]["dp_bound_tpr_at_fpr_0.01_cumulative"] == pytest.approx(np.e ** 2 * 0.01 + 1e-6)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown privacy settings"):
        load_config({"epsilon": 1})
    c = load_config({"privacy": {"unit": "customer"}})
    assert c["unit"] == "customer" and c["contexts"]["public"]["external"]


def _validate(doc):
    schema = json.loads((Path(report.__file__).with_name("report_schema.json")).read_text())
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(doc, schema)


def test_cli_privacy_writes_report_and_gates(star_csv, tmp_path):
    import yaml

    from relsynth.cli import main
    spec, _ = star_csv
    sp, cfg = tmp_path / "spec.yaml", tmp_path / "privacy.yaml"
    sp.write_text(yaml.safe_dump(spec))
    cfg.write_text(yaml.safe_dump({"unit": "customer", "sensitive": ["income"], "attacks": {
        "record_isolation": {"n_attacks": 50}, "linkage": {"n_attacks": 50}, "inference": {"n_attacks": 50}}}))
    out = tmp_path / "rep"
    code = main(["privacy", str(sp), "--config", str(cfg), "--no-shadow", "--out", str(out), "--json"])
    assert code == 0
    assert (out / "privacy_report.html").exists() and (out / "privacy_report.json").exists()
    salt = (out / "privacy_salt.key").read_text().strip()
    assert len(salt) == 64 and salt not in (out / "privacy_report.json").read_text()
    if os.name == "posix":          # Windows has no POSIX permission bits
        assert (out / "privacy_salt.key").stat().st_mode & 0o077 == 0
    # external contexts are blocked without DP, so gating on 'public' fails the job
    assert main(["privacy", str(sp), "--config", str(cfg), "--no-shadow", "--out", str(out),
                 "--context", "public"]) == 1
