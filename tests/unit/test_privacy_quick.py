# SPDX-License-Identifier: Apache-2.0
"""The quick privacy check of `relsynth run` (relsynth.privacy.quick, G5,
Props. 1740-1741) and the per-generator privacy rows of the benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakedata"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conftest import scenario  # noqa: E402

from relsynth.privacy import quick as Q, unit as U  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402


@pytest.fixture(scope="module")
def shop():
    t, s, _ = scenario("S9_small_shop")
    return t, load_spec(s)


def _copy(tables, seed):
    """A generator that releases its training rows: the worst case."""
    return {k: v.copy() for k, v in tables.items()}


def test_quick_check_passes_a_fitted_release_and_reports_counts(shop):
    t, s = shop
    r = Q.quick_check(s, t, seed=1, unit="customer")
    assert r["status"] in ("pass", "review")
    assert r["attack"] in ("domias", "closest_distance") and r["unit"] == "customer" and r["context"] == "internal"
    assert set(r["attacks"]) == {"domias", "closest_distance"}
    p = r["people"]
    assert p["total"] == 200 and p["members"] + p["non_members"] + p["reference"] == 200
    assert p["non_members"] == p["reference"] >= Q.MIN_GROUP
    lo, hi = r["auc_ci"]
    assert lo <= r["auc"] <= hi and hi < 0.8
    assert "warning" in r                                      # 200 people < small_population
    # the proxy keeps the unit, what it owns and their parents; playlists are not reached
    assert set(r["proxy_tables"]) == {"customer", "invoice", "invoice_line", "track", "album", "genre", "staff"}
    txt = Q.line(r)
    assert txt.startswith("privacy check: ") and "DOMIAS AUC" in txt and "distance AUC" in txt
    assert "little power" in txt


def test_quick_check_catches_a_copying_generator(shop):
    t, s = shop
    r = Q.quick_check(s, t, seed=1, unit="customer", generator=_copy)
    assert r["status"] == "fail" and r["auc_ci"][0] > Q.LIMITS["mia_auc"]
    assert "relsynth privacy" in Q.line(r)


def test_quick_check_is_deterministic_and_leaves_tables_alone(shop):
    t, s = shop
    before = {k: v.copy() for k, v in t.items()}
    a = Q.quick_check(s, t, seed=3, unit="customer", generator=_copy)
    b = Q.quick_check(s, t, seed=3, unit="customer", generator=_copy)
    assert a["auc"] == b["auc"] and a["people"] == b["people"]
    for k in t:
        pd.testing.assert_frame_equal(t[k], before[k])


def test_quick_check_skips_without_unit_too_few_people_or_pushdown(shop):
    t, s = shop
    r = Q.quick_check(s, t, seed=1)                            # no table looks personal in S9
    assert r["status"] == "skipped" and "privacy unit" in r["reason"]
    few = dict(t)
    few["customer"] = t["customer"].iloc[:30]
    r = Q.quick_check(s, few, seed=1, unit="customer")
    assert r["status"] == "skipped" and "30 people" in r["reason"]

    class St:
        n_rows = 10 ** 7
    r = Q.quick_check(s, t, seed=1, unit="customer", stats={"invoice_line": St()})
    assert r["status"] == "skipped" and "invoice_line" in r["reason"]
    assert Q.line(r).startswith("privacy check: skipped")
    assert Q.line(None) == "privacy check: off"


def test_quick_check_reports_generator_errors_without_raising(shop):
    t, s = shop

    def broken(tables, seed):
        raise RuntimeError("cannot reach postgresql://admin:s3cret@db.internal/prod")
    r = Q.quick_check(s, t, seed=1, unit="customer", generator=broken)
    assert r["status"] == "error" and "RuntimeError" in r["reason"]
    # a generator's message can hold a credential: only the exception type is kept
    assert "s3cret" not in str(r) and "postgresql://" not in str(r)


def test_an_attack_that_cannot_run_is_named(shop, monkeypatch):
    t, s = shop
    from relsynth.privacy import mia
    monkeypatch.setattr(mia, "domias", lambda *a, **k: {"identified": False, "reason": "constant profiles"})
    r = Q.quick_check(s, t, seed=1, unit="customer")
    assert r["status"] in ("pass", "review") and r["attack"] == "closest_distance"
    assert r["incomplete"] == {"domias": "constant profiles"}
    assert "DOMIAS could not run (constant profiles)" in Q.line(r)


def test_pipeline_error_keeps_only_the_exception_type(monkeypatch, caplog):
    from relsynth.pipeline import execute
    t, s = _person_shop()

    def boom(*a, **k):
        raise ValueError("password=hunter2")
    monkeypatch.setattr(Q, "quick_check", boom)
    with caplog.at_level("DEBUG"):
        r = execute(s, tables=t, write=False, privacy_check=True)
    assert r.bundle["privacy_check"] == {"status": "error", "reason": "the check could not run (ValueError)"}
    assert "hunter2" not in caplog.text and not r.errors


def test_proxy_spec_keeps_lookup_tables_and_drops_unrelated(shop):
    _, s = shop
    u = U.resolve(s, "customer")
    ps = Q.proxy_spec(s, u)
    assert "playlist" not in ps["tables"] and "playlist_track" not in ps["tables"]
    assert all(r["child"] in ps["tables"] and r["parent"] in ps["tables"] for r in ps["relationships"])
    assert all(r.get("table") in ps["tables"] for r in ps["rules"])
    assert s["tables"].keys() >= ps["tables"].keys()
    # a table mentioned in a kept table's settings (a lookup) is kept
    s2 = {**s, "tables": {**s["tables"], "customer": {**s["tables"]["customer"],
                                                         "x_lookup": {"table": "playlist"}}}}
    assert "playlist" in Q.proxy_spec(s2, U.resolve(s2, "customer"))["tables"]


def _person_shop():
    """S9 with an e-mail column on the customer, so that the default unit is found."""
    t, s, _ = scenario("S9_small_shop")
    t["customer"]["email"] = [f"c{i}@example.org" for i in range(len(t["customer"]))]
    s["tables"]["customer"]["columns"]["email"] = {"type": "pii", "pii": {"generator": "email"}}
    return t, s


def test_run_adds_the_check_and_keeps_the_release_identical():
    from relsynth.pipeline import execute
    t, s = _person_shop()
    on = execute(s, tables=t, write=False, privacy_check=True)
    off = execute(s, tables=t, write=False, privacy_check=False)
    pc = on.bundle["privacy_check"]
    assert pc["status"] in ("pass", "review") and pc["unit"] == "customer"
    assert "privacy_check" not in off.bundle
    for k in on.data:
        pd.testing.assert_frame_equal(on.data[k], off.data[k])


def test_setting_turns_the_check_off():
    from relsynth.pipeline import execute
    t, s = _person_shop()
    s.setdefault("synthesis", {}).setdefault("evaluation", {})["privacy_check"] = False
    r = execute(s, tables=t, write=False, privacy_check=True)
    assert "privacy_check" not in r.bundle


def test_settings_registry_knows_the_keys():
    from relsynth.settings import Settings
    st = Settings.defaults()
    assert st.get("evaluation.privacy_check") is True
    assert st.get("evaluation.privacy_check_people") == 3000


# ------------------------------------------------------------ benchmark rows
def test_bench_privacy_rows_carry_the_generator():
    from bench.assess import privacy_checks, with_generator
    rep = {"contexts": {"internal": {"verdict": "pass", "checks": [
        {"criterion": "membership", "attack": "domias_auc", "value": 0.51, "ci": [0.48, 0.54], "limit": 0.65,
         "status": "pass"}]}, "public": {"external": True, "verdict": "blocked"}}, "release": {"dp": None},
        "guards": {}}
    own = {c["id"] for c in privacy_checks(rep)}
    cart = {c["id"] for c in privacy_checks(rep, "cart")}
    assert "privacy:membership|domias_auc" in own and "goal:privacy_internal" in own
    assert "privacy[cart]:membership|domias_auc" in cart and "goal:privacy_internal[cart]" in cart
    assert "goal:privacy_public_needs_dp[cart]" in cart and "metric:privacy_membership_domias_auc_cart" in cart
    assert not own & cart
    s = {"synthesis": {"generator": "copula"}, "tables": {"a": {"generator": "cart"}, "b": {}}}
    w = with_generator(s, "independent")
    assert w["synthesis"]["generator"] == "independent" and "generator" not in w["tables"]["a"]
    assert s["tables"]["a"]["generator"] == "cart"


def test_bench_lists_every_builtin_generator_for_acs_and_berka():
    from bench.datasets import ALL_GENERATORS, DATASETS
    from relsynth.generators.registry import default_registry
    assert set(ALL_GENERATORS) <= set(default_registry().names())
    for name in ("acs", "berka"):
        assert tuple(DATASETS[name].privacy["generators"]) == ALL_GENERATORS


def test_dp_specs_never_reach_the_quick_check():
    """DP runs go through relsynth.dp (cmd_run_dp); the ordinary pipeline, which
    holds the quick check, refuses a spec in DP mode."""
    from relsynth.errors import PrivacyError
    from relsynth.pipeline import execute
    t, s = _person_shop()
    s.setdefault("synthesis", {}).setdefault("privacy", {})["dp"] = {"epsilon": 1.0, "unit": "customer"}
    with pytest.raises(PrivacyError):
        execute(s, tables=t, write=False, privacy_check=True)
