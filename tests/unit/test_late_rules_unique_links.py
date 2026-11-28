# SPDX-License-Identifier: Apache-2.0
"""Late rules enforced after the aggregates (Prop. 1881) and unique link keys placed on free parents (Prop. 1880).

Classes, each tested on schemas other than the Berka bank data that revealed them:

* late rules: a check rule that reads an aggregate column through a link (a review written on or after
  the customer's first order, when the first order date is min(order date) of the orders; a support
  ticket opened on or after the account's first login) can only be evaluated after every table is drawn.
  Before, such rules were only counted.
* unique link keys: a unique key of one link and a column fixed before the links (one holder per
  insurance policy: [policy_id, role]; one desk per employee: [employee_id]) where nearly every parent
  holds the key, so the link model's proposals and a uniform fallback rarely reach the free parents.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import relsynth as rs

IDC = {"type": "id"}


def _write(db, frames):
    with sqlite3.connect(db) as con:
        for name, df in frames.items():
            df.to_sql(name, con, index=False)


# ---------------------------------------------------------------- late rules

def _shop(tmp_path, n_cust=400, seed=3):
    """Customers, their orders (a fact) and reviews; first_order = min(orders.order_date)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    seg = rng.choice(["retail", "trade"], n_cust, p=[0.7, 0.3])
    first = start + pd.to_timedelta(rng.integers(0, 900, n_cust), unit="D")
    rows = []
    for i in range(n_cust):
        k = 1 + rng.poisson(4 if seg[i] == "trade" else 2)
        gaps = np.r_[0, np.sort(rng.integers(1, 400, k - 1))]
        for g in gaps:
            rows.append((i + 1, first[i] + pd.Timedelta(days=int(g)), float(np.round(rng.gamma(2, 40), 2))))
    orders = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))
    cust = pd.DataFrame({"customer_id": np.arange(1, n_cust + 1), "segment": seg, "first_order": first})
    who = rng.choice(n_cust, int(0.6 * n_cust), replace=False)
    rev = pd.DataFrame({"review_id": np.arange(1, len(who) + 1), "customer_id": who + 1,
                        "review_date": first[who] + pd.to_timedelta(rng.integers(0, 500, len(who)), unit="D"),
                        "stars": rng.integers(1, 6, len(who))})
    for df, c in ((orders, "order_date"), (cust, "first_order"), (rev, "review_date")):
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    db = tmp_path / "shop.db"
    _write(db, {"customer": cust, "orders": orders, "review": rev})
    spec = {"version": 1, "name": "shop", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "customer": {"role": "entity", "primary_key": "customer_id",
                             "columns": {"customer_id": IDC, "segment": {"type": "categorical"},
                                         "first_order": {"type": "date"}}},
                "orders": {"role": "fact", "primary_key": "order_id",
                           "fact": {"anchor": "customer", "time_column": "order_date", "time_bucket": "month"},
                           "columns": {"order_id": IDC, "customer_id": IDC, "order_date": {"type": "date"},
                                       "amount": {"type": "float"}}},
                "review": {"role": "entity", "primary_key": "review_id",
                           "columns": {"review_id": IDC, "customer_id": IDC, "review_date": {"type": "date"},
                                       "stars": {"type": "integer"}}}},
            "relationships": [
                {"child": "orders", "child_key": "customer_id", "parent": "customer", "parent_key": "customer_id"},
                {"child": "review", "child_key": "customer_id", "parent": "customer", "parent_key": "customer_id"}],
            "rules": [
                {"table": "customer", "type": "aggregate", "column": "first_order", "agg": "min", "from": "orders",
                 "of": "order_date"},
                {"table": "orders", "type": "check", "expr": "days(customer_id.first_order) <= days(order_date)"},
                {"table": "review", "type": "check", "expr": "days(customer_id.first_order) <= days(review_date)"},
                {"table": "review", "type": "check", "expr": "stars >= 1"}],
            "synthesis": {"seed": 5}}
    return spec


def _late_viol(out):
    first = pd.to_datetime(out["customer"].set_index("customer_id")["first_order"])
    rd = pd.to_datetime(out["review"]["review_date"])
    return int((rd < out["review"]["customer_id"].map(first)).sum())


def test_late_rule_enforced(tmp_path):
    spec = _shop(tmp_path)
    rep = rs.fit(spec, settings={"rules": {"late": "report"}})
    before = _late_viol(rep.sample().to_dict())
    assert before > 0, "the data must break the late rule without enforcement for the test to mean anything"
    m = rs.fit(spec)
    out = m.sample().to_dict()
    assert _late_viol(out) == 0
    log = m.synthesizer.sample_log
    assert log["late_rules"]["review"]["still_violating"] == 0
    assert log["late_rules"]["review"]["redrawn"] > 0
    assert log["rule_violations"]["review"]["days(customer_id.first_order) <= days(review_date)"] == 0
    assert log["rule_violations"]["review"]["stars >= 1"] == 0          # other rules of the row still hold
    # the redraw keeps the rows' count and the link; only violating rows change
    assert len(out["review"]) == len(rep.sample(dataset=0).to_dict()["review"])
    # the aggregate is untouched by the redraw of a table it does not read
    first = out["orders"].groupby("customer_id")["order_date"].min()
    cust = out["customer"].set_index("customer_id")["first_order"]
    assert (pd.to_datetime(cust.reindex(first.index)) == pd.to_datetime(first)).all()


def test_late_rule_report_mode_counts_only(tmp_path):
    spec = _shop(tmp_path)
    m = rs.fit(spec, settings={"rules": {"late": "report"}})
    out = m.sample().to_dict()
    n = _late_viol(out)
    assert n > 0
    assert "late_rules" not in m.synthesizer.sample_log
    assert m.synthesizer.sample_log["rule_violations"]["review"][
        "days(customer_id.first_order) <= days(review_date)"] == n


def test_late_rule_not_enforced_when_a_child_reads_the_columns(tmp_path):
    """A redraw must not leave another table's rule on stale values: reported, with the reason."""
    spec = _shop(tmp_path)
    db = spec["source"]["path"]
    with sqlite3.connect(db) as con:
        rev = pd.read_sql("select review_id, review_date from review", con)
        rev["reply_id"] = rev["review_id"]
        rev["reply_date"] = rev["review_date"]
        rev[["reply_id", "review_id", "reply_date"]].to_sql("reply", con, index=False)
    spec["tables"]["reply"] = {"role": "entity", "primary_key": "reply_id",
                               "columns": {"reply_id": IDC, "review_id": IDC, "reply_date": {"type": "date"}}}
    spec["relationships"].append({"child": "reply", "child_key": "review_id", "parent": "review",
                                  "parent_key": "review_id"})
    spec["rules"].append({"table": "reply", "type": "check", "expr": "days(review_id.review_date) <= days(reply_date)"})
    m = rs.fit(spec)
    m.sample()
    rec = m.synthesizer.sample_log["late_rules"]["review"]
    assert rec["violating"] > 0 and rec["redrawn"] == 0
    assert "reply" in rec["not_enforced"]


# ---------------------------------------------------------------- unique link keys

def _policies(tmp_path, n_pol=3000, n_named=600, seed=8):
    """Every policy has exactly one holder and at most one named driver: unique [policy_id, role]."""
    rng = np.random.default_rng(seed)
    pol = pd.DataFrame({"policy_id": np.arange(1, n_pol + 1), "product": rng.choice(["car", "van"], n_pol, p=[0.8, 0.2]),
                        "premium": np.round(rng.gamma(3, 150, n_pol), 2)})
    n_p = n_pol + n_named
    per = pd.DataFrame({"person_id": np.arange(1, n_p + 1), "age": rng.integers(18, 85, n_p)})
    hp = np.r_[np.arange(1, n_pol + 1), rng.choice(np.arange(1, n_pol + 1), n_named, replace=False)]
    role = pd.DataFrame({"role_id": np.arange(1, n_p + 1), "policy_id": hp, "person_id": per["person_id"],
                         "role": ["HOLDER"] * n_pol + ["NAMED"] * n_named})
    # claims are filed by holders only: the role explains which rows have children, so it stratifies
    # the rows and is set before the links (as a disposition's type is)
    hold = role.loc[role["role"] == "HOLDER", "role_id"].to_numpy()
    nc = int(0.8 * n_pol)
    claim = pd.DataFrame({"claim_id": np.arange(1, nc + 1), "role_id": rng.choice(hold, nc),
                          "amount": np.round(rng.gamma(2, 500, nc), 2)})
    db = tmp_path / "pol.db"
    _write(db, {"policy": pol, "person": per, "policy_role": role, "claim": claim})
    spec = {"version": 1, "name": "pol", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "policy": {"role": "entity", "primary_key": "policy_id",
                           "columns": {"policy_id": IDC, "product": {"type": "categorical"},
                                       "premium": {"type": "float"}}},
                "person": {"role": "entity", "primary_key": "person_id",
                           "columns": {"person_id": IDC, "age": {"type": "integer"}}},
                "policy_role": {"role": "entity", "primary_key": "role_id",
                                "unique": [["person_id"], ["policy_id", "role"]],
                                "columns": {"role_id": IDC, "policy_id": IDC, "person_id": IDC,
                                            "role": {"type": "categorical"}}},
                "claim": {"role": "entity", "primary_key": "claim_id",
                          "columns": {"claim_id": IDC, "role_id": IDC, "amount": {"type": "float"}}}},
            "relationships": [
                {"child": "claim", "child_key": "role_id", "parent": "policy_role", "parent_key": "role_id"},
                {"child": "policy_role", "child_key": "policy_id", "parent": "policy", "parent_key": "policy_id"},
                {"child": "policy_role", "child_key": "person_id", "parent": "person", "parent_key": "person_id"}],
            "rules": [],
            "synthesis": {"seed": 3}}
    return spec


def test_unique_link_key_exact(tmp_path):
    m = rs.fit(_policies(tmp_path))
    out = m.sample().to_dict()
    r = out["policy_role"]
    assert len(r) and (r["role"] == "NAMED").any()
    assert not r.duplicated(["policy_id", "role"]).any()       # one holder, at most one named driver per policy
    assert not r.duplicated(["person_id"]).any()
    u = m.synthesizer.sample_log["uniqueness"]["policy_role"]
    assert u["unresolved_rows"] == 0


def test_unique_key_of_the_link_alone_dense(tmp_path):
    """One desk per employee with as many desks as employees: a dense matching on one link."""
    rng = np.random.default_rng(1)
    n = 300
    emp = pd.DataFrame({"employee_id": np.arange(1, n + 1), "team": rng.choice(list("ABCD"), n)})
    desk = pd.DataFrame({"desk_id": np.arange(1, n + 1), "employee_id": rng.permutation(n) + 1,
                         "floor": rng.integers(1, 5, n)})
    db = tmp_path / "desk.db"
    _write(db, {"employee": emp, "desk": desk})
    spec = {"version": 1, "name": "desk", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "employee": {"role": "entity", "primary_key": "employee_id",
                             "columns": {"employee_id": IDC, "team": {"type": "categorical"}}},
                "desk": {"role": "entity", "primary_key": "desk_id", "unique": [["employee_id"]],
                         "columns": {"desk_id": IDC, "employee_id": IDC, "floor": {"type": "integer"}}}},
            "relationships": [{"child": "desk", "child_key": "employee_id", "parent": "employee",
                               "parent_key": "employee_id"}],
            "rules": [], "synthesis": {"seed": 2}}
    m = rs.fit(spec)
    d = m.sample().to_dict()["desk"]
    assert d["employee_id"].notna().all()
    assert not d.duplicated(["employee_id"]).any()
    assert m.synthesizer.sample_log["uniqueness"]["desk"]["unresolved_rows"] == 0
