# SPDX-License-Identifier: Apache-2.0
"""Checks C1700-C1705: privacy units with composite keys and several units
(gap plan G1; docs/proofs section v2.6, Props. 1700-1704).

C1700  ownership comes from the IR: composite links are followed, and a row
       reached along several paths belongs to every person they reach.
C1701  the attack suite's subsets and profiles use the same union ownership.
C1702  DP mode decodes real composite keys: groups of rows with line numbers,
       unique keys and valid composite links, group sizes within the cap.
C1703  two units in one DP run: removing one person of either unit moves each
       statistic of that unit's tables by at most its charged sensitivity and
       leaves the other unit's statistics exactly as they were; each unit's
       accountant stays within its own budget.
C1704  refusals before any read: a table holding both units' data, a link
       between the units' data, and a governed run with a unit its domain lacks.
C1705  the same replay on AdventureWorks DW, customers and employees (needs the
       bench cache; skipped without it).

    pytest -m science tests/science/test_privacy_units_checks.py
"""
from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.science]
opendp = pytest.importorskip("opendp")

import relsynth.dp as dp  # noqa: E402
from relsynth.dp import mechanisms as M  # noqa: E402
from relsynth.dp import relational as R  # noqa: E402
from relsynth.errors import PrivacyError  # noqa: E402
from relsynth.ir import lift  # noqa: E402
from relsynth.ir.privacy import contribution  # noqa: E402
from relsynth.privacy import unit as U  # noqa: E402
from relsynth.spec import load_spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AW_DB = ROOT / "bench" / ".cache" / "adventureworks" / "AdventureWorksDW.db"
NAME = {"type": "pii", "pii": {"generator": "person_name"}}


# --------------------------------------------------------------------- fixture
def _raw_spec():
    """Customers with orders of several lines, each line with sales reasons
    (composite link order_no + line_no), and employees with quotas."""
    return {
        "name": "units",
        "source": {"kind": "csv", "path": "."},
        "tables": {
            "customer": {"role": "entity", "primary_key": "customer_id", "columns": {
                "customer_id": {"type": "id"}, "name": NAME, "age": {"type": "integer"},
                "segment": {"type": "categorical"}}},
            "employee": {"role": "entity", "primary_key": "employee_id", "columns": {
                "employee_id": {"type": "id"}, "email": {"type": "pii", "pii": {"generator": "email"}},
                "grade": {"type": "categorical"}, "salary": {"type": "float"}}},
            "reason": {"role": "entity", "primary_key": "reason_id", "columns": {
                "reason_id": {"type": "id"}, "label": {"type": "categorical"}}},
            "product": {"role": "entity", "primary_key": "product_id", "columns": {
                "product_id": {"type": "id"}, "price": {"type": "float"}}},
            "sale": {"role": "fact", "unique": [["order_no", "line_no"]],
                     "fact": {"anchor": "customer", "partner": "product",
                              "header": {"key": "order_no", "columns": ["customer_id"], "line_number": "line_no"}},
                     "columns": {
                "customer_id": {"type": "id"}, "product_id": {"type": "id"}, "order_no": {"type": "id"},
                "line_no": {"type": "sequence", "within": "order_no"},
                "amount": {"type": "float"}, "channel": {"type": "categorical"}}},
            "sale_reason": {"role": "fact", "fact": {}, "unique": [["order_no", "line_no", "reason_id"]],
                            "columns": {"order_no": {"type": "id"}, "line_no": {"type": "id"},
                                        "reason_id": {"type": "id"}}},
            "quota": {"role": "fact", "fact": {}, "columns": {
                "employee_id": {"type": "id"}, "target": {"type": "float"}}},
        },
        "relationships": [
            {"child": "sale", "child_key": "customer_id", "parent": "customer", "parent_key": "customer_id"},
            {"child": "sale", "child_key": "product_id", "parent": "product", "parent_key": "product_id"},
            {"child": "sale_reason", "child_key": ["order_no", "line_no"], "parent": "sale",
             "parent_key": ["order_no", "line_no"]},
            {"child": "sale_reason", "child_key": "reason_id", "parent": "reason", "parent_key": "reason_id"},
            {"child": "quota", "child_key": "employee_id", "parent": "employee", "parent_key": "employee_id"},
        ],
    }


def _tables(n_cust=120, n_emp=40, seed=5):
    rng = np.random.default_rng(seed)
    cust = pd.DataFrame({"customer_id": np.arange(1, n_cust + 1), "name": [f"c{i}" for i in range(n_cust)],
                         "age": rng.integers(18, 80, n_cust), "segment": rng.choice(["a", "b", "c"], n_cust)})
    emp = pd.DataFrame({"employee_id": np.arange(1, n_emp + 1), "email": [f"e{i}@x.org" for i in range(n_emp)],
                        "grade": rng.choice(["g1", "g2"], n_emp), "salary": rng.gamma(4, 1e4, n_emp)})
    reason = pd.DataFrame({"reason_id": [1, 2, 3, 4], "label": ["price", "quality", "ad", "other"]})
    rows, rs = [], []
    order = 1000
    for c in cust["customer_id"]:
        # the first customer is heavy, so truncation binds for them
        for _ in range(int(rng.integers(0, 4)) + (6 if c == 1 else 0)):
            order += 1
            for line in range(1, int(rng.integers(1, 5)) + 1):
                rows.append((c, int(rng.integers(1, 21)), f"SO{order}", line, float(rng.gamma(2, 50)),
                             str(rng.choice(["web", "shop"]))))
                for r in rng.choice([1, 2, 3, 4], int(rng.integers(0, 3)), replace=False):
                    rs.append((f"SO{order}", line, int(r)))
    sale = pd.DataFrame(rows, columns=["customer_id", "product_id", "order_no", "line_no", "amount", "channel"])
    product = pd.DataFrame({"product_id": np.arange(1, 21), "price": rng.gamma(2, 20, 20)})
    sr = pd.DataFrame(rs, columns=["order_no", "line_no", "reason_id"])
    quota = pd.DataFrame({"employee_id": rng.choice(emp["employee_id"], 3 * n_emp),
                          "target": rng.gamma(3, 1e5, 3 * n_emp)})
    return {"customer": cust, "employee": emp, "reason": reason, "product": product, "sale": sale, "sale_reason": sr,
            "quota": quota}


def _spec(**dp_kw):
    s = _raw_spec()
    if dp_kw:
        s["synthesis"] = {"privacy": {"dp": {"epsilon": 2.0, "delta": 1e-9, "public_tables": ["reason", "product"], **dp_kw}}}
    return load_spec(s)


# --------------------------------------------------------------------- C1700
def test_c1700_composite_links_are_followed_and_paths_union():
    """Prop. 1700: the owned tables come from the IR trace, a composite link
    included; with two paths to the unit a row belongs to both people reached."""
    s = _spec()
    t = _tables()
    u = U.resolve(s, "customer")
    assert u.owned == ["sale", "sale_reason"]
    own = U.owners(u, t)
    # every sales reason belongs to the customer of its order line
    look = t["sale"].set_index(["order_no", "line_no"])["customer_id"]
    want = [look[(o, ln)] for o, ln in zip(t["sale_reason"]["order_no"], t["sale_reason"]["line_no"])]
    assert own["sale_reason"].sort_index().tolist() == want
    # the IR's contribution and the attack suite agree on whose rows are whose
    ids = list(t["customer"]["customer_id"].sample(30, random_state=1))
    masks = contribution(lift(s), t, ids, "customer")
    sub = U.subset(u, t, ids)
    for tab in ("customer", "sale", "sale_reason"):
        assert int(masks[tab].sum()) == len(sub[tab]), tab
    assert U.resolve(s, "employee").owned == ["quota"]

    # two paths that disagree: a referral row reaches the referrer and the referred customer's
    # account; the row belongs to both
    raw = {"tables": {
        "customer": {"role": "entity", "primary_key": "id", "columns": {"id": {"type": "id"}, "name": NAME}},
        "account": {"role": "entity", "primary_key": "id", "columns": {"id": {"type": "id"}, "cid": {"type": "id"}}},
        "event": {"role": "fact", "fact": {}, "columns": {"cid": {"type": "id"}, "aid": {"type": "id"},
                                                          "x": {"type": "float"}}}},
        "relationships": [{"child": "account", "child_key": "cid", "parent": "customer", "parent_key": "id"},
                          {"child": "event", "child_key": "cid", "parent": "customer", "parent_key": "id"},
                          {"child": "event", "child_key": "aid", "parent": "account", "parent_key": "id"}]}
    s2 = load_spec(raw)
    d = {"customer": pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}),
         "account": pd.DataFrame({"id": [10, 20], "cid": [1, 2]}),
         "event": pd.DataFrame({"cid": [1, 1, 2], "aid": [10, 20, 20], "x": [1.0, 2.0, 3.0]})}
    u2 = U.resolve(s2, "customer")
    assert len(u2.paths["event"]) == 2
    o = U.owners(u2, d)["event"]
    assert sorted(o[o.index == 1].tolist()) == [1, 2]          # row 1: customer 1 directly, customer 2 via account
    assert U.subset(u2, d, [1])["event"]["x"].tolist() == [1.0, 2.0]
    assert U.subset(u2, d, [2])["event"]["x"].tolist() == [2.0, 3.0]

    # people who share a row go to the same side of the attack suite's split, so no control
    # profile holds a row the release was fitted on
    d3 = {"customer": pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "name": list("abcdef")}),
          "account": pd.DataFrame({"id": [10, 20, 30], "cid": [1, 2, 3]}),
          "event": pd.DataFrame({"cid": [1, 4, 5, 6], "aid": [20, 30, 30, 10], "x": [1.0, 2.0, 3.0, 4.0]})}
    for seed in range(8):
        tr, ctl = U.split(u2, d3, 0.5, seed=seed)
        assert sorted([*tr, *ctl]) == [1, 2, 3, 4, 5, 6]
        side = {**{x: "t" for x in tr}, **{x: "c" for x in ctl}}
        assert side[1] == side[2] and side[3] == side[4] == side[5] and side[6] == side[1]
    # one group holds everyone: the split falls back to people and keeps both sides
    d3["event"] = pd.concat([d3["event"], pd.DataFrame({"cid": [2], "aid": [30], "x": [5.0]})], ignore_index=True)
    tr, ctl = U.split(u2, d3, 0.5, seed=0)
    assert len(tr) == 3 and len(ctl) == 3
    info: dict = {}
    U.split_people(u2, d3, np.array([1, 2, 3, 4, 5, 6]), [3], info=info)
    assert info == {"shared_rows_fallback": 6}
    # a group never pushes control past its size: it goes on to the next part that holds it
    d4 = {**d3, "event": d3["event"].iloc[:2]}                    # groups {1, 2}, {3, 4}; 5 and 6 alone
    ctl, tr = U.split_people(u2, d4, np.array([5, 1, 2, 6, 3, 4]), [3])
    assert sorted(ctl) == [1, 2, 5] and sorted(tr) == [3, 4, 6]
    ctl, tr = U.split_people(u2, d4, np.array([5, 6, 1, 2, 3, 4]), [3])
    assert len(ctl) <= 3 and sorted([*ctl, *tr]) == [1, 2, 3, 4, 5, 6]
    assert not ({1, 2} & set(ctl) and {1, 2} & set(tr))


# --------------------------------------------------------------------- C1701
def test_c1701_profiles_count_a_shared_row_for_each_owner():
    """Prop. 1700 in the attack suite: a person's profile counts every row they
    own, through the composite link too."""
    s = _spec()
    t = _tables()
    u = U.resolve(s, "customer")
    prof = U.profiles(s, u, t)
    assert "sale_reason.n_rows" in prof.columns
    n = t["sale_reason"].merge(t["sale"], on=["order_no", "line_no"]).groupby("customer_id").size()
    got = prof["sale_reason.n_rows"]
    assert all(got.loc[c] == n.get(c, 0) for c in prof.index)


# --------------------------------------------------------------------- C1702
def test_c1702_dp_decodes_real_composite_keys():
    """Prop. 1702: the seed has orders of several lines (not one order per row),
    line numbers 1..k within each order, unique composite keys and valid
    composite links; no order is larger than the rows a customer may keep."""
    t = _tables()
    spec = _spec(unit="customer", max_degree={"sale": 6, "sale_reason": 2},
                 public_tables=["reason", "product", "employee", "quota"])
    cfg = dp.parse(spec)
    pl = R.make_plan(spec, cfg)
    P = pl.tables["sale"]
    assert P.key_seq == {"line_no": "order_no"} and P.key_groups == ["order_no"]
    model = R.fit(spec, t, cfg, M.Accountant(cfg.rho, cfg.delta), pl)
    assert "order_no" in model.tables["sale"].group_sizes
    seed = R.decode(spec, model)
    sale, sr = seed["sale"], seed["sale_reason"]
    assert len(sale) > 0 and not sale.duplicated(["order_no", "line_no"]).any()
    per_order = sale.groupby("order_no").size()
    assert per_order.max() > 1 and per_order.max() <= 6
    assert sale.groupby("order_no")["customer_id"].nunique().max() == 1      # an order is one customer's
    uk = ["order_no", "line_no", "reason_id"]
    assert not sr.dropna(subset=uk).duplicated(uk).any()        # a key with a null part never collides (SQL)
    keys = set(zip(sale["order_no"], sale["line_no"]))
    assert all(k in keys for k in zip(sr["order_no"], sr["line_no"]))

    # a declared unique key with a null part never collides (as in SQL): those rows stay;
    # a repeated full key is dropped
    res = {k: v.copy() for k, v in seed.items()}
    res["sale"] = sale.head(3).assign(order_no="SO1", line_no=[None, None, 1]).reset_index(drop=True)
    res["sale"] = pd.concat([res["sale"], res["sale"].tail(1)], ignore_index=True)
    R._enforce_checks(spec, pl, res)
    assert res["sale"]["line_no"].isna().sum() == 2 and (res["sale"]["line_no"] == 1).sum() == 1


# --------------------------------------------------------------------- C1703
class _Recorder:
    """mechanisms.gaussian stand-in (as in C902): records each exact statistic and
    its sensitivity; replays the first run's outputs into the neighbour's run."""

    def __init__(self, replay=None):
        self.calls, self.replay = [], replay

    def __call__(self, values, sensitivity, rho, acct, stage, mechanism):
        v = np.asarray(values, dtype=np.int64).ravel()
        acct.charge(M.Charge(stage, mechanism, rho))
        i = len(self.calls)
        self.calls.append((mechanism, v.copy(), float(sensitivity)))
        if self.replay is None:
            return v
        name, out, _ = self.replay.calls[i]
        assert name == mechanism and len(out) == len(v), (name, mechanism, len(out), len(v))
        return out


def _publicize(spec, tables, units, max_degree):
    """Ranges and levels declared public, and fixed degree caps, so the shapes of
    the statistics do not depend on the data (as C902 does)."""
    s = copy.deepcopy(spec)
    cfg = dp.parse(s)
    pl = R.make_plan(s, cfg)
    pub = {}
    for t in pl.order:
        P = pl.tables[t]
        for c in P.numeric:
            x = R._numeric_values(tables[t][c], s["tables"][t]["columns"][c]["type"])
            x = x[~np.isnan(x)]
            if len(x):
                pub[f"{t}.{c}"] = {"min": float(x.min()), "max": float(x.max()) + 1}
        for c in P.categorical:
            lv = sorted(tables[t][c].dropna().astype(str).unique())
            if lv:
                pub[f"{t}.{c}"] = {"levels": lv}
            else:
                pub[f"{t}.{c}"] = {"levels": ["-"]}
    md = {t: max_degree for t in pl.order if t not in units}
    s["synthesis"]["privacy"]["dp"].update(public=pub, max_degree=md)
    return load_spec(s)


def _fit(spec, tables, rec, monkeypatch):
    monkeypatch.setattr(R, "gaussian", rec)
    monkeypatch.setattr(R, "_hash_key", lambda: "0123456789abcdef")
    cfg = dp.parse(spec)
    pl = R.make_plan(spec, cfg)
    accts = {u: M.Accountant(cfg.rho, cfg.delta) for u in cfg.units}
    R.fit(spec, {t: df.reset_index(drop=True) for t, df in tables.items()}, cfg, accts, pl)
    return pl, accts


def _replay_two_units(spec, tables, removals, monkeypatch):
    """Base run, then one run per removed person; returns the worst ratio of
    movement to sensitivity per statistic."""
    base = _Recorder()
    pl, accts = _fit(spec, tables, base, monkeypatch)
    cfg = dp.parse(spec)
    for a in accts.values():
        assert a.spent[0] <= cfg.rho * (1 + 1e-9)
    table_of = {m: m.split(":")[0].split(".")[0] for m, _, _ in base.calls}
    worst: dict = {}
    for u_name, pid in removals:
        unit = U.resolve(spec, u_name, others=set(cfg.units))
        keep = tables[u_name][unit.key][tables[u_name][unit.key] != pid]
        nb = {**tables, **{t: df for t, df in U.subset(unit, tables, keep).items() if t in unit.paths}}
        rep = _Recorder(replay=base)
        _fit(spec, nb, rep, monkeypatch)
        for (name, v0, sens), (_, v1, _) in zip(base.calls, rep.calls):
            d = float(np.linalg.norm(v0 - v1))
            owner = pl.owner[table_of[name]]
            if owner != u_name:
                # the other unit's statistics read none of this person's rows (Prop. 1703)
                assert d == 0.0, (u_name, pid, name, d)
                continue
            worst[name] = max(worst.get(name, 0.0), d / sens)
            assert d <= sens + 1e-9, (u_name, pid, name, d, sens)
    return worst


def test_c1703_two_units_sensitivity_replay(monkeypatch):
    """Props. 1702-1703: customers and employees in one run."""
    t = _tables()
    spec = _publicize(_spec(unit=["customer", "employee"]), t, {"customer", "employee"}, 4)
    heavy = int(t["sale"]["customer_id"].value_counts().index[0])
    busy = int(t["quota"]["employee_id"].value_counts().index[0])
    rem = [("customer", heavy), ("customer", 7), ("customer", 50), ("employee", busy), ("employee", 3)]
    worst = _replay_two_units(spec, t, rem, monkeypatch)
    assert worst["sale: marginals"] > 0.3 and worst["customer: marginals"] > 0.3, worst
    assert worst["quota: marginals"] > 0.0, worst


def test_c1703_dropped_link_between_units_is_never_read(monkeypatch):
    """Prop. 1703: a dropped link from a customer to their account manager (an
    employee) is not measured; removing a customer leaves every employee
    statistic as it was, and removing an employee every customer statistic."""
    raw = _raw_spec()
    raw["tables"]["customer"]["columns"]["rep_id"] = {"type": "id"}
    raw["relationships"].append({"child": "customer", "child_key": "rep_id", "parent": "employee",
                                 "parent_key": "employee_id"})
    raw["synthesis"] = {"privacy": {"dp": {"unit": ["customer", "employee"], "epsilon": 2.0, "delta": 1e-9,
                                           "public_tables": ["reason", "product"],
                                           "drop_columns": ["customer.rep_id"]}}}
    t = _tables()
    t["customer"]["rep_id"] = np.random.default_rng(3).choice(t["employee"]["employee_id"], len(t["customer"]))
    spec = load_spec(raw)
    pl, _, _ = dp.plan(spec)
    assert [r["child"] for r in pl.tables["employee"].children] == ["quota"]
    assert "rep_id" in pl.tables["customer"].dropped
    spec = _publicize(spec, t, {"customer", "employee"}, 4)
    rep = int(t["customer"]["rep_id"].value_counts().index[0])
    heavy = int(t["sale"]["customer_id"].value_counts().index[0])
    worst = _replay_two_units(spec, t, [("customer", heavy), ("customer", 9), ("employee", rep)], monkeypatch)
    assert not any("per employee" in m and "quota" not in m for m in worst), worst


# --------------------------------------------------------------------- C1704
def test_c1704_refusals_before_any_read():
    """Prop. 1703 (scope): a table with data of both units, a link between the
    units' data are refused by the plan, and a governed run with a unit its
    domain does not have before anything is reserved."""
    raw = _raw_spec()
    raw["tables"]["sale"]["columns"]["employee_id"] = {"type": "id"}
    raw["relationships"].append({"child": "sale", "child_key": "employee_id", "parent": "employee",
                                 "parent_key": "employee_id"})
    raw["synthesis"] = {"privacy": {"dp": {"unit": ["customer", "employee"], "epsilon": 1.0, "delta": 1e-9,
                                           "public_tables": ["reason", "product"]}}}
    with pytest.raises(PrivacyError, match="holds data of both customer and employee"):
        dp.plan(load_spec(raw))
    raw2 = _raw_spec()
    raw2["tables"]["customer"]["columns"]["rep_id"] = {"type": "id"}
    raw2["relationships"].append({"child": "customer", "child_key": "rep_id", "parent": "employee",
                                  "parent_key": "employee_id"})
    raw2["synthesis"] = raw["synthesis"]
    with pytest.raises(PrivacyError, match="links data of customer to data of employee"):
        dp.plan(load_spec(raw2))
    raw2["synthesis"]["privacy"]["dp"]["drop_columns"] = ["customer.rep_id"]
    pl, cfg, _ = dp.plan(load_spec(raw2))
    assert cfg.units == ["customer", "employee"] and pl.owned_by("employee") == ["employee", "quota"]

    class _Pol:
        domains = {"d": {"unit": {"entity": "customer"}}}

        @staticmethod
        def caps(_d):
            class C:
                delta = 1e-9
            return C
    # a governed run may protect only units of its domain (Prop. 1704)
    with pytest.raises(PrivacyError, match="the spec protects 'employee' but the privacy unit of domain 'd'"):
        dp.domain_override(load_spec(raw2), _Pol, "d")


def test_c1704_run_states_one_guarantee_per_unit():
    """Prop. 1703: a whole two-unit run states each unit's guarantee and spends
    at most the budget of each."""
    t = _tables()
    res = dp.synthesize(_spec(unit=["customer", "employee"]), tables=t, write=False, evaluate=False)
    st = res.statement
    assert st["schema"] == "relsynth.dp-statement/2" and st["unit"] == ["customer", "employee"]
    assert set(st["units"]) == {"customer", "employee"}
    for u, v in res.per_unit.items():
        assert v["rho"] <= res.config.rho * (1 + 1e-9) and v["epsilon"] <= 2.0 + 1e-9
    assert {c.unit for c in res.charges} == {"customer", "employee"}
    assert res.data["quota"]["employee_id"].isin(res.data["employee"]["employee_id"]).all()


# --------------------------------------------------------------------- C1705
@pytest.mark.skipif(not AW_DB.exists(), reason="AdventureWorks DW not in the bench cache (python -m bench fetch)")
def test_c1705_adventureworks_customers_and_employees(monkeypatch):
    """The acceptance check of G1: FactInternetSalesReason is owned through its
    composite link, and the two-unit replay holds on AdventureWorks DW."""
    from relsynth.data import coerce
    s = load_spec(ROOT / "bench" / "specs" / "adventureworks.yaml")
    assert "FactInternetSalesReason" in U.resolve(s, "DimCustomer").owned
    s["synthesis"].setdefault("privacy", {})["dp"] = {
        "unit": ["DimCustomer", "DimEmployee"], "epsilon": 2.0, "delta": 1e-9, "public_tables": "all_unowned",
        "drop_columns": ["DimEmployee.ParentEmployeeKey"]}
    s = load_spec(s)
    con = sqlite3.connect(AW_DB)
    try:
        names = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        t = {n: coerce(pd.read_sql(f'select * from "{n}"', con), s["tables"][n], optional=set())  # noqa: S608
             for n in s["tables"] if n in names}
    finally:
        con.close()
    spec = _publicize(s, t, {"DimCustomer", "DimEmployee"}, 4)
    pl, _ = dp.plan(spec)[:2]
    assert pl.owner["FactInternetSalesReason"] == "DimCustomer" and pl.owner["FactResellerSales"] == "DimEmployee"
    sales = t["FactInternetSales"]["CustomerKey"].value_counts()
    reps = t["FactResellerSales"]["EmployeeKey"].value_counts()
    rem = [("DimCustomer", int(sales.index[0])), ("DimCustomer", int(sales.index[len(sales) // 2])),
           ("DimEmployee", int(reps.index[0])), ("DimEmployee", int(t["DimEmployee"]["EmployeeKey"].iloc[0]))]
    worst = _replay_two_units(spec, t, rem, monkeypatch)
    assert worst["FactInternetSalesReason: marginals"] > 0.0, worst
