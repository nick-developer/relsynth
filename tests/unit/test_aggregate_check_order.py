# SPDX-License-Identifier: Apache-2.0
"""A check that reads a parent's aggregated column through a link (Berka:
account.date = min(date) of trans; loan.date > account_id.date and
trans.date >= account_id.date) is evaluated after the aggregate, not while the
links are drawn (sampling failed with "account.date is not available yet")."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import relsynth as rs


def test_check_through_link_to_parent_aggregate(tmp_path):
    rng = np.random.default_rng(0)
    n_acc = 60
    opened = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 300, n_acc), unit="D")
    rows = []
    for a in range(n_acc):
        k = int(rng.integers(3, 12))
        days = np.sort(rng.integers(0, 200, k))
        days[0] = 0
        for d in days:
            rows.append((a + 1, (opened[a] + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d"),
                         float(rng.gamma(2, 50))))
    trans = pd.DataFrame(rows, columns=["account_id", "date", "amount"])
    trans.insert(0, "trans_id", range(1, len(trans) + 1))
    account = pd.DataFrame({"account_id": range(1, n_acc + 1), "date": opened.strftime("%Y-%m-%d"),
                            "frequency": rng.choice(["M", "W"], n_acc)})
    has_loan = rng.random(n_acc) < 0.5
    loan = pd.DataFrame({"account_id": account["account_id"][has_loan].to_numpy(),
                         "date": (opened[has_loan] + pd.to_timedelta(rng.integers(30, 400, has_loan.sum()), unit="D"))
                         .strftime("%Y-%m-%d"), "amount": rng.gamma(3, 1000, has_loan.sum())})
    loan.insert(0, "loan_id", range(1, len(loan) + 1))
    db = tmp_path / "bank.db"
    with sqlite3.connect(db) as con:
        account.to_sql("account", con, index=False)
        trans.to_sql("trans", con, index=False)
        loan.to_sql("loan", con, index=False)
    spec = {"version": 1, "name": "bank", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"account": {"role": "entity", "primary_key": "account_id",
                                   "columns": {"account_id": {"type": "id"}, "date": {"type": "date"},
                                               "frequency": {"type": "categorical"}}},
                       "trans": {"role": "fact", "primary_key": "trans_id",
                                 "fact": {"anchor": "account", "time_column": "date", "time_bucket": "month"},
                                 "columns": {"trans_id": {"type": "id"}, "account_id": {"type": "id"},
                                             "date": {"type": "date"}, "amount": {"type": "float"}}},
                       "loan": {"role": "entity", "primary_key": "loan_id",
                                "columns": {"loan_id": {"type": "id"}, "account_id": {"type": "id"},
                                            "date": {"type": "date"}, "amount": {"type": "float"}}}},
            "relationships": [{"child": "trans", "child_key": "account_id", "parent": "account",
                               "parent_key": "account_id"},
                              {"child": "loan", "child_key": "account_id", "parent": "account",
                               "parent_key": "account_id"}],
            "rules": [{"table": "account", "type": "aggregate", "column": "date", "agg": "min", "from": "trans",
                       "of": "date"},
                      {"table": "trans", "type": "check", "expr": "days(account_id.date) <= days(date)"},
                      {"table": "loan", "type": "check", "expr": "days(account_id.date) < days(date)"}]}
    r = rs.run(spec, write=False, privacy=False, seed=1)
    acc, tr = r.data["account"], r.data["trans"]
    first = pd.to_datetime(tr["date"]).groupby(tr["account_id"]).min()
    got = pd.to_datetime(acc.set_index("account_id")["date"]).reindex(first.index)
    assert (got == first).all()
    assert len(r.data["loan"]) > 0
