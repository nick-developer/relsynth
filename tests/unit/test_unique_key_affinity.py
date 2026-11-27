# SPDX-License-Identifier: Apache-2.0
"""A unique key over a link and a stratifying column, together with link affinity.

Class: link tables with one row per parent and role (an account's owner disposition, a policy's
holder, a vehicle's registered keeper) whose second link points mostly at a parent sharing a value
(the client lives in the account's district).  The unique key is enforced when the role column
stratifies the rows (it is set before the links), and affinity moves keep it: a move to a parent
whose key is taken exchanges the two rows' links.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import relsynth as rs


def _db(tmp_path, n_acc=600, n_disp2=120, n_dist=10, match=0.9, seed=4):
    rng = np.random.default_rng(seed)
    dist = pd.DataFrame({"district_id": np.arange(1, n_dist + 1), "region": [f"R{i % 3}" for i in range(n_dist)]})
    acc = pd.DataFrame({"account_id": np.arange(1, n_acc + 1), "district_id": rng.integers(1, n_dist + 1, n_acc),
                        "frequency": rng.choice(["monthly", "weekly"], n_acc, p=[0.8, 0.2])})
    n_cli = n_acc + n_disp2
    owners = np.arange(1, n_acc + 1)
    co = np.r_[owners, rng.choice(owners, n_disp2, replace=False)]
    same = rng.random(n_cli) < match
    cdist = np.where(same, acc.set_index("account_id").loc[co, "district_id"].to_numpy(),
                     rng.integers(1, n_dist + 1, n_cli))
    cli = pd.DataFrame({"client_id": np.arange(1, n_cli + 1), "district_id": cdist,
                        "gender": rng.choice(["F", "M"], n_cli)})
    disp = pd.DataFrame({"disp_id": np.arange(1, n_cli + 1), "client_id": cli["client_id"], "account_id": co,
                         "type": ["OWNER"] * n_acc + ["DISPONENT"] * n_disp2})
    db = tmp_path / "disp.db"
    with sqlite3.connect(db) as con:
        for name, df in (("district", dist), ("account", acc), ("client", cli), ("disp", disp)):
            df.to_sql(name, con, index=False)
    return db, disp, cli, acc


def _spec(db):
    idc = {"type": "id"}
    return {"version": 1, "name": "disp", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {
                "district": {"role": "entity", "primary_key": "district_id", "mode": "copy",
                             "columns": {"district_id": idc, "region": {"type": "categorical"}}},
                "account": {"role": "entity", "primary_key": "account_id",
                            "columns": {"account_id": idc, "district_id": idc, "frequency": {"type": "categorical"}}},
                "client": {"role": "entity", "primary_key": "client_id",
                           "columns": {"client_id": idc, "district_id": idc, "gender": {"type": "categorical"}}},
                "disp": {"role": "entity", "primary_key": "disp_id",
                         "unique": [["client_id"], ["account_id", "type"]],
                         "link_affinity": [{"match": ["client_id.district_id", "account_id.district_id"]}],
                         "columns": {"disp_id": idc, "client_id": idc, "account_id": idc,
                                     "type": {"type": "categorical"}}}},
            "relationships": [
                {"child": "account", "child_key": "district_id", "parent": "district", "parent_key": "district_id"},
                {"child": "client", "child_key": "district_id", "parent": "district", "parent_key": "district_id"},
                {"child": "disp", "child_key": "client_id", "parent": "client", "parent_key": "client_id"},
                {"child": "disp", "child_key": "account_id", "parent": "account", "parent_key": "account_id"}],
            "rules": []}


def _match(d, cli, acc):
    c = cli.set_index("client_id")["district_id"]
    a = acc.set_index("account_id")["district_id"]
    return float((d["client_id"].map(c) == d["account_id"].map(a)).mean())


def test_unique_role_key_and_affinity_both_hold(tmp_path):
    db, disp, cli, acc = _db(tmp_path)
    out = rs.run(_spec(db), write=False, privacy=False, seed=2, privacy_check=False).data
    d = out["disp"]
    assert len(d) > 0 and (d["type"] == "DISPONENT").any()
    assert d.duplicated(["account_id", "type"]).mean() <= 0.01          # one owner per account
    assert not d.duplicated(["client_id"]).any()
    src, syn = _match(disp, cli, acc), _match(d, out["client"], out["account"])
    assert abs(syn - src) < 0.08, (src, syn)                            # the affinity is kept
