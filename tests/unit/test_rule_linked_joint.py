# SPDX-License-Identifier: Apache-2.0
"""Categoricals that check rules constrain together are modelled as one joint over their observed
combinations (TableModel._join_rule_linked).

Two classes of data: coded transactions whose type, operation and symbol codes allow only some
combinations (bank transactions, ledger entries), and one-hot indicator groups where exactly one
indicator is set (land-cover areas, survey answer options).  Drawn one by one through the copula,
most rows break the rules and rejecting them shifts each column's shares; drawn jointly, the rows
lie in the support and keep the source's shares.
"""
from __future__ import annotations

import copy
import sqlite3

import numpy as np
import pandas as pd

import relsynth as rs

# Codes of a ledger: each operation and each symbol belongs to one type (credit or debit), interest has
# no operation; operation x symbol has more combinations than one pairwise joint may hold, so only a
# joint of all three keeps them
CR_OPS, DR_OPS = ["CASH_IN", "XFER_IN", "CHEQUE_IN"], ["CASH_OUT", "XFER_OUT", "CARD", "STANDING"]
CR_SYM, DR_SYM = ["PENSION", "SALARY", "REFUND"], ["FEE", "BILL", "LOAN", "INSURANCE", "RENT", "TAX"]


def _combos(rng):
    out = [("CR", None, "INT")]
    for t, ops, syms in (("CR", CR_OPS, CR_SYM), ("DR", DR_OPS, DR_SYM)):
        for o in ops:
            for s in [None, *syms]:
                if rng.random() < 0.6:
                    out.append((t, o, s))
    w = rng.uniform(1.0, 6.0, len(out))
    return out, w / w.sum()


def _coded_db(tmp_path, n=8000, seed=2):
    rng = np.random.default_rng(seed)
    combos, p = _combos(rng)
    pick = rng.choice(len(combos), n, p=p)
    df = pd.DataFrame({"tid": np.arange(1, n + 1), "type": [combos[i][0] for i in pick],
                       "operation": [combos[i][1] for i in pick], "symbol": [combos[i][2] for i in pick],
                       "amount": np.round(rng.gamma(2.0, 300.0, n), 2)})
    db = tmp_path / "coded.db"
    with sqlite3.connect(db) as con:
        df.to_sql("trans", con, index=False)
    return db, df


def _in(col, vals):
    return f"{col} in ({', '.join(repr(v) for v in vals)})"


def _coded_spec(db):
    return {"version": 1, "name": "coded", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"trans": {"role": "entity", "primary_key": "tid",
                                 "columns": {"tid": {"type": "id"}, "type": {"type": "categorical"},
                                             "operation": {"type": "categorical"},
                                             "symbol": {"type": "categorical"}, "amount": {"type": "float"}}}},
            "relationships": [],
            "rules": [{"table": "trans", "type": "check",
                       "expr": "isnull(operation) == (notnull(symbol) and symbol == 'INT')"},
                      {"table": "trans", "type": "check",
                       "expr": f"(type == 'CR' and (isnull(operation) or {_in('operation', CR_OPS)})) or "
                               f"(type == 'DR' and {_in('operation', DR_OPS)})"},
                      {"table": "trans", "type": "check",
                       "expr": f"isnull(symbol) or (type == 'CR' and {_in('symbol', ['INT', *CR_SYM])}) or "
                               f"(type == 'DR' and {_in('symbol', DR_SYM)})"}]}


def _shares(df, cols):
    return df[cols].astype(object).where(df[cols].notna(), "<NA>").value_counts(normalize=True)


def test_coded_transactions_keep_their_joint(tmp_path):
    db, src = _coded_db(tmp_path)
    r = rs.run(_coded_spec(db), write=False, privacy=False, seed=3, privacy_check=False)
    syn = r.data["trans"]
    cols = ["type", "operation", "symbol"]
    a, b = _shares(src, cols), _shares(syn, cols)
    idx = a.index.union(b.index)
    tvd = 0.5 * (a.reindex(idx, fill_value=0) - b.reindex(idx, fill_value=0)).abs().sum()
    assert tvd < 0.04                                         # the joint is the source's
    assert set(b.index) <= set(a.index)                       # no combination outside the source
    assert abs(syn["symbol"].isna().mean() - src["symbol"].isna().mean()) < 0.02
    assert sorted(r.model.synthesizer.tm["trans"].rule_groups[0]) == ["operation", "symbol", "type"]


def test_one_hot_group_keeps_exactly_one_indicator(tmp_path):
    rng = np.random.default_rng(5)
    n = 4000
    area = rng.choice(4, n, p=[0.45, 0.05, 0.38, 0.12])
    df = pd.DataFrame({"pid": np.arange(1, n + 1), "elev": rng.normal(2900, 250, n).round()})
    for k in range(4):
        df[f"area_{k}"] = (area == k).astype(int)
    db = tmp_path / "onehot.db"
    with sqlite3.connect(db) as con:
        df.to_sql("plots", con, index=False)
    cols = {f"area_{k}": {"type": "boolean"} for k in range(4)}
    spec = {"version": 1, "name": "oh", "source": {"kind": "sqlite", "path": str(db)},
            "tables": {"plots": {"role": "entity", "primary_key": "pid",
                                 "columns": {"pid": {"type": "id"}, "elev": {"type": "float"}, **cols}}},
            "relationships": [],
            "rules": [{"table": "plots", "type": "check", "expr": "area_0 + area_1 + area_2 + area_3 == 1"}]}
    syn = rs.run(spec, write=False, privacy=False, seed=1, privacy_check=False).data["plots"]
    on = syn[[f"area_{k}" for k in range(4)]].astype(int)
    assert (on.sum(axis=1) == 1).all()
    share = on.mean().to_numpy()
    assert np.abs(share - np.array([0.45, 0.05, 0.38, 0.12])).max() < 0.03


def test_rare_combination_is_not_joined(tmp_path):
    """A combination held by fewer rows than the small-count threshold is not made a level."""
    from relsynth.spec import load_spec
    from relsynth.table_model import TableModel
    db, src = _coded_db(tmp_path, n=3000)
    src.loc[src.index[:2], ["type", "operation", "symbol"]] = ["DR", "CARD", "FEE"]   # 2 rows only
    spec = load_spec(_coded_spec(db))
    tm = TableModel("trans", spec["tables"]["trans"], src, _cfg(), rule_groups=[{"type", "operation", "symbol"}])
    assert tm.rule_groups == []
    tm2 = TableModel("trans", spec["tables"]["trans"], src.iloc[2:], _cfg(),
                     rule_groups=[{"type", "operation", "symbol"}])
    assert sorted(tm2.rule_groups[0]) == ["operation", "symbol", "type"]


def _cfg():
    from relsynth.spec import DEFAULT_SYNTHESIS
    return copy.deepcopy(DEFAULT_SYNTHESIS)
