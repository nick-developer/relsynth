# SPDX-License-Identifier: Apache-2.0
"""Berka (PKDD'99 Czech bank) ground-truth checks on a set of tables.

Independent of relsynth.  Each check is a share of rows (or accounts) that obey
a rule of the bank's data, or a measure; `checks(tables, ref)` computes them on
any copy of the database, `ref` being the real one (some rules are "the pairs
the real data has").  F1-F11 are identities on the real data (share 1.0);
F12-F16 are measures recorded without a verdict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]


def load(folder) -> dict:
    out = {}
    for t in TABLES:
        p = Path(folder) / f"{t}.csv"
        if p.exists():
            out[t] = pd.read_csv(p, low_memory=False)
    return out


def _date(s):
    return pd.to_datetime(s, errors="coerce")


def _share(mask) -> float | None:
    mask = pd.Series(mask).dropna()
    return float(mask.mean()) if len(mask) else None


def checks(T: dict, ref: dict) -> dict:
    acc, disp, trans = T["account"], T["disp"], T["trans"]
    out = {}
    owners = disp[disp["type"] == "OWNER"].groupby("account_id").size()
    out["F1 one OWNER disposition per account"] = _share(owners.reindex(acc["account_id"], fill_value=0) == 1)
    out["F2 at most one disposition per client"] = _share(disp.groupby("client_id").size() == 1)
    cd = T["card"].merge(disp.rename(columns={"type": "type_disp"}), on="disp_id", how="left")
    out["F3 cards only on OWNER dispositions"] = _share(cd["type_disp"] == "OWNER")
    out["F4 at most one loan per account"] = _share(T["loan"].groupby("account_id").size() == 1)
    ln = T["loan"]
    out["F5 loan amount = duration x payments"] = _share((ln["amount"] - ln["duration"] * ln["payments"]).abs() < 1)
    adate = acc.set_index("account_id")["date"].pipe(_date)
    out["F6 transaction not before its account opened"] = _share(
        _date(trans["date"]).values >= adate.reindex(trans["account_id"]).values)
    ca = cd.merge(acc[["account_id", "date"]], on="account_id", how="left")
    out["F7 card not issued before its account opened"] = _share(_date(ca["issued"]) >= _date(ca["date"]))
    first = _date(trans["date"]).groupby(trans["account_id"]).min()
    out["F8 account opened on its first transaction"] = _share(first.reindex(adate.index) == adate)
    pairs = set(zip(ref["trans"]["type"], ref["trans"]["operation"].fillna("")))
    out["F9 transaction type and operation pair seen in the bank"] = _share(
        [p in pairs for p in zip(trans["type"], trans["operation"].fillna(""))])
    out["F10 interest (UROK) exactly on rows without operation"] = _share(
        (trans["k_symbol"] == "UROK") == trans["operation"].isna())
    t1 = trans.assign(_d=_date(trans["date"])).sort_values(["account_id", "_d", "trans_id"]).groupby("account_id").head(1)
    out["F11 first transaction of an account is a cash deposit"] = _share(t1["operation"] == "VKLAD")
    # measures
    dd = disp.merge(T["client"][["client_id", "district_id"]], on="client_id", how="left") \
        .merge(acc[["account_id", "district_id"]], on="account_id", how="left", suffixes=("_client", "_account"))
    out["F12 client and account in the same district (share)"] = _share(
        dd["district_id_client"] == dd["district_id_account"])
    out["F13 transactions per account (mean)"] = float(len(trans) / max(len(acc), 1))
    out["F14 accounts with a loan (share)"] = float(ln["account_id"].nunique() / max(len(acc), 1))
    st_r = ref["loan"]["status"].value_counts(normalize=True)
    st_s = ln["status"].value_counts(normalize=True)
    out["F15 loan status, total variation from the bank"] = float(0.5 * st_r.subtract(st_s, fill_value=0).abs().sum())
    amt = trans["amount"].astype(float)
    out["F16 median transaction amount"] = float(np.nanmedian(amt)) if len(amt) else None
    return out
