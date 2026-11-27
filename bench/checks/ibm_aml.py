# SPDX-License-Identifier: Apache-2.0
"""IBM AML (IT-AML, HI-Small) ground-truth checks.

Independent of relsynth.  M1-M7 and M12 are identities on the real data (share 1.0, except M7): the dataset
specification's composite account identity, self-loops and FX coupling, and the data facts the bench
spec states (bench/specs/ibm_aml.yaml).  M8-M11 are measures recorded without a verdict.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = ("bank", "account", "currency", "trans")


def load(folder) -> dict:
    out = {}
    for t in TABLES:
        p = Path(folder) / f"{t}.csv"
        if p.exists():
            # codes and account ids are text (leading zeros)
            out[t] = pd.read_csv(p, low_memory=False, dtype={"bank_id": str, "account_id": str, "from_bank": str,
                                                              "to_bank": str})
    return out


def _share(mask) -> float | None:
    mask = pd.Series(mask).dropna()
    return float(mask.mean()) if len(mask) else None


def checks(T: dict, ref: dict) -> dict:
    tr, acc, ccy = T["trans"], T["account"], ref["currency"].set_index("currency")["usd_rate"]
    self_loop = tr["from_account_key"] == tr["to_account_key"]
    same = tr["payment_currency"] == tr["receiving_currency"]
    rate = tr["receiving_currency"].map(ccy) / tr["payment_currency"].map(ccy)
    expect = tr["amount_paid"] * rate
    a = acc.set_index("account_key")["bank_id"]
    out = {
        "M1 account id unique within its bank": _share(~acc.duplicated(["bank_id", "account_id"])),
        "M2 sender's bank is the account's bank": _share(tr["from_account_key"].map(a) == tr["from_bank"]),
        "M3 reinvestment is a self-loop": _share(self_loop[tr["payment_format"] == "Reinvestment"]),
        "M4 same currency, same amount": _share((tr["amount_paid"] - tr["amount_received"]).abs()[same] <= 0.005),
        "M5 cross-currency payments are ACH or Wire": _share(tr["payment_format"][~same].isin(["ACH", "Wire"])),
        "M6 received amount at the fixed rate": _share(
            ((tr["amount_received"] - expect).abs() <= (0.002 * tr["amount_received"].abs()).clip(lower=0.01))[~same]),
        "M7 Bitcoin payments in Bitcoin": _share((tr["payment_currency"] == "Bitcoin")[tr["payment_format"] == "Bitcoin"]),
        "M8 self-loop share": _share(self_loop),
        "M9 intra-bank share": _share((tr["from_bank"] == tr["to_bank"]) & ~self_loop),
        "M10 laundering share": _share(tr["is_laundering"] == 1),
        "M11 laundering share of ACH": _share((tr["is_laundering"] == 1)[tr["payment_format"] == "ACH"]),
    }
    # M12: the specification applies the rate "at Timestamp"; the share of (currency pair, day) groups whose
    # median rate is within 0.5 % of the pair's rate over the whole period says whether it moves with time
    x = tr[~same & (tr["amount_paid"] >= 100)].assign(day=lambda d: pd.to_datetime(d["ts"]).dt.date,
                                                       r=lambda d: d["amount_received"] / d["amount_paid"])
    if len(x):
        pair = x.groupby(["payment_currency", "receiving_currency"])["r"].transform("median")
        g = x.assign(dev=(x["r"] / pair - 1).abs()).groupby(["payment_currency", "receiving_currency", "day"])
        med = g["dev"].median()[g.size() >= 5]
        out["M12 rate constant through the period (pair-days within 0.5 %)"] = _share(med <= 0.005)
    else:
        out["M12 rate constant through the period (pair-days within 0.5 %)"] = None
    return out
