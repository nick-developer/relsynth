# SPDX-License-Identifier: Apache-2.0
"""AdventureWorks DW ground-truth checks (G1..G13 in LOG.md) on a synthetic CSV folder.

Independent of relsynth: reads the synthetic CSVs and the source database, and reports each
invariant's share of rows/orders that satisfy it in the source and in the synthetic data.

    python aw_checks.py --synth run1_raw_draft/synthetic --url "<sqlalchemy url>" --out run1_raw_draft/aw_checks.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

TABLES = ["DimDate", "DimCustomer", "DimGeography", "DimProduct", "DimPromotion", "DimReseller", "DimEmployee",
          "FactInternetSales", "FactResellerSales", "FactInternetSalesReason", "FactProductInventory",
          "FactCurrencyRate", "FactFinance", "FactSalesQuota", "FactSurveyResponse", "FactCallCenter",
          "DimProductSubcategory", "DimCurrency", "DimSalesTerritory"]


def load_synth(folder):
    out = {}
    for t in TABLES:
        p = Path(folder) / f"{t}.csv"
        if p.exists():
            out[t] = pd.read_csv(p, low_memory=False)
    return out


def load_source(url):
    eng = create_engine(url)
    return {t: pd.read_sql(f"SELECT * FROM dbo.{t}", eng) for t in TABLES}


def ymd(s):
    d = pd.to_datetime(s, errors="coerce")
    return (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).astype("Int64")


def share(mask):
    mask = pd.Series(mask).dropna()
    return float(mask.mean()) if len(mask) else float("nan")


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def checks(db):
    r = {}
    for t in ("FactInternetSales", "FactResellerSales"):
        f = db.get(t)
        if f is None or not len(f):
            continue
        k = "FIS" if t == "FactInternetSales" else "FRS"
        up, q, ext, dis, pct = num(f, "UnitPrice"), num(f, "OrderQuantity"), num(f, "ExtendedAmount"), num(f, "DiscountAmount"), num(f, "UnitPriceDiscountPct")
        sa, cost, sc, tax, frt = num(f, "SalesAmount"), num(f, "TotalProductCost"), num(f, "ProductStandardCost"), num(f, "TaxAmt"), num(f, "Freight")
        r[f"G1 {k} Extended = UnitPrice*Qty"] = share((ext - up * q).abs() <= 0.01)
        r[f"G1 {k} Sales = Extended - Discount"] = share((sa - (ext - dis)).abs() <= 0.01)
        r[f"G1 {k} Discount = Extended*Pct"] = share((dis - ext * pct).abs() <= 0.01)
        r[f"G1 {k} Cost = StdCost*Qty"] = share((cost - sc * q).abs() <= 0.01)
        r[f"G1 {k} Tax = 8% Sales"] = share((tax - sa * 0.08).abs() <= 0.01)
        r[f"G1 {k} Freight = 2.5% Sales"] = share((frt - sa * 0.025).abs() <= 0.01)
        hdr = ["CustomerKey"] if k == "FIS" else ["ResellerKey", "EmployeeKey"]
        hdr = [c for c in hdr + ["OrderDateKey", "SalesTerritoryKey", "CurrencyKey"] if c in f]
        g = f.groupby("SalesOrderNumber")
        r[f"G2 {k} orders"] = int(g.ngroups)
        r[f"G2 {k} lines per order (mean)"] = round(len(f) / max(g.ngroups, 1), 3)
        r[f"G2 {k} orders with one header (customer/date/territory/currency)"] = share(g[hdr].nunique().max(axis=1) <= 1)
        ln = num(f, "SalesOrderLineNumber")
        r[f"G2 {k} line numbers 1..n"] = share(ln.groupby(f["SalesOrderNumber"]).agg(lambda x: sorted(x) == list(range(1, len(x) + 1))))
        r[f"G2 {k} unique (order, line)"] = share(~f.duplicated(["SalesOrderNumber", "SalesOrderLineNumber"]))
        od, dd, sd = ymd(f["OrderDate"]), ymd(f["DueDate"]), ymd(f["ShipDate"])
        r[f"G6 {k} OrderDateKey = OrderDate"] = share(num(f, "OrderDateKey") == od)
        r[f"G6 {k} ShipDateKey = ShipDate"] = share(num(f, "ShipDateKey") == sd)
        o, s_, d_ = pd.to_datetime(f["OrderDate"], errors="coerce"), pd.to_datetime(f["ShipDate"], errors="coerce"), pd.to_datetime(f["DueDate"], errors="coerce")
        r[f"G6 {k} Ship = Order + 7d"] = share((s_ - o).dt.days == 7)
        r[f"G6 {k} Due = Order + 12d"] = share((d_ - o).dt.days == 12)
        r[f"G6 {k} OrderDateKey <= ShipDateKey <= DueDateKey"] = share((num(f, "OrderDateKey") <= num(f, "ShipDateKey")) & (num(f, "ShipDateKey") <= num(f, "DueDateKey")))
        if "DimProduct" in db:
            p = db["DimProduct"].set_index("ProductKey")
            lp = num(f, "ProductKey").map(pd.to_numeric(p["ListPrice"], errors="coerce"))
            r[f"G5 {k} UnitPrice = product ListPrice"] = share((up - lp).abs() <= 0.005)
            stc = num(f, "ProductKey").map(pd.to_numeric(p["StandardCost"], errors="coerce"))
            r[f"G5 {k} ProductStandardCost = product StandardCost"] = share((sc - stc).abs() <= 0.005)
        geo_parent = "DimCustomer" if k == "FIS" else "DimReseller"
        geo_key = "CustomerKey" if k == "FIS" else "ResellerKey"
        if geo_parent in db and "DimGeography" in db:
            gp = db[geo_parent].set_index(geo_parent.replace("Dim", "") + "Key")["GeographyKey"]
            gt = db["DimGeography"].set_index("GeographyKey")["SalesTerritoryKey"]
            terr = num(f, geo_key).map(num(gp.reset_index(), "GeographyKey").set_axis(gp.index)).map(gt)
            r[f"G3 {k} SalesTerritoryKey = territory of {geo_parent} geography"] = share(num(f, "SalesTerritoryKey") == terr)
    f = db.get("FactInternetSales")
    if f is not None and "DimCustomer" in db:
        c = db["DimCustomer"]
        r["G4 customers with >= 1 internet sale"] = share(num(c, "CustomerKey").isin(num(f, "CustomerKey")))
        first = pd.to_datetime(f["OrderDate"], errors="coerce").groupby(num(f, "CustomerKey")).min()
        dfp = pd.to_datetime(c["DateFirstPurchase"], errors="coerce")
        r["G13 DateFirstPurchase = first internet order"] = share(dfp.dt.normalize() == num(c, "CustomerKey").map(first).dt.normalize())
    fr = db.get("FactInternetSalesReason")
    if fr is not None and f is not None:
        key = lambda d: d["SalesOrderNumber"].astype(str) + "#" + num(d, "SalesOrderLineNumber").astype("Int64").astype(str)
        r["G12 FIS lines without a sales reason"] = share(~key(f).isin(set(key(fr))))
        r["G12 FISR rows whose (order, line) exists in FIS"] = share(key(fr).isin(set(key(f))))
        r["G12 FISR unique (order, line, reason)"] = share(~fr.duplicated(["SalesOrderNumber", "SalesOrderLineNumber", "SalesReasonKey"]))
    inv = db.get("FactProductInventory")
    if inv is not None:
        np_, nd = num(inv, "ProductKey").nunique(), num(inv, "DateKey").nunique()
        r["G8 FPI rows / (products x dates)"] = round(len(inv) / max(np_ * nd, 1), 4)
        r["G8 FPI unique (product, date)"] = share(~inv.duplicated(["ProductKey", "DateKey"]))
        r["G8 FPI MovementDate = DateKey"] = share(num(inv, "DateKey") == ymd(inv["MovementDate"]))
        s = inv.assign(_p=num(inv, "ProductKey"), _d=num(inv, "DateKey")).sort_values(["_p", "_d"])
        prev = num(s, "UnitsBalance").groupby(s["_p"]).shift()
        ok = (num(s, "UnitsBalance") - (prev + num(s, "UnitsIn") - num(s, "UnitsOut"))).abs() < 0.5
        r["G8 FPI UnitsBalance = prev + In - Out"] = share(ok[prev.notna()])
    for t, col in [("FactFinance", "Date"), ("FactCurrencyRate", "Date"), ("FactSalesQuota", "Date"),
                   ("FactSurveyResponse", "Date"), ("FactCallCenter", "Date")]:
        if t in db and col in db[t]:
            r[f"G6 {t} DateKey = {col}"] = share(num(db[t], "DateKey") == ymd(db[t][col]))
    if "DimDate" in db:
        d = db["DimDate"]
        r["G10 DimDate DateKey = FullDateAlternateKey"] = share(num(d, "DateKey") == ymd(d["FullDateAlternateKey"]))
        r["G10 DimDate unique FullDateAlternateKey"] = share(~d["FullDateAlternateKey"].duplicated())
    fsr = db.get("FactSurveyResponse")
    if fsr is not None and "DimProductSubcategory" in db:
        sub = db["DimProductSubcategory"].set_index("ProductSubcategoryKey")["ProductCategoryKey"]
        r["G14 FSR category = category of subcategory"] = share(num(fsr, "ProductCategoryKey") == num(fsr, "ProductSubcategoryKey").map(num(sub.reset_index(), "ProductCategoryKey").set_axis(sub.index)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, syn = load_source(a.url), load_synth(a.synth)
    rs, rf = checks(src), checks(syn)
    rows = ["| Check | Source | Synthetic |", "|---|---:|---:|"]
    fmt = lambda v: "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    for k in list(dict.fromkeys(list(rs) + list(rf))):
        rows.append(f"| {k} | {fmt(rs.get(k))} | {fmt(rf.get(k))} |")
    counts = ["| Table | Source rows | Synthetic rows |", "|---|---:|---:|"] + \
             [f"| {t} | {len(src[t])} | {len(syn[t]) if t in syn else 'n/a'} |" for t in TABLES]
    Path(a.out).write_text("# AdventureWorks ground-truth checks\n\n" + "\n".join(counts) + "\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(counts + [""] + rows))


if __name__ == "__main__":
    main()
