# SPDX-License-Identifier: Apache-2.0
"""Ground-truth checks for the Contoso Data Generator V2 schema (sales, orders,
orderrows, customer, product, store, date, currencyexchange).

Every check states a property of the generator's code (DatabaseGenerator/Engine.cs,
sql-bi/Contoso-Data-Generator-V2) and measures it on the source and on the
synthetic folder.  Both folders hold one CSV per table; names are matched without
case.  Output: a markdown table (and optionally JSON) of source vs synthetic.

    python contoso_checks.py --real SRC_DIR --syn SYN_DIR --out checks.md [--json checks.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ["sales", "orders", "orderrows", "customer", "product", "store", "date", "currencyexchange"]


def load(folder):
    folder = Path(folder)
    files = {p.stem.lower(): p for p in folder.glob("*.csv")}
    out = {}
    for t in TABLES:
        if t in files:
            df = pd.read_csv(files[t], low_memory=False)
            df.columns = [c.lower() for c in df.columns]
            out[t] = df
    return out


def dt(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def share(mask):
    mask = pd.Series(mask)
    return float(mask.mean()) if len(mask) else float("nan")


def measures(d):
    m = {}
    s, o, r = d.get("sales"), d.get("orders"), d.get("orderrows")
    cu, p, st, ce, da = (d.get(k) for k in ("customer", "product", "store", "currencyexchange", "date"))
    for t in TABLES:
        m[f"rows {t}"] = len(d[t]) if t in d else None
    if s is not None:
        s = s.copy()
        s["od"], s["dd"] = dt(s["orderdate"]), dt(s["deliverydate"])
        # T1 one header per order
        hdr = ["customerkey", "storekey", "orderdate", "deliverydate", "currencycode"]
        g = s.groupby("orderkey")[hdr].nunique()
        m["T1 sales: orders with one header (customer, store, dates, currency)"] = share((g == 1).all(axis=1))
        m["T1 sales: unique (order, line)"] = share(~s.duplicated(["orderkey", "linenumber"]))
        n = s.groupby("orderkey").size()
        m["T1 sales: orders"] = int(len(n))
        m["T1 sales: lines per order (mean)"] = round(float(n.mean()), 3)
        m["T1 sales: max lines per order"] = int(n.max())
        ln = s.groupby("orderkey")["linenumber"]
        m["T1 sales: first line number is 0"] = share(ln.min() == 0)
        m["T1 sales: line numbers without gaps (generator skips some)"] = share((ln.max() - ln.min() + 1) == ln.size())
        # T2 order key encodes the day: keys sorted by number never go back in time
        so = s.drop_duplicates("orderkey").sort_values("orderkey")
        m["T2 order key order = date order"] = share(so["od"].diff().dropna() >= pd.Timedelta(0))
        # T3 exchange rate = rate(OrderDate, USD -> currency)
        if ce is not None:
            ce2 = ce.copy()
            ce2["d"], ce2["to"] = dt(ce2["date"]), ce2["tocurrency"].astype(str).str.strip()
            look = ce2[ce2["fromcurrency"].astype(str).str.strip() == "USD"].drop_duplicates(["d", "to"]).set_index(
                ["d", "to"])["exchange"]
            key = pd.MultiIndex.from_arrays([s["od"], s["currencycode"].astype(str).str.strip()])
            rate = look.reindex(key).to_numpy()
            m["T3 ExchangeRate = rate(OrderDate, USD -> CurrencyCode)"] = share(np.isclose(rate, s["exchangerate"], rtol=1e-6))
        # T4 currency of the customer's country
        if cu is not None:
            cc = s.merge(cu[["customerkey", "country", "geoareakey", "startdt", "enddt"]], on="customerkey", how="left")
            mp = {"AU": "AUD", "CA": "CAD", "FR": "EUR", "DE": "EUR", "IT": "EUR", "NL": "EUR", "GB": "GBP", "US": "USD"}
            m["T4 CurrencyCode = currency of customer's country"] = share(cc["country"].map(mp) == cc["currencycode"].str.strip())
            # T7 customer active on the order date
            m["T7 customer active: StartDT < OrderDate < EndDT"] = share((dt(cc["startdt"]) < cc["od"]) & (cc["od"] < dt(cc["enddt"])))
            if st is not None:
                cs = cc.merge(st[["storekey", "geoareakey", "countrycode", "opendate", "closedate"]].rename(
                    columns={"geoareakey": "store_geo"}), on="storekey", how="left")
                online = cs["store_geo"] == -1
                m["T5 online share of lines"] = round(share(online), 4)
                phys = cs[~online]
                m["T5 store in the customer's geo area (physical stores)"] = share(phys["store_geo"] == phys["geoareakey"])
                m["T5 store in the customer's country (physical stores)"] = share(
                    phys["countrycode"].astype(str).str.strip() == phys["country"].astype(str).str.strip())
                od = phys["od"]
                op, cl = dt(phys["opendate"]), dt(phys["closedate"])
                m["T6 store open on the order date"] = share((op.isna() | (op <= od)) & (cl.isna() | (cl >= od)))
                gap = (cs["dd"] - cs["od"]).dt.days
                m["T8 physical store: DeliveryDate = OrderDate"] = share(gap[~online] == 0)
                m["T8 online: DeliveryDate >= OrderDate + 1"] = share(gap[online] >= 1)
                m["T8 online: mean delivery days"] = round(float(gap[online].mean()), 3)
        # T9 prices
        disc = 1 - s["netprice"] / s["unitprice"]
        m["T9 NetPrice = UnitPrice x (1 - d/100), d integer 0..14"] = share(
            (np.abs(disc * 100 - np.round(disc * 100)) < 1e-3) & (np.round(disc * 100) >= 0) & (np.round(disc * 100) <= 14))
        m["T9 share without discount"] = round(share(np.round(disc * 100) == 0), 4)
        m["T10 Quantity in 1..10"] = share(s["quantity"].between(1, 10))
        m["T10 Quantity mean"] = round(float(s["quantity"].mean()), 3)
        if p is not None:
            sp = s.merge(p[["productkey", "price", "cost", "categorykey"]], on="productkey", how="left")
            f_price = sp["unitprice"] / sp["price"]
            f_cost = sp["unitcost"] / sp["cost"]
            m["T11 UnitPrice/Price = UnitCost/Cost (one price factor)"] = share(np.isclose(f_price, f_cost, rtol=1e-4))
            sp["f"] = f_price.round(4)
            gg = sp.groupby(["categorykey", "od"])["f"].nunique()
            m["T11 one price factor per (category, day)"] = share(gg == 1)
    if s is not None and o is not None and r is not None:
        j = r.merge(o, on="orderkey", how="left", suffixes=("", "_o"))
        cols = ["customerkey", "storekey", "orderdate", "deliverydate", "currencycode", "productkey", "quantity",
                "unitprice", "netprice", "unitcost"]
        a = j.set_index(["orderkey", "linenumber"])[cols].astype(str)
        b = s.set_index(["orderkey", "linenumber"])[cols].astype(str)
        common = a.index.intersection(b.index)
        m["T12 sales = orders x orderrows: rows matched both ways"] = round(
            len(common) / max(len(a.index.union(b.index)), 1), 4)
        eq = (a.loc[common].apply(lambda c: c.str.replace(r"\.0+$", "", regex=True)) ==
              b.loc[common].apply(lambda c: c.str.replace(r"\.0+$", "", regex=True))).all(axis=1)
        m["T12 matched rows identical in every column"] = share(eq)
        m["T12 order rows whose order exists"] = share(r["orderkey"].isin(o["orderkey"]))
        m["T12 orders with at least one row"] = share(o["orderkey"].isin(r["orderkey"]))
    if p is not None:
        m["T13 product: SubCategoryKey -> SubCategoryName, CategoryKey"] = share(
            p.groupby("subcategorykey")[["subcategoryname", "categorykey"]].nunique().eq(1).all(axis=1))
        m["T13 product: Price > Cost"] = share(p["price"] > p["cost"])
    if st is not None:
        m["T13 store: CountryCode -> CountryName"] = share(st.groupby("countrycode")["countryname"].nunique() == 1)
        m["T13 store: one online store (GeoAreaKey -1)"] = int((st["geoareakey"] == -1).sum())
        stat = st["status"].fillna("").astype(str).str.strip()
        m["T13 store: CloseDate set <=> Status set (Closed, Restructured)"] = share(st["closedate"].notna() == (stat != ""))
        m["T13 store: OpenDate < CloseDate"] = share(st["closedate"].isna() | (dt(st["opendate"]) < dt(st["closedate"])))
    if cu is not None:
        m["T14 customer: Country -> Continent"] = share(cu.groupby("country")["continent"].nunique() == 1)
        m["T14 customer: (Country, State) -> StateFull"] = share(cu.groupby(["country", "state"])["statefull"].nunique() == 1)
        m["T14 customer: StartDT < EndDT"] = share(dt(cu["startdt"]) < dt(cu["enddt"]))
        b = dt(cu["birthday"])
        m["T14 customer: Age = whole years to one reference date (modal share)"] = round(float(
            (cu["age"] + b.dt.year).value_counts(normalize=True).iloc[0]), 4) if len(cu) else None
        m["T14 customer: unique key"] = share(~cu["customerkey"].duplicated())
        if s is not None:
            m["T14 customers with at least one order"] = round(share(cu["customerkey"].isin(s["customerkey"])), 4)
            m["T14 orders per buying customer (mean)"] = round(float(s.groupby("customerkey")["orderkey"].nunique().mean()), 3)
    if da is not None:
        d0 = dt(da["date"])
        m["T15 date: Year = year(Date)"] = share(da["year"] == d0.dt.year)
        m["T15 date: MonthNumber = month(Date)"] = share(da["monthnumber"] == d0.dt.month)
        m["T15 date: DateKey = yyyymmdd"] = share(da["datekey"].astype(str) == d0.dt.strftime("%Y%m%d"))
        m["T15 date: one row per day, no gaps"] = share(d0.sort_values().diff().dropna() == pd.Timedelta(days=1))
    if ce is not None:
        e = ce.copy()
        e["fromcurrency"], e["tocurrency"] = e["fromcurrency"].str.strip(), e["tocurrency"].str.strip()
        same = e["fromcurrency"] == e["tocurrency"]
        m["T16 rate(X -> X) = 1"] = share(np.isclose(e.loc[same, "exchange"], 1))
        inv = e.merge(e, left_on=["date", "fromcurrency", "tocurrency"], right_on=["date", "tocurrency", "fromcurrency"])
        m["T16 rate(X -> Y) x rate(Y -> X) = 1 (4 decimals)"] = share(np.isclose(inv["exchange_x"] * inv["exchange_y"], 1, atol=5e-3))
        m["T16 unique (date, from, to)"] = share(~e.duplicated(["date", "fromcurrency", "tocurrency"]))
    return m


def dists(d):
    """Distributions compared between source and synthetic (T17): where orders
    come from, how the online share and the latitude depend on the country, and
    how customers spread over geography codes."""
    o, cu = d.get("orders"), d.get("customer")
    if o is None or cu is None:
        return {}
    oc = o.merge(cu[["customerkey", "country", "latitude", "geoareakey"]], on="customerkey", how="left")
    online = oc["storekey"] == oc["storekey"].max()
    return {"orders by customer country": oc["country"].value_counts(normalize=True),
            "online share by customer country": online.groupby(oc["country"]).mean(),
            "mean latitude by country (customers)": cu.groupby("country")["latitude"].mean(),
            "customers by geography code": cu["geoareakey"].value_counts(normalize=True),
            "customers by country": cu["country"].value_counts(normalize=True)}


def compare(dr, ds):
    out = {}
    for k, a in dr.items():
        if k not in ds:
            continue
        b = ds[k]
        ix = a.index.union(b.index)
        a2, b2 = a.reindex(ix, fill_value=0), b.reindex(ix, fill_value=0)
        if k.startswith(("orders by", "customers by")):
            out[f"T17 {k}: TVD"] = round(float(0.5 * (a2 - b2).abs().sum()), 4)
        else:
            out[f"T17 {k}: max |diff|"] = round(float((a.reindex(ix) - b.reindex(ix)).abs().max()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--syn")
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args()
    real = load(a.real)
    mr = measures(real)
    ms = {}
    if a.syn:
        syn = load(a.syn)
        ms = measures(syn)
        cmp_ = compare(dists(real), dists(syn))
        mr.update({k: 0.0 for k in cmp_})
        ms.update(cmp_)
    fmt = lambda v: "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = ["# Contoso ground-truth checks", "", "| Check | Source | Synthetic |", "|---|---:|---:|"]
    for k in mr:
        lines.append(f"| {k} | {fmt(mr[k])} | {fmt(ms.get(k)) if a.syn else '-'} |")
    txt = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
    if a.json:
        Path(a.json).write_text(json.dumps({"real": mr, "syn": ms}, indent=1, default=str))


if __name__ == "__main__":
    main()
