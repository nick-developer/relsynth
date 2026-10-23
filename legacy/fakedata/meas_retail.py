import sys, pickle
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as S

def measures(d, truth):
    m = {}
    o, r, s, c, st, p, ce = (d[k].copy() for k in ["Orders", "OrderRows", "Sales", "Customer", "Store", "Product", "CurrencyExchange"])
    dt = lambda x: pd.to_datetime(x, errors="coerce")
    j = r.merge(o, on="OrderKey", how="left")
    cols = ["OrderKey", "LineNumber", "OrderDate", "DeliveryDate", "CustomerKey", "StoreKey", "ProductKey", "Quantity",
            "UnitPrice", "NetPrice", "UnitCost", "CurrencyCode"]
    a = j[cols].astype(str).sort_values(["OrderKey", "LineNumber"]).reset_index(drop=True)
    b = s[cols].astype(str).sort_values(["OrderKey", "LineNumber"]).reset_index(drop=True)
    m["view = join (rows and cells)"] = float((len(a) == len(b)) and (a.values == b.values).all())
    rk = ce[ce.FromCurrency == "USD"].assign(d=dt(ce.Date).dt.strftime("%Y-%m-%d")).set_index(["d", "ToCurrency"]).Exchange
    rate = rk.reindex(pd.MultiIndex.from_arrays([dt(s.OrderDate).dt.strftime("%Y-%m-%d"), s.CurrencyCode])).to_numpy()
    m["ExchangeRate = rate(OrderDate, USD, CurrencyCode)"] = float(np.isclose(rate.astype(float), s.ExchangeRate.astype(float)).mean())
    oc = o.merge(c[["CustomerKey", "Country", "StartDT", "EndDT"]], on="CustomerKey", how="left") \
          .merge(st, on="StoreKey", how="left")
    cmap = {k: v[0] for k, v in truth["countries"].items()}
    m["CurrencyCode = map(country)"] = float((oc.Country.map(cmap) == oc.CurrencyCode).mean())
    geo = truth["geo"].set_index("GeoAreaKey")
    g = c.set_index("GeoAreaKey")[["State", "StateFull", "Country", "Continent"]]
    m["customer geography combos observed"] = float((g.astype(str).values == geo.reindex(g.index.astype(int)).astype(str).values).all(axis=1).mean())
    sub = p.groupby("SubCategoryKey")[["SubCategoryName", "CategoryKey", "CategoryName"]].nunique()
    m["SubCategoryKey -> names, category"] = float((sub == 1).all(axis=1).mean())
    rp = r.merge(p[["ProductKey", "Price", "Cost"]], on="ProductKey")
    f = rp.UnitPrice.astype(float) / rp.Price.astype(float)
    m["UnitPrice/Price in factor set"] = float(np.isclose(f.to_numpy()[:, None], truth["factors"][None, :], atol=6e-4).any(axis=1).mean())
    m["UnitCost = round(Cost x same factor, 4)"] = float(np.isclose(rp.UnitCost.astype(float), np.round(rp.Cost.astype(float) * f.round(4), 4), atol=2e-4).mean())
    disc = 1 - rp.NetPrice.astype(float) / rp.UnitPrice.astype(float)
    m["NetPrice = UnitPrice (1 - d/100), d in 0..10"] = float(((np.abs(disc * 100 - np.round(disc * 100)) < 0.05) & (np.round(disc * 100).between(0, 10))).mean())
    m["share without discount"] = float((np.round(disc * 100) == 0).mean())
    ln = r.groupby("OrderKey").LineNumber
    m["line numbers 0..n-1"] = float(((ln.min() == 0) & (ln.max() == ln.size() - 1) & (ln.nunique() == ln.size())).mean())
    so = o.assign(k=pd.to_numeric(o.OrderKey)).sort_values("k")
    m["order key order = date order"] = float((dt(so.OrderDate).diff().dropna() >= pd.Timedelta(0)).mean())
    od = dt(oc.OrderDate)
    m["customer active"] = float(((dt(oc.StartDT) < od) & (od < dt(oc.EndDT))).mean())
    online = oc.StoreKey.astype(int) == 999
    phys = oc[~online]
    m["store open"] = float(((dt(phys.OpenDate) <= dt(phys.OrderDate)) & (dt(phys.CloseDate).isna() | (dt(phys.OrderDate) <= dt(phys.CloseDate)))).mean())
    m["store in customer's country"] = float((phys.CountryCode == phys.Country).mean())
    gap = (dt(oc.DeliveryDate) - od).dt.days
    m["store: same-day delivery"] = float((gap[~online] == 0).mean())
    m["online: delivery >= 1 day"] = float((gap[online] >= 1).mean())
    m["online share"] = float(online.mean())
    m["online mean delivery days"] = float(gap[online].mean())
    m["orders per customer country"] = oc.Country.value_counts(normalize=True).round(3).to_dict()
    m["orders per store"] = oc.StoreKey.astype(int).value_counts(normalize=True).round(3).to_dict()
    m["factor shares"] = f.round(2).value_counts(normalize=True).round(3).to_dict()
    m["customers with an order"] = float(c.CustomerKey.isin(o.CustomerKey).mean())
    m["orders per buying customer"] = float(o.groupby("CustomerKey").size().mean())
    if "Latitude" in c:
        m["mean latitude by country"] = c.groupby("Country").Latitude.mean().round(3).to_dict()
        m["latitude sd within country"] = c.groupby("Country").Latitude.std().round(3).to_dict()
    return m

if __name__ == "__main__":
    tables, spec, truth = S.retail_dw()
    out = pickle.load(open(sys.argv[1], "rb"))
    mr, ms = measures(tables, truth), measures(out, truth)
    for k in mr:
        print(f"{k:50s} {mr[k]}  |  {ms[k]}")
