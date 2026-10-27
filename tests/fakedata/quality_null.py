# SPDX-License-Identifier: Apache-2.0
"""A fixed generating law for the 2.0 quality checks.

`shop(seed, scale)` draws one database from a law whose parameters do not
depend on the seed, so two seeds give two independent draws of the same law:
the null hypothesis of every quality check.  `defects` derives databases that
break one property each, to test that the checks see it.

    region (8, copied) <- customer (1200 x scale) <- orders <- lines -> product (150 x scale)

Customers carry a latent segment that drives age, income, region, the number
of orders (negative binomial) and which products their lines buy; order
amounts follow the customer's income; line quantities follow the product price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 3
SEG_P = np.array([0.5, 0.3, 0.2])
AGE_MU = np.array([30.0, 45.0, 62.0])
INC_MU = np.array([10.0, 10.6, 11.2])
REGION_P = np.array([[.30, .20, .10, .10, .10, .10, .05, .05],
                     [.05, .10, .30, .25, .10, .10, .05, .05],
                     [.05, .05, .05, .10, .15, .20, .20, .20]])
ORDERS_MU = np.array([1.5, 3.0, 6.0])
ORDERS_SIZE = 2.0                                  # negative binomial size (over-dispersed)
CAT_LEVELS = ["food", "home", "toys", "tech", "garden"]
CAT_P = np.array([.35, .25, .15, .15, .10])
AFFINITY = np.array([[3.0, 1.0, 1.0, .5, .5], [1.0, 2.0, .5, 2.0, 1.0], [.5, 1.0, .5, 1.0, 3.0]])
N_PROD = 150


def _products(n=N_PROD):
    rng = np.random.default_rng(12345)              # the catalogue is part of the law (at scale 1)
    cat = rng.choice(len(CAT_LEVELS), n, p=CAT_P)
    price = np.round(np.exp(rng.normal(2.5 + 0.3 * cat, 0.4)), 2)
    return pd.DataFrame({"product_id": np.arange(1, n + 1), "category": np.array(CAT_LEVELS)[cat],
                         "price": price}), cat


def spec():
    return {
        "name": "quality_null", "source": {"kind": "csv", "path": "."},
        "tables": {
            "region": {"primary_key": "region_id", "mode": "copy",
                       "columns": {"region_id": "id", "region_name": "categorical"}},
            "customer": {"primary_key": "customer_id", "columns": {
                "customer_id": "id", "region_id": "id", "age": {"type": "integer"}, "income": "float",
                "member": "boolean", "signup": "date"}},
            "product": {"primary_key": "product_id", "columns": {
                "product_id": "id", "category": "categorical", "price": "float"}},
            "orders": {"primary_key": "order_id", "columns": {
                "order_id": "id", "customer_id": "id", "channel": "categorical", "amount": "float",
                "order_date": "date"}},
            "lines": {"role": "fact", "primary_key": "line_id", "fact": {"anchor": "orders"}, "columns": {
                "line_id": "id", "order_id": "id", "product_id": "id", "quantity": {"type": "integer"},
                "discount": "float"}},
        },
        "relationships": [
            {"child": "customer", "child_key": "region_id", "parent": "region"},
            {"child": "orders", "child_key": "customer_id", "parent": "customer"},
            {"child": "lines", "child_key": "order_id", "parent": "orders"},
            {"child": "lines", "child_key": "product_id", "parent": "product"},
        ],
    }


def shop(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    n = int(round(1200 * scale))
    prod, pcat = _products(int(round(N_PROD * scale)))   # resized like relsynth resizes a dimension
    region = pd.DataFrame({"region_id": np.arange(1, 9), "region_name": [f"R{i}" for i in range(1, 9)]})
    z = rng.choice(K, n, p=SEG_P)
    reg = np.array([rng.choice(8, p=REGION_P[k]) for k in z]) + 1
    age = np.clip(np.round(rng.normal(AGE_MU[z], 8)), 18, 95).astype(int)
    income = np.round(np.exp(rng.normal(INC_MU[z], 0.3)), 2)
    member = rng.random(n) < 0.2 + 0.25 * z
    signup = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2000, n), unit="D")
    cust = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(1, n + 1)], "region_id": reg, "age": age,
                         "income": income, "member": member, "signup": signup})
    no = rng.negative_binomial(ORDERS_SIZE, ORDERS_SIZE / (ORDERS_SIZE + ORDERS_MU[z]))
    ou = np.repeat(np.arange(n), no)
    m = len(ou)
    channel = np.where(rng.random(m) < np.array([0.7, 0.5, 0.3])[z[ou]], "store", "online")
    amount = np.round(np.exp(0.6 * (np.log(income[ou]) - 10.5) + rng.normal(3.5, 0.5, m)), 2)
    odate = signup[ou] + pd.to_timedelta(rng.integers(0, 600, m), unit="D")
    orders = pd.DataFrame({"order_id": np.arange(1, m + 1), "customer_id": cust.customer_id.to_numpy()[ou],
                           "channel": channel, "amount": amount, "order_date": odate})
    nl = 1 + rng.poisson(np.where(channel == "store", 1.0, 2.0))
    lo = np.repeat(np.arange(m), nl)
    w = AFFINITY[z[ou[lo]]][:, pcat]                # (lines, products) affinity weights
    w = w / w.sum(1, keepdims=True)
    pick = (rng.random(len(lo))[:, None] > w.cumsum(1)).sum(1)
    pick = np.minimum(pick, len(prod) - 1)
    qty = 1 + rng.poisson(8.0 / prod.price.to_numpy()[pick])
    disc = np.where(rng.random(len(lo)) < 0.3, np.round(rng.uniform(0, 0.3, len(lo)), 2), np.nan)
    lines = pd.DataFrame({"line_id": np.arange(1, len(lo) + 1), "order_id": orders.order_id.to_numpy()[lo],
                          "product_id": prod.product_id.to_numpy()[pick], "quantity": qty, "discount": disc})
    return {"region": region, "customer": cust, "product": prod, "orders": orders, "lines": lines}


def defects(db, seed=0):
    """Databases that each break one property of `db`."""
    rng = np.random.default_rng(seed)
    out = {}
    d = {k: v.copy() for k, v in db.items()}
    d["customer"]["income"] = d["customer"]["income"] * 1.3            # marginal shift
    out["income_shift"] = d
    d = {k: v.copy() for k, v in db.items()}
    d["customer"]["age"] = rng.permutation(d["customer"]["age"].to_numpy())   # breaks age~income~region
    out["age_shuffled"] = d
    d = {k: v.copy() for k, v in db.items()}
    d["orders"]["customer_id"] = rng.permutation(d["orders"]["customer_id"].to_numpy())  # cross-table link
    out["orders_relinked"] = d
    d = {k: v.copy() for k, v in db.items()}
    keep = rng.random(len(d["customer"])) < 0.5                      # half the customers lose their orders,
    lost = set(d["customer"].customer_id[~keep])                      # the others get them (same total)
    kept = d["customer"].customer_id[keep].to_numpy()
    o = d["orders"]
    mv = o.customer_id.isin(lost).to_numpy()
    o.loc[mv, "customer_id"] = rng.choice(kept, mv.sum())
    out["fanout_concentrated"] = d
    return out
