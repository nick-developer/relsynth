# SPDX-License-Identifier: Apache-2.0
"""A fake document database with every shape the document engine handles.

customers: ObjectId _id, optional nested address, a set-valued array of tags,
           a map keyed by generated ids whose key repeats a field of the value
orders:    a reference to customers._id, an array of line items (1..8) each
           with a nested array of discounts, a polymorphic payment
           (discriminator `method`), an optional shipping sub-document, a
           length field n_items == len(items), a union field rating
           (int, double or string), a nullable and optional note
"""
from __future__ import annotations

import datetime as dt
import random

from relsynth.documents.values import Long, ObjectId, object_id

TAGS = ["new", "vip", "b2b", "promo", "eu", "us"]
SKUS = [f"SKU{i:02d}" for i in range(15)]


def collections(n_customers: int = 300, seed: int = 3) -> dict[str, list[dict]]:
    rnd = random.Random(seed)
    custs, orders = [], []
    for i in range(n_customers):
        cid = object_id("cust", seed, i)
        c = {"_id": cid, "name": f"Customer {i}", "segment": rnd.choice(["retail", "pro", "pro", "gov"]),
             "since": dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=rnd.randint(0, 2000)),
             "tags": rnd.sample(TAGS, rnd.randint(0, 3))}
        if rnd.random() < 0.8:
            c["address"] = {"city": rnd.choice(["Athens", "Berlin", "Paris", "Rome"]),
                            "zip": f"{rnd.randint(10000, 99999)}"}
        wallets = {}
        for _ in range(rnd.choice([0, 1, 1, 2])):
            wid = str(object_id("w", seed, i, len(wallets)))[:16]
            wallets[wid] = {"id": wid, "balance": round(rnd.uniform(0, 500), 2)}
        c["wallets"] = wallets
        custs.append(c)
        for k in range(rnd.choice([0, 1, 2, 2, 3, 4])):
            items = []
            for _ in range(rnd.randint(1, 8)):
                items.append({"sku": rnd.choice(SKUS), "qty": rnd.randint(1, 5),
                              "price": round(rnd.uniform(1, 90), 2),
                              "discounts": [round(rnd.uniform(0, 0.3), 2) for _ in range(rnd.choice([0, 0, 1, 2]))]})
            method = rnd.choice(["card", "card", "paypal", "bank"])
            pay = {"method": method, "amount": round(sum(x["qty"] * x["price"] for x in items), 2)}
            if method == "card":
                pay.update(last4=f"{rnd.randint(0, 9999):04d}", network=rnd.choice(["visa", "mc"]))
            elif method == "paypal":
                pay["payer"] = f"p{i}@example.com"
            else:
                pay["iban_country"] = rnd.choice(["DE", "GR", "FR"])
            o = {"_id": object_id("ord", seed, i, k), "customer_id": cid,
                 "created": dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=rnd.randint(0, 500000)),
                 "status": rnd.choice(["new", "paid", "shipped", "paid"]), "items": items, "n_items": len(items),
                 "payment": pay, "seq": Long(rnd.randint(1, 10 ** 12))}
            r = rnd.random()
            o["rating"] = rnd.randint(1, 5) if r < 0.5 else (round(rnd.uniform(1, 5), 1) if r < 0.9 else "n/a")
            if rnd.random() < 0.6:
                o["shipping"] = {"city": rnd.choice(["Athens", "Berlin", "Paris"]), "express": rnd.random() < 0.3}
            r = rnd.random()
            if r < 0.2:
                o["note"] = None
            elif r < 0.4:
                o["note"] = rnd.choice(["leave at door", "call first", "gift"])
            orders.append(o)
    return {"customers": custs, "orders": orders}


__all__ = ["collections", "ObjectId"]
