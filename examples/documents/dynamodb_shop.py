# SPDX-License-Identifier: Apache-2.0
"""A DynamoDB single-table e-commerce example (customers, orders, order items),
following the access patterns of the AWS single-table design guidance:

    get a customer's profile and orders   PK = CUSTOMER#<customer_id>
    get an order with its items           GSI1PK = ORDER#<order_id>

The items are generated (no AWS data is copied); `items(n_customers, seed)` is
deterministic.  See docs/documents.md, "DynamoDB single-table designs".
"""
from __future__ import annotations

import random

COUNTRIES = ["DE", "FR", "GR", "IT", "NL", "ES", "US"]
STATUS = ["PLACED", "SHIPPED", "DELIVERED", "RETURNED"]
SKUS = [f"SKU-{i:03d}" for i in range(40)]


def items(n_customers: int = 200, seed: int = 1) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    order_no = 0
    for c in range(n_customers):
        cid = f"C{c:05d}"
        country = rnd.choice(COUNTRIES)
        created = f"20{rnd.randint(19, 23)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"
        out.append({"PK": f"CUSTOMER#{cid}", "SK": "PROFILE", "entity_type": "CUSTOMER", "customer_id": cid,
                    "email": f"user{c}@example.com", "country": country, "created_at": created})
        for _ in range(rnd.choice([0, 1, 1, 2, 3, 5])):
            order_no += 1
            oid = f"O{order_no:06d}"
            at = f"2024-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}T{rnd.randint(0, 23):02d}:00:00"
            lines = []
            for ln in range(1, rnd.randint(1, 6) + 1):
                q = rnd.randint(1, 4)
                price = round(rnd.uniform(2, 120), 2)
                lines.append((ln, rnd.choice(SKUS), q, price))
            total = round(sum(q * p for _, _, q, p in lines), 2)
            out.append({"PK": f"CUSTOMER#{cid}", "SK": f"ORDER#{at}#{oid}", "GSI1PK": f"ORDER#{oid}",
                        "GSI1SK": f"ORDER#{oid}", "entity_type": "ORDER", "order_id": oid, "customer_id": cid,
                        "ordered_at": at, "status": rnd.choice(STATUS), "total": total})
            for ln, sku, q, p in lines:
                out.append({"PK": f"ORDER#{oid}", "SK": f"ITEM#{ln}", "GSI1PK": f"ORDER#{oid}",
                            "GSI1SK": f"ITEM#{ln}", "entity_type": "ITEM", "order_id": oid, "line_no": ln,
                            "sku": sku, "quantity": q, "price": p})
    return out


TABLE = {"TableName": "app-table",
         "KeySchema": [{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
         "AttributeDefinitions": [{"AttributeName": a, "AttributeType": "S"} for a in ("PK", "SK", "GSI1PK", "GSI1SK")],
         "GlobalSecondaryIndexes": [{"IndexName": "GSI1",
                                     "KeySchema": [{"AttributeName": "GSI1PK", "KeyType": "HASH"},
                                                   {"AttributeName": "GSI1SK", "KeyType": "RANGE"}],
                                     "Projection": {"ProjectionType": "ALL"}}],
         "BillingMode": "PAY_PER_REQUEST"}
