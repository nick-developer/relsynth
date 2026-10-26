# SPDX-License-Identifier: Apache-2.0
"""Fake data models with known ground truth.

Each scenario returns (tables, spec, truth).  `python scenarios.py` writes every
scenario to fakedata/out/<name>/ as CSV files plus spec.yaml, so the same data
can be run through the command-line tool exactly like a real database.

S1 mixed_table   one table, every column type, latent Gaussian copula, nulls
S2 star          fact(customer, product), Poisson block model, 4x3 segments
S3 snowflake     department -> category -> product, fact(customer, product)
S4 tree          self-referencing employees (multi-type Galton-Watson) + FK
S5 temporal      event fact with segment-specific seasonality
S6 text          support tickets with category-conditional free text
S7 enterprise    all of the above in one schema, with business rules
S8 reporting     regulatory reporting: versioned forms, a grain entity (one report per
                 reporter and period), facts nested in form filings through a composite
                 key, whole-form omissions, partial filings, open-row items, aggregates
S9 small_shop    a small operational shop (lookups, aggregates, grains, party tables)
S10 order_dw     a sales warehouse: order headers and lines, date offsets, calendar,
                 lookups through two hops, amount identities, running balance, products
                 that are never sold
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr
from scipy.stats import norm

OUT = Path(__file__).parent / "out"


def _syn(cfg=None):
    base = {"seed": 11, "gibbs": {"iterations": 100, "final_iterations": 25},
            "copula": {"iterations": 100, "burn_in": 50}}
    for k, v in (cfg or {}).items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return base


def _cat(rng, probs, levels):
    p = np.asarray(probs, dtype=float)
    return np.array(levels, dtype=object)[rng.choice(len(levels), p=p / p.sum(), size=None)]


def _cat_rows(rng, P, levels):
    """one draw per row, P is (n, V)"""
    c = P.cumsum(1)
    c /= c[:, -1:]
    idx = (rng.random(len(P))[:, None] > c).sum(1)
    return np.array(levels, dtype=object)[idx]


# ---------------------------------------------------------------- S1
def mixed_table(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    C = np.array([[1.0, 0.45, 0.10, 0.30, 0.00, 0.20, 0.50],
                  [0.45, 1.0, 0.20, 0.35, 0.00, 0.00, 0.40],
                  [0.10, 0.20, 1.0, 0.25, 0.00, 0.00, 0.10],
                  [0.30, 0.35, 0.25, 1.0, 0.00, 0.00, 0.30],
                  [0.00, 0.00, 0.00, 0.00, 1.0, 0.00, 0.00],
                  [0.20, 0.00, 0.00, 0.00, 0.00, 1.0, 0.15],
                  [0.50, 0.40, 0.10, 0.30, 0.00, 0.15, 1.0]])
    Z = rng.multivariate_normal(np.zeros(7), C, n)
    U = ndtr(Z)
    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": np.round(18 + 72 * np.asarray(pd.Series(U[:, 0]).pipe(lambda s: s ** 0.9))).astype(int),
        "bmi": np.round(np.exp(3.2 + 0.18 * Z[:, 1]), 1),
        "sex": np.where(U[:, 2] < 0.52, "F", "M"),
        "smoker": U[:, 3] > 0.78,
        "blood_type": np.select([U[:, 4] < .45, U[:, 4] < .85, U[:, 4] < .96], ["O", "A", "B"], "AB"),
        "visit_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.floor(U[:, 5] * 730), unit="D"),
        "cholesterol": np.where(U[:, 6] < 0.12, np.nan, np.round(150 + 40 * norm.ppf((U[:, 6] - .12) / .88 * .998 + .001), 0)),
        "email": [f"p{i}@mail{i % 7}.com" for i in range(n)],
    })
    spec = {"name": "S1_mixed_table", "source": {"kind": "csv", "path": "."},
            "tables": {"patient": {"primary_key": "patient_id", "columns": {
                "patient_id": {"type": "id"}, "age": {"type": "integer", "min": 18, "max": 90},
                "bmi": {"type": "float", "min": 10}, "sex": {"type": "categorical"},
                "smoker": {"type": "boolean"}, "blood_type": {"type": "categorical"},
                "visit_date": {"type": "date"}, "cholesterol": {"type": "float"},
                "email": {"type": "pii", "pii": {"generator": "email"}}}}},
            "relationships": [], "synthesis": _syn({"segments": {"default_cap": 1},
                                                    "copula": {"sample_rows": 4000, "iterations": 500, "burn_in": 100}})}
    return {"patient": df}, spec, {"C": C, "cols": ["age", "bmi", "sex", "smoker", "blood_type", "visit_date", "cholesterol"]}


# ---------------------------------------------------------------- S2
STAR_ETA = np.array([[0.30, 0.02, 0.01],
                     [0.02, 0.25, 0.03],
                     [0.01, 0.03, 0.20],
                     [0.10, 0.10, 0.10]])


def _customers(rng, n, K, pi, region_levels=("North", "South", "East", "West")):
    z = rng.choice(K, size=n, p=pi)
    age_mu = np.linspace(28, 62, K)
    inc_mu = np.linspace(10.0, 11.4, K)
    regP = rng.dirichlet(np.ones(len(region_levels)) * 2, size=K)
    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "age": np.clip(np.round(rng.normal(age_mu[z], 7)), 18, 95).astype(int),
        "income": np.round(np.exp(rng.normal(inc_mu[z], 0.35)), 2),
        "region": _cat_rows(rng, regP[z], region_levels),
    })
    return df, z


def _products(rng, n, L):
    z = rng.choice(L, size=n, p=np.full(L, 1 / L))
    price_mu = np.linspace(2.5, 5.5, L)
    catP = rng.dirichlet(np.ones(5) * 0.6, size=L)
    df = pd.DataFrame({
        "product_id": np.arange(1, n + 1),
        "price": np.round(np.exp(rng.normal(price_mu[z], 0.3)), 2),
        "category": _cat_rows(rng, catP[z], ["food", "home", "toys", "tech", "garden"]),
    })
    return df, z


def _sales_from_blocks(rng, zc, zp, eta, price, cust_ids, prod_ids, sale_id_start=1, frailty=None,
                       block_only=False):
    K, L = eta.shape
    rows_u, rows_v = [], []
    w = np.ones(len(zc)) if frailty is None else rng.gamma(frailty, 1 / frailty, len(zc))
    for k in range(K):
        U = np.flatnonzero(zc == k)
        for l in range(L):
            V = np.flatnonzero(zp == l)
            N = rng.poisson(eta[k, l] * w[U].sum() * len(V))
            if N == 0:
                continue
            rows_u.append(U[rng.choice(len(U), N, p=w[U] / w[U].sum())])
            rows_v.append(V[rng.integers(len(V), size=N)])
    u, v = np.concatenate(rows_u), np.concatenate(rows_v)
    k, l = zc[u], zp[v]
    qty = 1 + rng.poisson(0.5 + 0.8 * (k == 0) + 0.6 * (l == 2))
    if block_only:   # measure depends on the segment block only: data drawn exactly from the model
        amount = np.round(qty * np.exp(rng.normal(3.0 + 0.4 * k - 0.3 * l, 0.4)), 2)
    else:            # realistic: amount follows the product's own price
        amount = np.round(price[v] * qty * np.exp(rng.normal(0.1 * k - 0.05 * l, 0.1)), 2)
    chP = np.array([[0.7, 0.3], [0.3, 0.7], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    channel = np.where(rng.random(len(u)) < chP[(k + l) % 5, 0], "store", "online")
    df = pd.DataFrame({"sale_id": np.arange(sale_id_start, sale_id_start + len(u)),
                       "customer_id": cust_ids[u], "product_id": prod_ids[v],
                       "quantity": qty, "amount": amount, "channel": channel})
    return df.sample(frac=1, random_state=1).reset_index(drop=True), u, v


def star(seed=2):
    rng = np.random.default_rng(seed)
    cust, zc = _customers(rng, 1500, 4, [0.3, 0.3, 0.25, 0.15])
    prod, zp = _products(rng, 300, 3)
    eta = STAR_ETA / 3
    sales, _, _ = _sales_from_blocks(rng, zc, zp, eta, prod.price.to_numpy(),
                                     cust.customer_id.to_numpy(), prod.product_id.to_numpy(), block_only=True)
    spec = {"name": "S2_star", "source": {"kind": "csv", "path": "."},
            "tables": {
                "customer": {"primary_key": "customer_id", "columns": {
                    "customer_id": "id", "age": {"type": "integer", "min": 18},
                    "income": {"type": "float", "min": 0}, "region": "categorical"}},
                "product": {"primary_key": "product_id", "columns": {
                    "product_id": "id", "price": {"type": "float", "min": 0}, "category": "categorical"}},
                "sales": {"role": "fact", "primary_key": "sale_id", "fact": {"anchor": "customer"}, "columns": {
                    "sale_id": "id", "customer_id": "id", "product_id": "id",
                    "quantity": {"type": "integer", "min": 1}, "amount": {"type": "float", "min": 0},
                    "channel": "categorical"}}},
            "relationships": [
                {"child": "sales", "child_key": "customer_id", "parent": "customer"},
                {"child": "sales", "child_key": "product_id", "parent": "product"}],
            "synthesis": _syn({"segments": {"default_cap": 8}})}
    return {"customer": cust, "product": prod, "sales": sales}, spec, {"z": {"customer": zc, "product": zp}, "eta": eta}


# ---------------------------------------------------------------- S3
def snowflake(seed=3):
    rng = np.random.default_rng(seed)
    nd, nc, npr = 12, 60, 600
    zd = np.repeat([0, 1], nd // 2)
    dept = pd.DataFrame({"department_id": np.arange(1, nd + 1),
                         "division": np.where(zd == 0, "consumer", "industrial"),
                         "floor_space": np.round(np.where(zd == 0, 800, 2500) * np.exp(rng.normal(0, .2, nd)))})
    psi_dc = np.array([[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]])        # dept seg -> category seg
    dpar = rng.integers(nd, size=nc)
    zcg = np.array([rng.choice(3, p=psi_dc[zd[d]]) for d in dpar])
    cat = pd.DataFrame({"category_id": np.arange(1, nc + 1), "department_id": dept.department_id.to_numpy()[dpar],
                        "margin": np.round(np.array([0.35, 0.22, 0.12])[zcg] + rng.normal(0, 0.03, nc), 3)})
    psi_cp = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
    cpar = np.concatenate([np.arange(nc), rng.integers(nc, size=npr - nc)])   # every category has products
    zp = np.array([rng.choice(3, p=psi_cp[zcg[c]]) for c in cpar])
    prod = pd.DataFrame({"product_id": np.arange(1, npr + 1), "category_id": cat.category_id.to_numpy()[cpar],
                         "price": np.round(np.exp(rng.normal(np.array([2.3, 3.8, 5.6])[zp], 0.25)), 2)})
    cust, zc = _customers(rng, 1000, 3, [0.4, 0.35, 0.25])
    eta = np.array([[0.30, 0.05, 0.00], [0.08, 0.25, 0.04], [0.02, 0.06, 0.35]]) / 4
    sales, _, _ = _sales_from_blocks(rng, zc, zp, eta, prod.price.to_numpy(),
                                     cust.customer_id.to_numpy(), prod.product_id.to_numpy())
    spec = {"name": "S3_snowflake", "source": {"kind": "csv", "path": "."},
            "tables": {
                "department": {"primary_key": "department_id", "columns": {
                    "department_id": "id", "division": "categorical", "floor_space": {"type": "float", "min": 0}}},
                "category": {"primary_key": "category_id", "columns": {
                    "category_id": "id", "department_id": "id", "margin": "float"}},
                "product": {"primary_key": "product_id", "columns": {
                    "product_id": "id", "category_id": "id", "price": {"type": "float", "min": 0}}},
                "customer": {"primary_key": "customer_id", "columns": {
                    "customer_id": "id", "age": {"type": "integer", "min": 18},
                    "income": {"type": "float", "min": 0}, "region": "categorical"}},
                "sales": {"role": "fact", "primary_key": "sale_id", "fact": {"anchor": "customer"}, "columns": {
                    "sale_id": "id", "customer_id": "id", "product_id": "id",
                    "quantity": {"type": "integer", "min": 1}, "amount": {"type": "float", "min": 0},
                    "channel": "categorical"}}},
            "relationships": [
                {"child": "category", "child_key": "department_id", "parent": "department"},
                {"child": "product", "child_key": "category_id", "parent": "category"},
                {"child": "sales", "child_key": "customer_id", "parent": "customer"},
                {"child": "sales", "child_key": "product_id", "parent": "product"}],
            "synthesis": _syn({"segments": {"default_cap": 6, "min_rows_per_segment": 2}})}
    return ({"department": dept, "category": cat, "product": prod, "customer": cust, "sales": sales}, spec,
            {"z": {"department": zd, "category": zcg, "product": zp, "customer": zc}})


# ---------------------------------------------------------------- S4
TREE_LAM = np.array([4.0, 6.0, 0.05])
TREE_PSI = np.array([[0.10, 0.85, 0.05], [0.00, 0.15, 0.85], [0.0, 0.0, 1.0]])


def tree(seed=4, roots=12):
    rng = np.random.default_rng(seed)
    z, par = [0] * roots, [-1] * roots
    frontier = list(range(roots))
    while frontier:
        nxt = []
        for u in frontier:
            for _ in range(rng.poisson(TREE_LAM[z[u]])):
                z.append(rng.choice(3, p=TREE_PSI[z[u]]))
                par.append(u)
                nxt.append(len(z) - 1)
        frontier = nxt
    z, par = np.array(z), np.array(par)
    n = len(z)
    ndep = 20
    zdep = np.arange(ndep) % 2
    dept = pd.DataFrame({"department_id": np.arange(1, ndep + 1),
                         "function": np.where(zdep == 0, "operations", "corporate")})
    # execs and managers sit in corporate departments, staff mostly in operations
    pd_ = np.array([[0.1, 0.9], [0.3, 0.7], [0.9, 0.1]])
    dseg = np.array([rng.choice(2, p=pd_[k]) for k in z])
    did = np.array([rng.choice(np.flatnonzero(zdep == s)) for s in dseg]) + 1
    emp = pd.DataFrame({
        "employee_id": np.arange(1, n + 1),
        "manager_id": pd.array([p + 1 if p >= 0 else None for p in par], dtype="Int64"),
        "department_id": did,
        "salary": np.round(np.exp(rng.normal(np.array([12.2, 11.4, 10.6])[z], 0.15)), -2),
        "grade": np.array(["E", "M", "S"], dtype=object)[z],
        "hire_date": pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 5000, n) // (1 + (z == 0) * 2), unit="D"),
    })
    spec = {"name": "S4_tree", "source": {"kind": "csv", "path": "."},
            "tables": {
                "department": {"primary_key": "department_id", "columns": {"department_id": "id", "function": "categorical"}},
                "employee": {"primary_key": "employee_id", "columns": {
                    "employee_id": "id", "manager_id": "id", "department_id": "id",
                    "salary": {"type": "float", "min": 0}, "grade": "categorical", "hire_date": "date"}}},
            "relationships": [
                {"child": "employee", "child_key": "manager_id", "parent": "employee", "kind": "self_reference"},
                {"child": "employee", "child_key": "department_id", "parent": "department"}],
            "synthesis": _syn({"segments": {"default_cap": 6, "min_rows_per_segment": 2}})}
    return {"department": dept, "employee": emp}, spec, {"z": {"employee": z}, "lam": TREE_LAM, "psi": TREE_PSI, "roots": roots}


# ---------------------------------------------------------------- S5
def temporal(seed=5):
    rng = np.random.default_rng(seed)
    n = 800
    z = rng.choice(2, size=n, p=[0.55, 0.45])
    cust = pd.DataFrame({"customer_id": np.arange(1, n + 1),
                         "plan": np.where(rng.random(n) < np.where(z == 0, 0.8, 0.25), "basic", "premium"),
                         "tenure_years": np.round(rng.gamma(np.where(z == 0, 2, 5), 1.0), 1)})
    months = pd.period_range("2024-01", "2025-12", freq="M")
    m = np.array([p.month for p in months])
    prof = np.stack([1 + 0.9 * np.cos((m - 7) / 12 * 2 * np.pi),       # summer peak
                     1 + 0.9 * np.cos((m - 1) / 12 * 2 * np.pi)])      # winter peak
    prof /= prof.sum(1, keepdims=True)
    lam = np.array([25.0, 12.0])
    N = rng.poisson(lam[z])
    u = np.repeat(np.arange(n), N)
    b = np.array([rng.choice(len(months), p=prof[z[i]]) for i in u])
    starts = months.start_time.to_numpy()
    days = np.array([months[i].days_in_month for i in b])
    ts = starts[b] + (rng.random(len(b)) * days * 86400).astype("timedelta64[s]")
    ev = pd.DataFrame({"event_id": np.arange(1, len(u) + 1), "customer_id": u + 1,
                       "event_time": ts,
                       "duration_min": np.round(rng.gamma(2, np.where(z[u] == 0, 5, 15)), 1),
                       "kind": np.where(rng.random(len(u)) < np.where(z[u] == 0, .7, .2), "call", "visit")})
    spec = {"name": "S5_temporal", "source": {"kind": "csv", "path": "."},
            "tables": {
                "customer": {"primary_key": "customer_id", "columns": {
                    "customer_id": "id", "plan": "categorical", "tenure_years": {"type": "float", "min": 0}}},
                "event": {"role": "fact", "primary_key": "event_id",
                          "fact": {"time_column": "event_time", "time_bucket": "month"},
                          "columns": {"event_id": "id", "customer_id": "id", "event_time": "datetime",
                                      "duration_min": {"type": "float", "min": 0}, "kind": "categorical"}}},
            "relationships": [{"child": "event", "child_key": "customer_id", "parent": "customer"}],
            "synthesis": _syn({"segments": {"default_cap": 6}})}
    return {"customer": cust, "event": ev}, spec, {"z": {"customer": z}, "profile": prof, "lam": lam}


# ---------------------------------------------------------------- S6
GRAMMAR = {
    "billing": (["I was charged twice for my {plan} subscription this month.",
                 "My invoice shows the wrong amount, please refund the difference.",
                 "Why did my {plan} plan price go up? I did not agree to this.",
                 "Payment failed with my card ending {num4}, can you check the billing?"],),
    "technical": (["The app crashes when I open the {feature} page on {device}.",
                   "Since the last update the {feature} feature is very slow on my {device}.",
                   "I get an error code {code} when I try to sync my {device}.",
                   "The {feature} screen stays blank after login on {device}."],),
    "account": (["Please change the email on my account, I cannot log in.",
                 "I forgot my password and the reset link does not arrive.",
                 "How do I close my account and delete my data?",
                 "Can you add a second user to my {plan} account?"],),
    "shipping": (["My order has not arrived after {days} days.",
                  "The parcel was delivered to the wrong address, please help.",
                  "Tracking shows delivered but I have not received the package.",
                  "Can I change the delivery date of my order to next week?"],),
}
SLOTS = {"feature": ["reports", "dashboard", "export", "calendar", "search"],
         "device": ["iPhone", "Android phone", "laptop", "tablet"],
         "plan": ["basic", "pro"], "days": ["5", "7", "10", "14"],
         "code": ["E101", "E205", "E330"]}
RARE_NAMES = []   # filled with the unique names planted in tickets


def _ticket_text(rng, cat, plan):
    t = rng.choice(GRAMMAR[cat][0])
    fills = {k: rng.choice(v) for k, v in SLOTS.items()}
    fills["plan"] = plan
    fills["num4"] = str(rng.integers(1000, 9999))
    s = t.format(**fills)
    if rng.random() < 0.3:
        s += " " + rng.choice(["Thanks.", "Please help.", "This is urgent.", "Thank you for your help."])
    if rng.random() < 0.01:                                   # a rare, identifying detail (unique per row)
        name = "".join(rng.choice(list("bcdfgklmnprstvz")) + rng.choice(list("aeiou")) for _ in range(3)).title()
        RARE_NAMES.append(name)
        s += f" Regards, {name} from flat {rng.integers(100, 999)}."
    return s


def text(seed=6):
    rng = np.random.default_rng(seed)
    n = 600
    z = rng.choice(2, size=n, p=[0.6, 0.4])
    cust = pd.DataFrame({"customer_id": np.arange(1, n + 1), "plan": np.where(z == 0, "basic", "pro"),
                         "age": np.clip(np.round(rng.normal(np.where(z == 0, 35, 48), 9)), 18, 90).astype(int)})
    N = rng.poisson(np.where(z == 0, 4, 7))
    u = np.repeat(np.arange(n), N)
    catP = np.array([[0.4, 0.2, 0.2, 0.2], [0.15, 0.45, 0.25, 0.15]])
    cats = np.array(list(GRAMMAR))
    c = np.array([cats[rng.choice(4, p=catP[z[i]])] for i in u])
    tk = pd.DataFrame({"ticket_id": np.arange(1, len(u) + 1), "customer_id": u + 1, "category": c,
                       "priority": np.where(c == "technical", "high", np.where(rng.random(len(u)) < .3, "high", "normal")),
                       "body": [_ticket_text(rng, ci, cust.plan[i]) for ci, i in zip(c, u)]})
    spec = {"name": "S6_text", "source": {"kind": "csv", "path": "."},
            "tables": {
                "customer": {"primary_key": "customer_id", "columns": {
                    "customer_id": "id", "plan": "categorical", "age": {"type": "integer", "min": 18}}},
                "ticket": {"role": "fact", "primary_key": "ticket_id", "columns": {
                    "ticket_id": "id", "customer_id": "id", "category": "categorical", "priority": "categorical",
                    "body": {"type": "text", "text": {"condition_on": "category"}}}}},
            "relationships": [{"child": "ticket", "child_key": "customer_id", "parent": "customer"}],
            "synthesis": _syn({"segments": {"default_cap": 4}})}
    return {"customer": cust, "ticket": tk}, spec, {"z": {"customer": z}, "rare_names": list(RARE_NAMES)}


# ---------------------------------------------------------------- S7
def enterprise(seed=7):
    rng = np.random.default_rng(seed)
    s3, sp3, t3 = snowflake(seed)
    region = pd.DataFrame({"region_id": np.arange(1, 9), "country": ["GR", "GR", "DE", "DE", "FR", "FR", "IT", "ES"],
                           "urban": [True, False, True, False, True, False, True, True]})
    cust = s3["customer"].drop(columns=["region"]).copy()
    zc = t3["z"]["customer"]
    regP = np.array([[.3, .05, .2, .05, .2, .05, .1, .05], [.05, .3, .05, .3, .05, .15, .05, .05],
                     [.15, .1, .15, .1, .15, .1, .15, .1]])
    cust["region_id"] = np.array([rng.choice(8, p=regP[k]) for k in zc]) + 1
    cust["email"] = [f"c{i}@example.org" for i in range(len(cust))]
    cust.loc[rng.random(len(cust)) < 0.05, "region_id"] = None
    cust["region_id"] = cust["region_id"].astype("Int64")
    store = pd.DataFrame({"store_id": np.arange(1, 31), "region_id": rng.integers(1, 9, 30),
                          "size_m2": np.round(rng.gamma(4, 300, 30))})
    sales = s3["sales"].rename(columns={"amount": "unit_price"}).copy()
    sales["unit_price"] = np.round(sales["unit_price"] / sales["quantity"], 2)
    sales["discount"] = np.round(np.where(rng.random(len(sales)) < 0.2, rng.uniform(0, 0.3, len(sales)), 0)
                                 * sales.unit_price * sales.quantity, 2)
    sales["total"] = np.round(sales.quantity * sales.unit_price - sales.discount, 2)
    cust_region = cust.set_index("customer_id").region_id
    st_by_reg = store.groupby("region_id").store_id.apply(list).to_dict()
    def pick_store(cid):
        r = cust_region.get(cid)
        opts = st_by_reg.get(r) if pd.notna(r) else None
        return rng.choice(opts) if opts and rng.random() < 0.8 else rng.integers(1, 31)
    sales["store_id"] = [pick_store(c) for c in sales.customer_id]
    months = pd.period_range("2024-01", "2025-06", freq="M")
    sales["sale_date"] = months.start_time.to_numpy()[rng.integers(len(months), size=len(sales))] + \
        pd.to_timedelta(rng.integers(0, 28, len(sales)), unit="D")
    s6, _, _ = text(seed)
    tk = s6["ticket"].copy()
    tk["customer_id"] = cust.customer_id.to_numpy()[rng.integers(len(cust), size=len(tk))]
    s4, _, _ = tree(seed, roots=4)
    emp = s4["employee"].drop(columns=["department_id"]).copy()
    emp["store_id"] = rng.integers(1, 31, len(emp))
    tables = {"region": region, "customer": cust, "store": store, **{k: s3[k] for k in ("department", "category", "product")},
              "sales": sales[["sale_id", "customer_id", "product_id", "store_id", "sale_date", "quantity",
                              "unit_price", "discount", "total", "channel"]],
              "ticket": tk, "employee": emp}
    spec = {"name": "S7_enterprise", "source": {"kind": "csv", "path": "."},
            "tables": {
                "region": {"primary_key": "region_id", "columns": {"region_id": "id", "country": "categorical", "urban": "boolean"}},
                "customer": {"primary_key": "customer_id", "columns": {
                    "customer_id": "id", "age": {"type": "integer", "min": 18}, "income": {"type": "float", "min": 0},
                    "region_id": "id", "email": {"type": "pii", "pii": {"generator": "email"}}}},
                "store": {"primary_key": "store_id", "columns": {"store_id": "id", "region_id": "id", "size_m2": "float"}},
                **{k: sp3["tables"][k] for k in ("department", "category", "product")},
                "sales": {"role": "fact", "primary_key": "sale_id",
                          "fact": {"anchor": "customer", "partner": "product", "time_column": "sale_date", "time_bucket": "month"},
                          "columns": {"sale_id": "id", "customer_id": "id", "product_id": "id", "store_id": "id",
                                      "sale_date": "date", "quantity": {"type": "integer", "min": 1},
                                      "unit_price": {"type": "float", "min": 0}, "discount": {"type": "float", "min": 0},
                                      "total": "float", "channel": "categorical"}},
                "ticket": {"role": "fact", "primary_key": "ticket_id", "columns": {
                    "ticket_id": "id", "customer_id": "id", "category": "categorical", "priority": "categorical",
                    "body": {"type": "text", "text": {"condition_on": "category"}}}},
                "employee": {"primary_key": "employee_id", "columns": {
                    "employee_id": "id", "manager_id": "id", "store_id": "id", "salary": {"type": "float", "min": 0},
                    "grade": "categorical", "hire_date": "date"}}},
            "relationships": [
                {"child": "customer", "child_key": "region_id", "parent": "region"},
                {"child": "store", "child_key": "region_id", "parent": "region"},
                {"child": "category", "child_key": "department_id", "parent": "department"},
                {"child": "product", "child_key": "category_id", "parent": "category"},
                {"child": "sales", "child_key": "customer_id", "parent": "customer"},
                {"child": "sales", "child_key": "product_id", "parent": "product"},
                {"child": "sales", "child_key": "store_id", "parent": "store"},
                {"child": "ticket", "child_key": "customer_id", "parent": "customer"},
                {"child": "employee", "child_key": "manager_id", "parent": "employee", "kind": "self_reference"},
                {"child": "employee", "child_key": "store_id", "parent": "store"}],
            "rules": [
                {"table": "sales", "type": "derive", "expr": "total = quantity * unit_price - discount"},
                {"table": "sales", "type": "check", "expr": "discount <= quantity * unit_price"},
                {"table": "customer", "type": "check", "expr": "age >= 18"}],
            "synthesis": _syn({"segments": {"default_cap": 6, "min_rows_per_segment": 2}})}
    return tables, spec, {}


# ---------------------------------------------------------------- S8
def reporting(seed=8, n_reporters=60, p_report=0.92):
    """Generic regulatory reporting model with known ground truth.

    version (copied)   two versions of the reporting framework, each valid for 4 periods
    period (copied)    8 quarters
    item (copied)      form items of both versions: sheet code (form code + sheet suffix),
                       value type, open-row flag.  Items of v1 and v2 are different sets,
                       but the item -> version link is NOT declared: it must be learned.
    code (copied)      a small code list used by 'code' items
    reporter           synthesised: type, country, a unique name (identifier)
    report             one per reporter and period (present with prob. p_report);
                       version determined by the period; lag, status, a rare resubmission flag
    form_filing        every report files every form of its version once (5 forms)
    item_value         nested in form_filing through (report_id, item.form_code); per filing:
                       whole-form omission with prob. 0.05, else complete (presence 0.80) or
                       partial (prob. 0.15, presence 0.25); open-row items repeat 1 + Poisson(3)
    """
    rng = np.random.default_rng(seed)
    version = pd.DataFrame({"version_id": [1, 2], "version_code": ["v1", "v2"],
                            "valid_from": [20240331, 20250331], "valid_to": [20241231, 20251231]})
    pend = ["20240331", "20240630", "20240930", "20241231", "20250331", "20250630", "20250930", "20251231"]
    period = pd.DataFrame({"period_id": [int(x) for x in pend], "period_label": [f"{x[:4]}-Q{i % 4 + 1}" for i, x in enumerate(pend)]})
    forms = ["F01", "F02", "F03", "F04", "F05"]
    sheets = {"F01": ["a"], "F02": ["a", "b"], "F03": ["a"], "F04": ["a", "b"], "F05": ["a"]}
    n_items = {"F01": 3, "F02": 30, "F03": 12, "F04": 40, "F05": 4}
    items = []
    iid = 100
    for v in (1, 2):
        for fcode in forms:
            for j in range(n_items[fcode]):
                sh = sheets[fcode][j % len(sheets[fcode])]
                vt = "amount" if j % 5 < 3 else ("flag" if j % 5 == 3 else "code")
                openrow = int(fcode == "F05")
                items.append({"item_id": iid, "sheet_code": f"{fcode}.{sh}", "item_code": f"v{v}-{fcode}-{j:03d}",
                              "value_type": vt, "open_row": openrow, "ver_": v, "form_": fcode,
                              "mu_": rng.normal(10 + j % 7, 0.5)})
                iid += 1
    item = pd.DataFrame(items)
    code = pd.DataFrame({"code_id": np.arange(1, 9), "code_value": [f"K{i}" for i in range(1, 9)]})
    rtype = rng.choice(["bank", "insurer", "fund"], n_reporters, p=[.5, .3, .2])
    reporter = pd.DataFrame({"reporter_id": np.arange(1, n_reporters + 1),
                             "reporter_name": [f"Reporter {i:03d} plc" for i in range(n_reporters)],
                             "reporter_type": rtype,
                             "country": rng.choice(["GR", "DE", "FR", "IT"], n_reporters)})
    rows = []
    rid = 1
    for r in reporter.itertuples():
        for per in period.period_id:
            if rng.random() > p_report:
                continue
            v = 1 if per <= 20241231 else 2
            lag = int(rng.normal({"bank": 30, "insurer": 40, "fund": 25}[r.reporter_type], 5))
            rows.append({"report_id": rid, "reporter_id": r.reporter_id, "period_id": per, "version_id": v,
                         "lag_days": lag, "resubmitted": False})
            rid += 1
    report = pd.DataFrame(rows)
    rs = rng.choice(len(report), 3, replace=False)          # a rare flag: 3 rows
    report.loc[rs, "resubmitted"] = True
    report["status"] = np.where(rng.random(len(report)) < 0.9, "accepted", "accepted with warnings")
    filing_rows, value_rows = [], []
    kinds = []
    for rep in report.itertuples():
        its = item[item.ver_ == rep.version_id]
        for fcode in forms:
            u = rng.random()
            kind = "omitted" if u < 0.05 else ("partial" if u < 0.20 else "complete")
            kinds.append(kind)
            fit = its[its.form_ == fcode]
            pres = {"omitted": 0.0, "partial": 0.25, "complete": 0.80}[kind]
            n = 0
            for it in fit.itertuples():
                if fcode == "F01":
                    present = kind != "omitted"          # header items: always there when the form is filed
                else:
                    present = rng.random() < pres
                if not present:
                    continue
                m = 1 + rng.poisson(3) if it.open_row else 1
                for _ in range(m):
                    amt = float(np.round(np.exp(rng.normal(it.mu_, 0.4)), 2)) if it.value_type == "amount" else None
                    flag = bool(rng.random() < 0.3) if it.value_type == "flag" else None
                    cd = int(rng.integers(1, 9)) if it.value_type == "code" else None
                    value_rows.append({"report_id": rep.report_id, "item_id": it.item_id, "amount": amt,
                                       "flag": flag, "code_id": cd})
                    n += 1
            filing_rows.append({"report_id": rep.report_id, "form_code": fcode, "item_count": n,
                                "filing_status": "filed" if n > 0 else "declared not filed"})
    filing = pd.DataFrame(filing_rows)
    value = pd.DataFrame(value_rows)
    value["code_id"] = value["code_id"].astype("Int64")
    item_out = item[["item_id", "sheet_code", "item_code", "value_type", "open_row"]].copy()
    tables = {"version": version, "period": period, "item": item_out, "code": code, "reporter": reporter,
              "report": report[["report_id", "reporter_id", "period_id", "version_id", "lag_days", "status", "resubmitted"]],
              "form_filing": filing, "item_value": value}
    spec = {"name": "S8_reporting", "source": {"kind": "csv", "path": "."},
            "tables": {
                "version": {"mode": "copy", "primary_key": "version_id", "columns": {
                    "version_id": "id", "version_code": "categorical", "valid_from": "integer", "valid_to": "integer"}},
                "period": {"mode": "copy", "primary_key": "period_id", "columns": {"period_id": "id", "period_label": "categorical"}},
                "item": {"mode": "copy", "primary_key": "item_id", "columns": {
                    "item_id": "id", "sheet_code": "categorical", "item_code": "categorical", "value_type": "categorical",
                    "open_row": "integer", "form_code": "categorical"}},
                "code": {"mode": "copy", "primary_key": "code_id", "columns": {"code_id": "id", "code_value": "categorical"}},
                "reporter": {"primary_key": "reporter_id", "columns": {
                    "reporter_id": "id", "reporter_name": {"type": "pii", "pii": {"generator": "pattern", "pattern": "Reporter 999 plc"}},
                    "reporter_type": "categorical", "country": "categorical"}},
                "report": {"primary_key": "report_id", "unique": [["reporter_id", "period_id"]], "columns": {
                    "report_id": "id", "reporter_id": "id", "period_id": "id", "version_id": "id",
                    "lag_days": "integer", "status": "categorical", "resubmitted": "boolean"}},
                "form_filing": {"role": "fact", "fact": {"anchor": "report"}, "unique": [["report_id", "form_code"]],
                                "columns": {"report_id": "id", "form_code": "categorical", "item_count": {"type": "integer", "min": 0},
                                            "filing_status": "categorical"}},
                "item_value": {"role": "fact", "fact": {"anchor": "form_filing", "partner": "item"},
                               "unique": [{"columns": ["report_id", "item_id"], "where": "item_id.open_row == 0"}],
                               "columns": {"report_id": "id", "item_id": "id", "code_id": "id", "amount": "float",
                                           "flag": "boolean"}}},
            "relationships": [
                {"child": "report", "child_key": "reporter_id", "parent": "reporter"},
                {"child": "report", "child_key": "period_id", "parent": "period"},
                {"child": "report", "child_key": "version_id", "parent": "version"},
                {"child": "form_filing", "child_key": "report_id", "parent": "report"},
                {"child": "item_value", "child_key": "report_id", "parent": "report"},
                {"child": "item_value", "child_key": "item_id", "parent": "item"},
                {"child": "item_value", "child_key": "code_id", "parent": "code"},
                {"child": "item_value", "child_key": ["report_id", "item_id.form_code"], "parent": "form_filing",
                 "parent_key": ["report_id", "form_code"]}],
            "rules": [
                {"table": "item", "type": "derive", "expr": 'form_code = regex_replace(sheet_code, "\\.[a-z]$", "")'},
                {"table": "report", "type": "check", "expr": "between(period_id, version_id.valid_from, version_id.valid_to)"},
                {"table": "form_filing", "type": "aggregate", "column": "item_count", "agg": "count", "from": "item_value"},
                {"table": "form_filing", "type": "derive",
                 "expr": "filing_status = where(item_count > 0, 'filed', 'declared not filed')"}],
            "synthesis": _syn()}
    truth = {"p_report": p_report, "omit": 0.05, "partial": 0.15, "pres_complete": 0.80, "pres_partial": 0.25,
             "open_mult_mean": 4.0, "item_version": dict(zip(item.item_id, item.ver_)),
             "kinds": kinds, "forms": forms}
    return tables, spec, truth


# ---------------------------------------------------------------- S9
def small_shop(seed=9, n_cust=200, inv_per_cust=5):
    """A small operational shop database with known ground truth (the shape of
    many sample databases).

    staff (8 rows)      a tiny hierarchy: 1 general manager -> sales manager -> 3
                        agents, general manager -> IT manager -> 2 IT staff; only
                        agents serve customers
    customer            city -> state -> country (states only in two countries),
                        an independent 'tier', served by an agent
    invoice             exactly inv_per_cust per customer
    invoice_line        lines per invoice exactly in {1, 2, 4, 6, 9}; unit_price is
                        the track's price (a lookup), quantity 1; invoice.total is
                        sum(unit_price * quantity) (an aggregate)
    genre (copied, 30)  a code list larger than stratify_max_levels
    album               keys and a title only (no modelled column)
    track               album, genre, length, price, composer (long-tail labels)
    playlist (copied)   12 playlists, each holding tracks of its own 2-3 genres (one empty)
    playlist_track      grain (playlist, track)
    """
    rng = np.random.default_rng(seed)
    titles = ["General Manager", "Sales Manager", "Sales Agent", "Sales Agent", "Sales Agent",
              "IT Manager", "IT Staff", "IT Staff"]
    staff = pd.DataFrame({"staff_id": np.arange(1, 9), "title": titles,
                          "reports_to": pd.array([None, 1, 2, 2, 2, 1, 6, 6], dtype="Int64"),
                          "office": ["HQ", "HQ", "Branch", "Branch", "Branch", "HQ", "Lab", "Lab"],
                          "hire_year": [2002, 2003, 2010, 2011, 2012, 2004, 2013, 2015]})
    geo = []
    for country, states in [("US", 4), ("CA", 2), ("FR", 0), ("DE", 0)]:
        for st in range(max(states, 1)):
            for c in range(3 if states else 4):
                geo.append((f"{country}-city{st}{c}", f"{country}-S{st}" if states else None, country))
    geo = pd.DataFrame(geo, columns=["city", "state", "country"])
    gi = rng.integers(0, len(geo), n_cust)
    agents = np.array([3, 4, 5])
    cust = pd.DataFrame({"customer_id": np.arange(1, n_cust + 1), "city": geo.city.to_numpy()[gi],
                         "state": geo.state.to_numpy()[gi], "country": geo.country.to_numpy()[gi],
                         "tier": rng.choice(["gold", "silver", "bronze"], n_cust, p=[0.2, 0.3, 0.5]),
                         "rep_id": rng.choice(agents, n_cust)})
    genre = pd.DataFrame({"genre_id": np.arange(1, 31), "genre": [f"genre{g:02d}" for g in range(1, 31)]})
    n_album = 60
    album = pd.DataFrame({"album_id": np.arange(1, n_album + 1),
                          "album_title": [f"{rng.choice(['Live', 'Best of', 'Songs of', 'Nights in'])} "
                                          f"{rng.choice(['Paris', 'Rome', 'the Sea', 'Spring', 'Fire', 'Gold'])} {a}"
                                          for a in range(n_album)]})
    alb_genre = rng.integers(1, 31, n_album)
    ta = rng.integers(1, n_album + 1, 900)
    first = ["Ann", "Bob", "Cid", "Dee", "Eve", "Fay", "Gus", "Hal", "Ian", "Joy"]
    last = ["Ray", "Moss", "Lane", "Hart", "Cole", "Wade", "Snow", "Reed", "Page", "Vale", "Park", "Holt"]
    comp_pool = [f"{a} {b}" for a in first for b in last]
    comp = [None if rng.random() < 0.2 else
            " / ".join(rng.choice(comp_pool, size=rng.integers(1, 3), replace=False)) for _ in range(900)]
    track = pd.DataFrame({"track_id": np.arange(1, 901), "album_id": ta, "genre_id": alb_genre[ta - 1],
                          "milliseconds": np.round(np.exp(rng.normal(12.3, 0.35, 900))).astype(int),
                          "unit_price": np.where(rng.random(900) < 0.1, 1.99, 0.99), "composer": comp})
    playlist = pd.DataFrame({"playlist_id": np.arange(1, 13), "playlist": [f"list{p}" for p in range(1, 13)]})
    pt = []
    for p in range(1, 12):                  # playlist 12 stays empty
        gs = rng.choice(np.arange(1, 31), size=rng.integers(2, 4), replace=False)
        tr = track.track_id[track.genre_id.isin(gs)].to_numpy()
        pt += [(p, t) for t in tr[rng.random(len(tr)) < 0.6]]
    playlist_track = pd.DataFrame(pt, columns=["playlist_id", "track_id"])
    inv_cust = np.repeat(cust.customer_id.to_numpy(), inv_per_cust)
    invoice = pd.DataFrame({"invoice_id": np.arange(1, len(inv_cust) + 1), "customer_id": inv_cust,
                            "billing_country": cust.set_index("customer_id").country.loc[inv_cust].to_numpy()})
    nl = rng.choice([1, 2, 4, 6, 9], size=len(invoice), p=[0.15, 0.3, 0.2, 0.2, 0.15])
    li = np.repeat(invoice.invoice_id.to_numpy(), nl)
    lt = rng.integers(1, 901, len(li))
    line = pd.DataFrame({"line_id": np.arange(1, len(li) + 1), "invoice_id": li, "track_id": lt,
                         "unit_price": track.unit_price.to_numpy()[lt - 1], "quantity": 1})
    invoice["total"] = np.round(invoice.invoice_id.map(line.groupby("invoice_id").unit_price.sum()).to_numpy(), 2)
    tables = {"staff": staff, "customer": cust, "invoice": invoice, "invoice_line": line, "genre": genre,
              "album": album, "track": track, "playlist": playlist, "playlist_track": playlist_track}
    spec = {"name": "S9_small_shop", "source": {"kind": "csv", "path": "."},
            "tables": {
                "staff": {"primary_key": "staff_id", "columns": {
                    "staff_id": "id", "title": "categorical", "reports_to": "id", "office": "categorical",
                    "hire_year": "integer"}},
                "customer": {"primary_key": "customer_id", "columns": {
                    "customer_id": "id", "city": "categorical", "state": "categorical", "country": "categorical",
                    "tier": "categorical", "rep_id": "id"}},
                "invoice": {"primary_key": "invoice_id", "columns": {
                    "invoice_id": "id", "customer_id": "id", "billing_country": {"type": "lookup", "lookup": "customer_id.country"},
                    "total": "float"}},
                "invoice_line": {"role": "fact", "primary_key": "line_id", "fact": {"anchor": "invoice"}, "columns": {
                    "line_id": "id", "invoice_id": "id", "track_id": "id",
                    "unit_price": {"type": "lookup", "lookup": "track_id.unit_price"}, "quantity": "integer"}},
                "genre": {"primary_key": "genre_id", "mode": "copy", "columns": {"genre_id": "id", "genre": "categorical"}},
                "album": {"primary_key": "album_id", "columns": {"album_id": "id", "album_title": "text"}},
                "track": {"primary_key": "track_id", "columns": {
                    "track_id": "id", "album_id": "id", "genre_id": "id", "milliseconds": "integer",
                    "unit_price": "float", "composer": "text"}},
                "playlist": {"primary_key": "playlist_id", "mode": "copy", "columns": {"playlist_id": "id", "playlist": "categorical"}},
                "playlist_track": {"role": "fact", "fact": {"anchor": "playlist"}, "unique": [["playlist_id", "track_id"]],
                                   "columns": {"playlist_id": "id", "track_id": "id"}}},
            "relationships": [
                {"child": "staff", "child_key": "reports_to", "parent": "staff", "kind": "self_reference"},
                {"child": "customer", "child_key": "rep_id", "parent": "staff"},
                {"child": "invoice", "child_key": "customer_id", "parent": "customer"},
                {"child": "invoice_line", "child_key": "invoice_id", "parent": "invoice"},
                {"child": "invoice_line", "child_key": "track_id", "parent": "track"},
                {"child": "track", "child_key": "album_id", "parent": "album"},
                {"child": "track", "child_key": "genre_id", "parent": "genre"},
                {"child": "playlist_track", "child_key": "playlist_id", "parent": "playlist"},
                {"child": "playlist_track", "child_key": "track_id", "parent": "track"}],
            "rules": [{"table": "invoice", "type": "aggregate", "column": "total", "agg": "sum",
                       "from": "invoice_line", "of": "unit_price * quantity"}],
            "synthesis": _syn()}
    truth = {"agents": set(agents.tolist()), "geo": geo, "inv_per_cust": inv_per_cust, "lines": {1, 2, 4, 6, 9}}
    return tables, spec, truth


# ---------------------------------------------------------------- S10
def order_dw(seed=10, n_cust=2000, n_prod=150, days=730):
    """A sales data warehouse with the structures of AdventureWorks DW, generated from
    a known model.

    dim_date (copied)   one row per day, date_key yyyymmdd, full_date, month, year
    dim_territory (5), dim_geography (60, copied code list), dim_customer (segment A/B/C
                        drives who buys, how often and which categories)
    dim_product         category, list_price, standard_cost = 0.55 * list_price (4 dp),
                        status; obsolete products (25%) are never sold
    dim_promotion (5)   discount_pct
    fact_sales          order lines: order_number (degenerate dimension) determines the
                        header (customer, territory, dates); line_number 1..n;
                        ship = order + 7 days (8 on 3% of orders), due = order + 12;
                        territory = the customer's geography's territory; order_date =
                        the calendar date of order_date_key; unit_price and
                        standard_cost are the product's; amounts are identities rounded
                        to 4 dp (extended = qty * price, discount = extended * pct,
                        sales = extended - discount, tax = 0.08 * sales, cost = qty * cost)
    fact_inventory      weekly stock per current product: balance = previous balance +
                        units_in - units_out (a running balance from 0)
    dim_customer.first_purchase = min(order_date) of the customer's orders (null if none)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    dim_date = pd.DataFrame({"date_key": dates.strftime("%Y%m%d").astype(int), "full_date": dates.strftime("%Y-%m-%d"),
                             "month_name": dates.strftime("%B"), "calendar_year": dates.year})
    terr = pd.DataFrame({"territory_key": np.arange(1, 6),
                         "territory_name": ["North", "South", "East", "West", "Central"]})
    geo = pd.DataFrame({"geography_key": np.arange(1, 61), "city": [f"City {g:02d}" for g in range(1, 61)],
                        "territory_key": rng.integers(1, 6, 60)})
    seg = rng.choice(["A", "B", "C"], n_cust, p=[0.25, 0.35, 0.40])
    inc_mu = {"A": 11.6, "B": 11.0, "C": 10.5}
    cust = pd.DataFrame({"customer_key": np.arange(1, n_cust + 1), "geography_key": rng.integers(1, 61, n_cust),
                         "segment": seg,
                         "yearly_income": np.round(np.exp([rng.normal(inc_mu[x], 0.3) for x in seg]), -3)})
    cats = ["Bikes", "Clothing", "Accessories", "Components"]
    pcat = rng.choice(cats, n_prod, p=[0.3, 0.25, 0.25, 0.2])
    base = {"Bikes": 7.0, "Clothing": 3.5, "Accessories": 3.0, "Components": 5.0}
    lp = np.round(np.exp([rng.normal(base[c], 0.4) for c in pcat]), 2)
    prod = pd.DataFrame({"product_key": np.arange(1, n_prod + 1), "category": pcat, "list_price": lp,
                         "standard_cost": np.round(0.55 * lp, 4),
                         "status": np.where(rng.random(n_prod) < 0.25, "obsolete", "current")})
    promo = pd.DataFrame({"promotion_key": np.arange(1, 6), "discount_pct": [0.0, 0.02, 0.05, 0.10, 0.15]})
    # orders: who buys, how many orders, which categories (by segment)
    p_buy = {"A": 0.95, "B": 0.7, "C": 0.35}
    rate = {"A": 3.0, "B": 1.2, "C": 0.4}
    pref = {"A": [0.6, 0.1, 0.15, 0.15], "B": [0.25, 0.3, 0.3, 0.15], "C": [0.05, 0.45, 0.45, 0.05]}
    cur = prod[prod.status == "current"]
    by_cat = {c: cur.product_key[cur.category == c].to_numpy() for c in cats}
    lines, orders = [], []
    on = 50000
    for ck, sg, gk in zip(cust.customer_key, cust.segment, cust.geography_key):
        if rng.random() >= p_buy[sg]:
            continue
        for _ in range(1 + rng.poisson(rate[sg])):
            on += 1
            od = int(rng.integers(0, days - 20))
            ship = od + (8 if rng.random() < 0.03 else 7)
            orders.append((f"SO{on}", ck, od, ship, od + 12))
            nl = min(1 + rng.poisson(0.8), 8)
            for ln in range(1, nl + 1):
                c = cats[rng.choice(4, p=pref[sg])]
                pk = int(rng.choice(by_cat[c]))
                lines.append((f"SO{on}", ln, pk, int(rng.choice(5, p=[0.7, 0.1, 0.1, 0.06, 0.04])) + 1,
                              int(rng.choice([1, 1, 1, 2, 2, 3, 4]))))
    o = pd.DataFrame(orders, columns=["order_number", "customer_key", "od", "sd", "dd"])
    li = pd.DataFrame(lines, columns=["order_number", "line_number", "product_key", "promotion_key", "order_quantity"])
    f = li.merge(o, on="order_number", how="left")
    dk = dim_date.date_key.to_numpy()
    f["order_date_key"], f["ship_date_key"], f["due_date_key"] = dk[f.od], dk[f.sd], dk[f.dd]
    f["order_date"] = dim_date.full_date.to_numpy()[f.od]
    gt = geo.set_index("geography_key").territory_key
    f["territory_key"] = gt.loc[cust.set_index("customer_key").geography_key.loc[f.customer_key]].to_numpy()
    pp = prod.set_index("product_key")
    f["unit_price"] = pp.list_price.loc[f.product_key].to_numpy()
    f["standard_cost"] = pp.standard_cost.loc[f.product_key].to_numpy()
    pct = promo.set_index("promotion_key").discount_pct.loc[f.promotion_key].to_numpy()
    f["extended_amount"] = np.round(f.order_quantity * f.unit_price, 4)
    f["discount_amount"] = np.round(f.extended_amount * pct, 4)
    f["sales_amount"] = np.round(f.extended_amount - f.discount_amount, 4)
    f["tax_amt"] = np.round(0.08 * f.sales_amount, 4)
    f["total_product_cost"] = np.round(f.order_quantity * f.standard_cost, 4)
    f["discount_pct"] = pct
    fact_sales = f[["order_number", "line_number", "customer_key", "territory_key", "order_date_key", "ship_date_key",
                    "due_date_key", "order_date", "product_key", "promotion_key", "order_quantity", "unit_price",
                    "discount_pct", "standard_cost", "extended_amount", "discount_amount", "sales_amount", "tax_amt",
                    "total_product_cost"]].reset_index(drop=True)
    first = fact_sales.groupby("customer_key").order_date.min()
    cust["first_purchase"] = cust.customer_key.map(first)
    # weekly inventory of current products: a running balance
    weeks = np.arange(0, 26 * 7, 7)
    inv = []
    for pk in cur.product_key:
        bal = 0
        lam = rng.uniform(3, 12)
        for w in weeks:
            ui = int(rng.poisson(lam))
            uo = int(min(rng.poisson(lam * 0.9), bal + ui))
            bal += ui - uo
            inv.append((int(pk), int(dk[w]), ui, uo, bal))
    fact_inventory = pd.DataFrame(inv, columns=["product_key", "date_key", "units_in", "units_out", "units_balance"])
    tables = {"dim_date": dim_date, "dim_territory": terr, "dim_geography": geo, "dim_customer": cust,
              "dim_product": prod, "dim_promotion": promo, "fact_sales": fact_sales, "fact_inventory": fact_inventory}
    look = {"type": "lookup"}
    spec = {"name": "S10_order_dw", "source": {"kind": "csv", "path": "."},
            "tables": {
                "dim_date": {"primary_key": "date_key", "mode": "copy", "columns": {
                    "date_key": "id", "full_date": "date", "month_name": "categorical", "calendar_year": "integer"}},
                "dim_territory": {"primary_key": "territory_key", "mode": "copy", "columns": {
                    "territory_key": "id", "territory_name": "categorical"}},
                "dim_geography": {"primary_key": "geography_key", "mode": "copy", "columns": {
                    "geography_key": "id", "city": "categorical", "territory_key": "id"}},
                "dim_customer": {"primary_key": "customer_key", "columns": {
                    "customer_key": "id", "geography_key": "id", "segment": "categorical", "yearly_income": "float",
                    "first_purchase": "date"}},
                "dim_product": {"primary_key": "product_key", "columns": {
                    "product_key": "id", "category": "categorical", "list_price": "float", "standard_cost": "float",
                    "status": "categorical"}},
                "dim_promotion": {"primary_key": "promotion_key", "mode": "copy", "columns": {
                    "promotion_key": "id", "discount_pct": "float"}},
                "fact_sales": {"role": "fact", "unique": [["order_number", "line_number"]],
                               "fact": {"anchor": "dim_customer", "partner": "dim_product",
                                        "header": {"key": "order_number", "line_number": "line_number",
                                                   "columns": ["customer_key", "territory_key", "order_date_key",
                                                               "ship_date_key", "due_date_key", "order_date"]}},
                               "columns": {
                    "order_number": "id", "line_number": "integer", "customer_key": "id",
                    "territory_key": {**look, "lookup": "customer_key.geography_key.territory_key"},
                    "order_date_key": "id", "ship_date_key": "id", "due_date_key": "id",
                    "order_date": {**look, "lookup": "order_date_key.full_date"},
                    "product_key": "id", "promotion_key": "id", "order_quantity": "integer",
                    "unit_price": {**look, "lookup": "product_key.list_price"},
                    "discount_pct": {**look, "lookup": "promotion_key.discount_pct"},
                    "standard_cost": {**look, "lookup": "product_key.standard_cost"},
                    "extended_amount": "float", "discount_amount": "float", "sales_amount": "float",
                    "tax_amt": "float", "total_product_cost": "float"}},
                "fact_inventory": {"role": "fact", "unique": [["product_key", "date_key"]],
                                   "fact": {"anchor": "dim_product", "partner": "dim_date"}, "columns": {
                    "product_key": "id", "date_key": "id", "units_in": "integer", "units_out": "integer",
                    "units_balance": "integer"}}},
            "relationships": [
                {"child": "dim_geography", "child_key": "territory_key", "parent": "dim_territory"},
                {"child": "dim_customer", "child_key": "geography_key", "parent": "dim_geography"},
                {"child": "fact_sales", "child_key": "customer_key", "parent": "dim_customer"},
                {"child": "fact_sales", "child_key": "territory_key", "parent": "dim_territory"},
                {"child": "fact_sales", "child_key": "order_date_key", "parent": "dim_date"},
                {"child": "fact_sales", "child_key": "ship_date_key", "parent": "dim_date",
                 "offset_from": "order_date_key", "order_by": "full_date"},
                {"child": "fact_sales", "child_key": "due_date_key", "parent": "dim_date",
                 "offset_from": "order_date_key", "order_by": "full_date"},
                {"child": "fact_sales", "child_key": "product_key", "parent": "dim_product"},
                {"child": "fact_sales", "child_key": "promotion_key", "parent": "dim_promotion"},
                {"child": "fact_inventory", "child_key": "product_key", "parent": "dim_product"},
                {"child": "fact_inventory", "child_key": "date_key", "parent": "dim_date"}],
            "rules": [
                {"table": "dim_product", "type": "derive", "expr": "standard_cost = 0.55 * list_price", "tolerance": 0.0001},
                {"table": "fact_sales", "type": "derive", "expr": "extended_amount = order_quantity * unit_price",
                 "tolerance": 0.0001},
                {"table": "fact_sales", "type": "derive", "expr": "discount_amount = extended_amount * discount_pct",
                 "tolerance": 0.0001},
                {"table": "fact_sales", "type": "derive", "expr": "sales_amount = extended_amount - discount_amount",
                 "tolerance": 0.0001},
                {"table": "fact_sales", "type": "derive", "expr": "tax_amt = 0.08 * sales_amount", "tolerance": 0.0001},
                {"table": "fact_sales", "type": "derive", "expr": "total_product_cost = order_quantity * standard_cost",
                 "tolerance": 0.0001},
                {"table": "fact_inventory", "type": "cumulative", "column": "units_balance", "by": "product_key",
                 "order": "date_key", "step": "units_in - units_out"},
                {"table": "dim_customer", "type": "aggregate", "column": "first_purchase", "agg": "min",
                 "from": "fact_sales", "of": "order_date"}],
            "synthesis": _syn()}
    truth = {"obsolete": set(prod.product_key[prod.status == "obsolete"]), "ship_gaps": {7: 0.97, 8: 0.03},
             "due_gap": 12, "lines_mean": float(fact_sales.groupby("order_number").size().mean())}
    return tables, spec, truth


def retail_dw(seed=11, n_cust=3000, n_prod=400, days=1100):
    """Retail warehouse shaped like Contoso V2 with known structure:

    * Customer: a geography code GeoAreaKey (60 areas) determines State, StateFull,
      Country, Continent (an implied dimension); an active window StartDT..EndDT;
    * Store: physical stores per country with open (and some close) dates, plus
      one online store (StoreKey 999, CountryCode '--');
    * Product: SubCategoryKey (integer code) determines its name, CategoryKey and
      CategoryName; list Price and Cost;
    * Orders: keys numbered in date order; the customer is active and a physical
      store open on the order date and in the customer's country; online orders
      (40%) deliver 1-7 days later, store orders the same day; CurrencyCode is a
      map of the customer's country;
    * OrderRows: line numbers 0..n-1; UnitPrice = Price x f and UnitCost = Cost x f
      with one price factor f per (year, category) from 6 values; NetPrice =
      UnitPrice x (1 - d/100), d in 0..10, all stored to 4 decimals;
    * Sales: the view OrderRows joined to Orders with ExchangeRate =
      rate(OrderDate, 'USD', CurrencyCode) from CurrencyExchange, a reference
      table keyed by (Date, FromCurrency, ToCurrency) with no links."""
    rng = np.random.default_rng(seed)
    countries = {"US": ("USD", "North America", 24), "CA": ("CAD", "North America", 10),
                 "DE": ("EUR", "Europe", 10), "FR": ("EUR", "Europe", 8), "GB": ("GBP", "Europe", 8)}
    geo = []
    for cc, (_, cont, ns) in countries.items():
        for i in range(ns):
            st = f"{cc}{i:02d}"
            geo.append((len(geo) + 1, st, f"State {st} of {cc}", cc, cont))
    geo = pd.DataFrame(geo, columns=["GeoAreaKey", "State", "StateFull", "Country", "Continent"])
    gw = rng.dirichlet(np.full(len(geo), 2.0))
    g = rng.choice(len(geo), size=n_cust, p=gw)
    d0 = pd.Timestamp("2018-01-01")
    start = d0 - pd.to_timedelta(rng.integers(-500, 900, n_cust), unit="D")
    end = start + pd.to_timedelta(rng.integers(400, 2500, n_cust), unit="D")
    cust = pd.DataFrame({"CustomerKey": np.arange(1, n_cust + 1), "GeoAreaKey": geo.GeoAreaKey.to_numpy()[g],
                         "Gender": rng.choice(["female", "male"], n_cust),
                         "StartDT": start.strftime("%Y-%m-%d"), "EndDT": end.strftime("%Y-%m-%d")})
    cust = cust.merge(geo, on="GeoAreaKey", how="left")
    # latitude of the customer's address: depends on the country, not determined by the area
    lat0 = {"US": 38.0, "CA": 51.0, "DE": 51.0, "FR": 46.5, "GB": 53.5}
    rlat = np.random.default_rng(seed + 1000)
    cust["Latitude"] = (cust.Country.map(lat0) + rlat.normal(0, 2.0, len(cust))).round(4)
    cust = cust[["CustomerKey", "GeoAreaKey", "Gender", "State", "StateFull", "Country", "Continent", "Latitude",
                 "StartDT", "EndDT"]]
    stores = []
    k = 10
    for cc in countries:
        for j in range(3):
            op = d0 + pd.Timedelta(days=int(rng.integers(-800, 400)))
            cl = op + pd.Timedelta(days=int(rng.integers(500, 1500))) if j == 2 else None
            stores.append((k, cc, op.strftime("%Y-%m-%d"), cl.strftime("%Y-%m-%d") if cl is not None else None))
            k += 10
    stores.append((999, "--", "2010-01-01", None))
    store = pd.DataFrame(stores, columns=["StoreKey", "CountryCode", "OpenDate", "CloseDate"])
    nsub = 20
    sub = pd.DataFrame({"SubCategoryKey": np.arange(101, 101 + nsub), "CategoryKey": 1 + np.arange(nsub) % 5})
    sub["SubCategoryName"] = [f"Sub {k}" for k in sub.SubCategoryKey]
    sub["CategoryName"] = [f"Category {k}" for k in sub.CategoryKey]
    ps = rng.integers(0, nsub, n_prod)
    price = np.round(np.exp(rng.normal(4, 1, n_prod)), 2)
    prod = pd.DataFrame({"ProductKey": np.arange(1, n_prod + 1), "SubCategoryKey": sub.SubCategoryKey.to_numpy()[ps],
                         "Price": price, "Cost": np.round(price * rng.uniform(0.4, 0.7, n_prod), 2)})
    prod = prod.merge(sub, on="SubCategoryKey", how="left")[["ProductKey", "SubCategoryKey", "SubCategoryName",
                                                             "CategoryKey", "CategoryName", "Price", "Cost"]]
    factors = np.array([0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
    fac = {(y, c): factors[rng.integers(0, 6)] for y in range(2015, 2025) for c in range(1, 6)}
    # orders: date, customer active, store open in the customer's country or online
    cal = pd.date_range(d0, periods=days)
    rows, lines = [], []
    sd = {r.StoreKey: (pd.Timestamp(r.OpenDate), pd.Timestamp(r.CloseDate) if r.CloseDate else None, r.CountryCode)
          for r in store.itertuples()}
    cs, ce = pd.to_datetime(cust.StartDT).to_numpy(), pd.to_datetime(cust.EndDT).to_numpy()
    activity = rng.gamma(0.7, 1.0, n_cust)
    n_orders = 6000
    od = np.sort(cal[rng.integers(0, days, n_orders)])
    for i, dt in enumerate(od):
        dt = pd.Timestamp(dt)
        act = np.flatnonzero((cs < dt.to_datetime64()) & (dt.to_datetime64() < ce))
        if not len(act):
            continue
        w = activity[act] / activity[act].sum()
        c = int(act[rng.choice(len(act), p=w)])
        cc = cust.Country.iat[c]
        online = rng.random() < 0.4
        opn = [k for k, (o, cl, ctry) in sd.items() if ctry == cc and o <= dt and (cl is None or dt <= cl)]
        if not online and not opn:
            online = True
        sk = 999 if online else int(rng.choice(opn))
        deliv = dt + pd.Timedelta(days=int(rng.integers(1, 8))) if online else dt
        rows.append((len(rows) + 1, int(cust.CustomerKey.iat[c]), sk, dt.strftime("%Y-%m-%d"),
                     deliv.strftime("%Y-%m-%d"), countries[cc][0]))
        for ln in range(1 + rng.poisson(1.2)):
            p = int(rng.integers(0, n_prod))
            f = fac[(dt.year, int(prod.CategoryKey.iat[p]))]
            up = round(float(prod.Price.iat[p]) * f, 4)
            d = int(rng.choice(11, p=np.r_[0.4, np.full(10, 0.06)]))
            lines.append((rows[-1][0], ln, int(prod.ProductKey.iat[p]), int(1 + rng.poisson(2)), up,
                          round(up * (1 - d / 100), 4), round(float(prod.Cost.iat[p]) * f, 4)))
    orders = pd.DataFrame(rows, columns=["OrderKey", "CustomerKey", "StoreKey", "OrderDate", "DeliveryDate",
                                         "CurrencyCode"])
    orows = pd.DataFrame(lines, columns=["OrderKey", "LineNumber", "ProductKey", "Quantity", "UnitPrice", "NetPrice",
                                         "UnitCost"])
    cur = ["USD", "CAD", "EUR", "GBP"]
    base = {"USD": 1.0, "CAD": 1.3, "EUR": 0.9, "GBP": 0.8}
    walk = {c: base[c] * np.exp(np.cumsum(rng.normal(0, 0.003, days))) for c in cur}
    walk["USD"] = np.ones(days)
    ce_rows = [(cal[i].strftime("%Y-%m-%d"), a, b, round(walk[b][i] / walk[a][i], 5))
               for i in range(days) for a in cur for b in cur]
    rates = pd.DataFrame(ce_rows, columns=["Date", "FromCurrency", "ToCurrency", "Exchange"])
    sales = orows.merge(orders, on="OrderKey", how="left")
    rk = rates[rates.FromCurrency == "USD"].set_index(["Date", "ToCurrency"]).Exchange
    sales["ExchangeRate"] = rk.reindex(pd.MultiIndex.from_arrays([sales.OrderDate, sales.CurrencyCode])).to_numpy()
    sales = sales[["OrderKey", "LineNumber", "OrderDate", "DeliveryDate", "CustomerKey", "StoreKey", "ProductKey",
                   "Quantity", "UnitPrice", "NetPrice", "UnitCost", "CurrencyCode", "ExchangeRate"]]
    tables = {"Customer": cust, "Store": store, "Product": prod, "Orders": orders, "OrderRows": orows,
              "Sales": sales, "CurrencyExchange": rates}
    cmap = {c: v[0] for c, v in countries.items()}
    spec = {"version": 1, "name": "retail_dw", "source": {"kind": "csv", "path": "."}, "tables": {
        "Customer": {"primary_key": "CustomerKey", "columns": {
            "CustomerKey": "id", "GeoAreaKey": "integer", "Gender": "categorical", "State": "categorical",
            "StateFull": "categorical", "Country": "categorical", "Continent": "categorical",
            "Latitude": "float", "StartDT": "date", "EndDT": {"type": "date", "offset_from": "StartDT"}},
            "dimensions": [{"key": "GeoAreaKey", "columns": ["State", "StateFull", "Country", "Continent"],
                            "anchor": "Country"}]},
        "Store": {"primary_key": "StoreKey", "mode": "copy", "columns": {
            "StoreKey": "id", "CountryCode": "categorical", "OpenDate": "date", "CloseDate": "date"}},
        "Product": {"primary_key": "ProductKey", "columns": {
            "ProductKey": "id", "SubCategoryKey": "integer", "SubCategoryName": "categorical",
            "CategoryKey": "integer", "CategoryName": "categorical", "Price": "float", "Cost": "float"},
            "dimensions": [{"key": "SubCategoryKey", "columns": ["SubCategoryName", "CategoryKey", "CategoryName"]}]},
        "Orders": {"primary_key": "OrderKey", "key_order": "OrderDate", "columns": {
            "OrderKey": "id", "CustomerKey": "id", "StoreKey": "id", "OrderDate": "date",
            "DeliveryDate": {"type": "date", "offset_from": "OrderDate"},
            "CurrencyCode": {"type": "lookup", "lookup": "CustomerKey.Country", "map": cmap}}},
        "OrderRows": {"role": "fact", "unique": [["OrderKey", "LineNumber"]], "fact": {"anchor": "Orders", "partner": "Product"},
                      "columns": {"OrderKey": "id", "LineNumber": {"type": "sequence", "within": "OrderKey", "start": 0},
                                  "ProductKey": "id", "Quantity": "integer",
                                  "UnitPrice": {"type": "float", "ratio_of": "ProductKey.Price"},
                                  "NetPrice": {"type": "float", "ratio_of": "UnitPrice"},
                                  "UnitCost": {"type": "float", "ratio_of": "ProductKey.Cost", "ratio_same_as": "UnitPrice"}}},
        "Sales": {"role": "view", "view_of": "OrderRows", "columns": {
            "OrderKey": "id", "LineNumber": "id", "OrderDate": "lookup:OrderKey.OrderDate",
            "DeliveryDate": "lookup:OrderKey.DeliveryDate", "CustomerKey": "lookup:OrderKey.CustomerKey",
            "StoreKey": "lookup:OrderKey.StoreKey", "ProductKey": "id", "Quantity": "integer", "UnitPrice": "float",
            "NetPrice": "float", "UnitCost": "float", "CurrencyCode": "lookup:OrderKey.CurrencyCode",
            "ExchangeRate": {"type": "lookup", "lookup": {"table": "CurrencyExchange", "value": "Exchange", "match": {
                "Date": "OrderDate", "FromCurrency": "'USD'", "ToCurrency": "CurrencyCode"}}}}},
        "CurrencyExchange": {"mode": "copy", "unique": [["Date", "FromCurrency", "ToCurrency"]], "columns": {
            "Date": "date", "FromCurrency": "categorical", "ToCurrency": "categorical", "Exchange": "float"}}},
        "relationships": [
            {"child": "Orders", "child_key": "CustomerKey", "parent": "Customer"},
            {"child": "Orders", "child_key": "StoreKey", "parent": "Store"},
            {"child": "OrderRows", "child_key": "OrderKey", "parent": "Orders"},
            {"child": "OrderRows", "child_key": "ProductKey", "parent": "Product"},
            {"child": "Sales", "child_key": "OrderKey", "parent": "Orders"},
            {"child": "Sales", "child_key": "ProductKey", "parent": "Product"}],
        "rules": [
            {"table": "Orders", "type": "check", "expr": "days(CustomerKey.StartDT) < days(OrderDate)"},
            {"table": "Orders", "type": "check", "expr": "days(OrderDate) < days(CustomerKey.EndDT)"},
            {"table": "Orders", "type": "check", "expr": "days(StoreKey.OpenDate) <= days(OrderDate)"},
            {"table": "Orders", "type": "check", "expr": "days(OrderDate) <= days(StoreKey.CloseDate)"},
            {"table": "Orders", "type": "check", "expr": "days(OrderDate) == days(DeliveryDate) or StoreKey == 999"},
            {"table": "Orders", "type": "check", "expr": "days(OrderDate) != days(DeliveryDate) or StoreKey != 999"},
            {"table": "Orders", "type": "check", "expr": "CustomerKey.Country == StoreKey.CountryCode or StoreKey == 999"}],
        "synthesis": _syn()}
    truth = {"factors": factors, "fac": fac, "countries": countries, "geo": geo}
    return tables, spec, truth


SCENARIOS = {"S1_mixed_table": mixed_table, "S2_star": star, "S3_snowflake": snowflake, "S4_tree": tree,
             "S5_temporal": temporal, "S6_text": text, "S7_enterprise": enterprise, "S8_reporting": reporting,
             "S9_small_shop": small_shop, "S10_order_dw": order_dw, "S11_retail_dw": retail_dw}


def write(name, tables, spec, out=OUT):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for t, df in tables.items():
        df.to_csv(d / f"{t}.csv", index=False)
    s = dict(spec)
    s["target"] = {"kind": "csv", "path": "synthetic"}
    (d / "spec.yaml").write_text(yaml.safe_dump(s, sort_keys=False, width=110))
    return d


if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENARIOS)
    for n in names:
        tables, spec, _ = SCENARIOS[n]()
        d = write(n, tables, spec)
        print(n, {t: len(df) for t, df in tables.items()}, "->", d)
