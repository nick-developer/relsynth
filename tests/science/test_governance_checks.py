# SPDX-License-Identifier: Apache-2.0
"""Checks C800-C803: the governance propositions of docs/proofs (section v2.5,
governance), tested against known answers.  T08 is in its design phase, so the
ledger, the reservation protocol and the hash chain are small reference
implementations kept in this file; the build phase replaces them with
`relsynth.governance` and keeps these checks.  About ten seconds:

    pytest -m science tests/science/test_governance_checks.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom, norm

pytestmark = [pytest.mark.science]


# ---------------------------------------------------------------------------
# Reference privacy-loss ledger (Prop. 800)
# ---------------------------------------------------------------------------

def to_azcdp(entry):
    """One ledger entry as delta-approximate rho-zCDP: (rho, delta).

    pure eps          -> (eps^2/2, 0)    Bun and Steinke 2016, Prop. 1.4
    zcdp rho          -> (rho, 0)
    approx (eps, del) -> (eps^2/2, del)  approximate zCDP, Bun and Steinke 2016
    """
    kind = entry["kind"]
    if kind == "pure":
        return entry["epsilon"] ** 2 / 2, 0.0
    if kind == "zcdp":
        return entry["rho"], 0.0
    if kind == "approx":
        return entry["epsilon"] ** 2 / 2, entry["delta"]
    raise ValueError(kind)


def compose(entries):
    """Adaptive composition: rho and delta add (Prop. 800 a)."""
    rho = sum(to_azcdp(e)[0] for e in entries)
    dlt = sum(to_azcdp(e)[1] for e in entries)
    return rho, dlt


def cks_delta(rho, eps):
    """delta(eps) of rho-zCDP, Canonne, Kamath and Steinke 2020, Prop. 12:
    inf_{a>1} exp((a-1)(a rho - eps)) / (a-1) * (1 - 1/a)^a."""
    if rho <= 0:
        return 0.0

    def logd(la):
        a = 1 + math.exp(la)
        return (a - 1) * (a * rho - eps) - math.log(a - 1) + a * math.log1p(-1 / a)

    grid = np.linspace(-12, 12, 481)
    vals = [logd(x) for x in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(logd, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return min(1.0, math.exp(min(r.fun, vals[i])))


def ledger_epsilon(entries, delta):
    """Smallest eps the ledger states at total delta <= `delta` (Prop. 800 b, c):
    the zCDP conversion, or basic composition when every entry is pure or
    approximate and that is smaller (both are valid, so their minimum is)."""
    rho, d0 = compose(entries)
    if d0 >= delta:
        return math.inf
    if rho == 0:
        return 0.0
    target = delta - d0
    hi = rho + 2 * math.sqrt(rho * math.log(1 / target)) + 1
    eps = brentq(lambda e: cks_delta(rho, e) - target, 0.0, hi, xtol=1e-9)
    if all(e["kind"] in ("pure", "approx") for e in entries):
        if sum(e.get("delta", 0.0) for e in entries) <= delta:
            eps = min(eps, sum(e["epsilon"] for e in entries))
    return eps


def classic_epsilon(rho, delta):
    """Bun and Steinke 2016, Prop. 1.3: eps = rho + 2 sqrt(rho log(1/delta))."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


# Exact privacy profiles of mechanisms with known answers -------------------

def gauss_delta(mu, eps):
    """Exact delta(eps) of a Gaussian mechanism with mu = sensitivity / sigma
    (Balle and Wang 2018, Thm. 8); a composition of Gaussians with zCDP rho_i
    is a Gaussian with mu = sqrt(2 * sum rho_i) (Dong, Roth and Su 2022)."""
    return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def gauss_epsilon(mu, delta):
    if gauss_delta(mu, 0.0) <= delta:
        return 0.0
    return brentq(lambda e: gauss_delta(mu, e) - delta, 0.0, 50 + mu * mu, xtol=1e-10)


def rr_delta(eps0, k, eps):
    """Exact delta(eps) of k-fold randomised response at eps0 (Kairouz, Oh and
    Viswanath 2015): the privacy loss is eps0 * (k - 2j), j ~ Bin(k, q)."""
    q = 1 / (1 + math.exp(eps0))
    j = np.arange(k + 1)
    loss = eps0 * (k - 2 * j)
    pj = binom.pmf(j, k, q)
    return float(np.sum(pj * np.clip(1 - np.exp(eps - loss), 0, None)))


def rr_epsilon(eps0, k, delta):
    if rr_delta(eps0, k, 0.0) <= delta:
        return 0.0
    return brentq(lambda e: rr_delta(eps0, k, e) - delta, 0.0, eps0 * k, xtol=1e-10)


def test_C800_ledger_bounds_exact_privacy_loss():
    """Prop. 800: for compositions with known exact privacy profiles, the
    ledger's (eps, delta) is never below the exact eps at that delta, and never
    above the classic zCDP conversion."""
    rng = np.random.default_rng(800)
    worst = 0.0
    for _ in range(60):
        k = int(rng.integers(1, 21))
        rhos = rng.uniform(0.001, 0.2, k)
        delta = float(10.0 ** rng.uniform(-10, -5))
        led = ledger_epsilon([{"kind": "zcdp", "rho": float(r)} for r in rhos], delta)
        exact = gauss_epsilon(math.sqrt(2 * rhos.sum()), delta)
        assert led >= exact - 1e-7, (k, rhos.sum(), delta, led, exact)
        assert led <= classic_epsilon(rhos.sum(), delta) + 1e-7
        worst = max(worst, led / exact)
    assert worst < 1.6        # the ledger is sound and not grossly loose
    for eps0, k in [(0.1, 1), (0.1, 50), (0.5, 10), (1.0, 4), (2.0, 2)]:
        for delta in (1e-6, 1e-9):
            led = ledger_epsilon([{"kind": "pure", "epsilon": eps0}] * k, delta)
            assert led >= rr_epsilon(eps0, k, delta) - 1e-7
            assert led <= eps0 * k + 1e-9          # never worse than basic composition
    # approximate entries carry their delta, and the target delta must exceed it
    assert ledger_epsilon([{"kind": "approx", "epsilon": 1.0, "delta": 1e-6}], 1e-6) == math.inf
    e1 = ledger_epsilon([{"kind": "approx", "epsilon": 1.0, "delta": 1e-7}], 1e-6)
    e0 = ledger_epsilon([{"kind": "pure", "epsilon": 1.0}], 1e-6 - 1e-7)
    assert abs(e1 - e0) < 1e-6


def test_C801_period_caps_understate_lifetime_loss():
    """Prop. 801: a person present in P periods, each charged rho, has privacy
    loss of P * rho.  A cap that resets each period states the per-period
    loss only; the lifetime ledger covers the exact loss."""
    delta = 1e-8
    for rho, P in [(0.05, 12), (0.2, 4), (0.01, 52)]:
        exact = gauss_epsilon(math.sqrt(2 * rho * P), delta)
        per_period = ledger_epsilon([{"kind": "zcdp", "rho": rho}], delta)
        lifetime = ledger_epsilon([{"kind": "zcdp", "rho": rho}] * P, delta)
        assert per_period < exact <= lifetime + 1e-7
        assert lifetime > 1.9 * per_period


# ---------------------------------------------------------------------------
# Reservation protocol (Prop. 802)
# ---------------------------------------------------------------------------

class Ledger:
    """Append-only ledger for one privacy domain with a lifetime cap on rho.

    reserve -> (touch data) -> settle(actual) | crash.  A reservation is
    refunded only if the run never touched data; otherwise it is charged in
    full until settled with the accountant's figure, which is at most the
    reservation."""

    def __init__(self, cap):
        self.cap, self.lock, self.log = cap, threading.Lock(), []

    def charged(self):
        state = {}
        for op, run, x in self.log:
            if op == "reserve":
                state[run] = x
            elif op == "settle":
                state[run] = x
            elif op == "release":
                state[run] = 0.0
        return sum(state.values())

    def reserve(self, run, rho):
        with self.lock:
            if self.charged() + rho > self.cap + 1e-12:
                return False
            self.log.append(("reserve", run, rho))
            return True

    def touched(self, run):
        with self.lock:
            self.log.append(("touch", run, None))

    def settle(self, run, actual):
        with self.lock:
            reserved = [x for op, r, x in self.log if op == "reserve" and r == run][-1]
            if actual > reserved + 1e-12:
                raise ValueError("a run may not spend more than it reserved")
            self.log.append(("settle", run, actual))

    def release(self, run):
        with self.lock:
            if any(op == "touch" and r == run for op, r, _ in self.log):
                raise ValueError("a run that touched data cannot be refunded")
            self.log.append(("release", run, None))


def test_C802_reservations_never_undercharge():
    """Prop. 802: with reserve-before-read, refunds only without data access
    and settlement at most the reservation, the charged total is at least the
    true privacy loss and at most the cap, under concurrency and crashes."""
    for trial in range(20):
        led, true_loss, tl_lock = Ledger(cap=1.0), [0.0], threading.Lock()
        rnd = random.Random(trial)

        def run(i, rnd_seed):
            r = random.Random(rnd_seed)
            want = r.uniform(0.02, 0.3)
            if not led.reserve(i, want):
                return
            if r.random() < 0.2:          # gives up before reading data
                led.release(i)
                return
            led.touched(i)
            steps = r.randint(1, 5)
            spent = 0.0
            for _ in range(steps):
                if r.random() < 0.15:     # crash: no settle, reservation stays charged
                    with tl_lock:
                        true_loss[0] += spent
                    return
                spent += want / steps * r.uniform(0.5, 1.0)
            with tl_lock:
                true_loss[0] += spent
            led.settle(i, spent)

        threads = [threading.Thread(target=run, args=(i, rnd.random())) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert led.charged() <= led.cap + 1e-9
        assert led.charged() >= true_loss[0] - 1e-9
    with pytest.raises(ValueError):
        led = Ledger(1.0)
        led.reserve("x", 0.1)
        led.touched("x")
        led.release("x")


# ---------------------------------------------------------------------------
# Hash chain and signed checkpoints (Prop. 803)
# ---------------------------------------------------------------------------

def canon(obj) -> bytes:
    """Canonical JSON for the objects relsynth writes (strings, integers,
    booleans, null, lists, objects; no floats in hashed fields).  On that
    subset this equals RFC 8785 (JCS) output."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


GENESIS = "0" * 64


def append(chain, body):
    prev = chain[-1]["hash"] if chain else GENESIS
    h = hashlib.sha256(prev.encode() + canon(body)).hexdigest()
    chain.append({"seq": len(chain), "prev": prev, "body": body, "hash": h})


def verify(chain, checkpoint, pub):
    """True iff the chain is intact and ends at the signed checkpoint."""
    from cryptography.exceptions import InvalidSignature
    try:
        pub.verify(bytes.fromhex(checkpoint["sig"]), canon({"seq": checkpoint["seq"], "hash": checkpoint["hash"]}))
    except InvalidSignature:
        return False
    prev = GENESIS
    for i, e in enumerate(chain):
        if e["seq"] != i or e["prev"] != prev:
            return False
        if hashlib.sha256(prev.encode() + canon(e["body"])).hexdigest() != e["hash"]:
            return False
        prev = e["hash"]
    return len(chain) - 1 == checkpoint["seq"] and prev == checkpoint["hash"]


def test_C803_tampering_is_detected():
    """Prop. 803: any edit, deletion, insertion, reordering or truncation of
    the log before a signed checkpoint is detected; a forged checkpoint is
    rejected."""
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    key = ed.Ed25519PrivateKey.generate()
    pub = key.public_key()
    chain = []
    for i in range(200):
        append(chain, {"type": "run", "run_id": f"r{i}", "rho_milli": i % 17, "verdict": ["pass", "review"][i % 2]})
    cp = {"seq": len(chain) - 1, "hash": chain[-1]["hash"]}
    cp["sig"] = key.sign(canon(cp)).hex()
    assert verify(chain, cp, pub)

    import copy
    rnd = random.Random(803)
    detected = 0
    for trial in range(400):
        c = copy.deepcopy(chain)
        kind = trial % 5
        j = rnd.randrange(1, len(c) - 1)
        if kind == 0:                      # edit one field
            c[j]["body"]["rho_milli"] += 1
        elif kind == 1:                    # delete an entry
            del c[j]
        elif kind == 2:                    # insert a forged entry
            c.insert(j, {"seq": j, "prev": c[j - 1]["hash"], "body": {"type": "run", "run_id": "x"},
                         "hash": hashlib.sha256(c[j - 1]["hash"].encode() + canon({"type": "run", "run_id": "x"})).hexdigest()})
        elif kind == 3:                    # swap two entries
            c[j], c[j + 1] = c[j + 1], c[j]
        else:                              # truncate the tail
            c = c[:j]
        detected += not verify(c, cp, pub)
    assert detected == 400

    forged = dict(cp, hash=hashlib.sha256(b"other").hexdigest())
    assert not verify(chain, forged, pub)
    other = ed.Ed25519PrivateKey.generate()
    resigned = {"seq": cp["seq"], "hash": cp["hash"], "sig": other.sign(canon({"seq": cp["seq"], "hash": cp["hash"]})).hex()}
    assert not verify(chain, resigned, pub)
