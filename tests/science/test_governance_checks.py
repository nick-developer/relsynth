# SPDX-License-Identifier: Apache-2.0
"""Checks C800-C803: the governance propositions of docs/proofs (section v2.5,
governance), tested against known answers on the shipped code
(`relsynth.governance`: accounting, the ledger on a local registry, the hash
chain and checkpoints).  The design phase kept reference implementations
here; the build replaced them and kept the checks.  About thirty seconds:

    pytest -m science tests/science/test_governance_checks.py
"""
from __future__ import annotations

import json
import math
import random
import threading

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import binom, norm

pytestmark = [pytest.mark.science]


# ---------------------------------------------------------------------------
# Reference privacy-loss ledger (Prop. 800)
# ---------------------------------------------------------------------------

from relsynth.governance.accounting import compose, to_azcdp  # noqa: E402,F401
from relsynth.governance.accounting import epsilon_of as _eps_of  # noqa: E402
from relsynth.governance.accounting import ledger_epsilon  # noqa: E402


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

def _ledger(tmp, cap_rho):
    from relsynth import governance as G
    owner = G.Signer.generate("owner")
    reg = G.LocalRegistry.create(tmp, owner)
    eps = _eps_of(cap_rho, 1e-9)
    pol = G.policy.load({
        "schema": "relsynth.org-policy/1", "organisation": "test", "version": "1",
        "privacy_owner": {"name": "o", "role": "dpo"},
        "domains": {"d": {"unit": {"entity": "P"}, "sources": [{"id": "s"}],
                          "budget": {"accounting": "zcdp", "delta": 1e-9, "lifetime": {"epsilon": eps}}}},
        "contexts": {"internal": {"external": False, "dp_required": False, "limits": {},
                                  "approval": {"on_verdicts": ["review"], "approver_roles": ["dpo"]}}}})
    return G.Ledger(reg, pol), pol.caps("d").lifetime_rho


def test_C802_reservations_never_undercharge(tmp_path):
    """Prop. 802: with reserve-before-read, refunds only without data access
    and settlement at most the reservation, the charged total is at least the
    true privacy loss and at most the cap, under concurrency and crashes
    (relsynth.governance.Ledger on a LocalRegistry)."""
    from relsynth.errors import PrivacyError
    for trial in range(20):
        led, cap = _ledger(tmp_path / f"r{trial}", 1.0)
        true_loss, tl_lock = [0.0], threading.Lock()
        rnd = random.Random(trial)

        def run(i, rnd_seed):
            r = random.Random(rnd_seed)
            want = r.uniform(0.02, 0.3)
            try:
                led.reserve("d", f"run{i}", want)
            except PrivacyError:
                return
            if r.random() < 0.2:          # gives up before reading data
                led.release("d", f"run{i}")
                return
            led.touch("d", f"run{i}")
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
            led.settle("d", f"run{i}", spent)

        threads = [threading.Thread(target=run, args=(i, rnd.random())) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        charged = led.summary("d")["rho"]
        assert charged <= cap + 1e-9
        assert charged >= true_loss[0] - 1e-9
    led, _ = _ledger(tmp_path / "x", 1.0)
    led.reserve("d", "x", 0.1)
    led.touch("d", "x")
    with pytest.raises(PrivacyError):
        led.release("d", "x")


# ---------------------------------------------------------------------------
# Hash chain and signed checkpoints (Prop. 803)
# ---------------------------------------------------------------------------

def test_C803_tampering_is_detected(tmp_path):
    """Prop. 803: any edit, deletion, insertion, reordering or truncation of
    the log before a signed checkpoint is detected; a forged checkpoint is
    rejected (relsynth.governance.LocalRegistry and verify_registry)."""
    import copy
    import hashlib as _h

    from relsynth import governance as G
    from relsynth.governance.registry import chain_hash
    owner = G.Signer.generate("owner")
    reg = G.LocalRegistry.create(tmp_path / "reg", owner)
    for i in range(200):
        reg.append({"type": "note", "run_id": f"r{i}", "rho_milli": i % 17, "verdict": ["pass", "review"][i % 2]})
    reg.checkpoint(owner)
    assert G.verify_registry(reg) == []
    log = reg.root / "log.jsonl"
    good = log.read_text().splitlines()
    rnd = random.Random(803)
    detected = 0
    for trial in range(400):
        c = [json.loads(x) for x in good]
        kind = trial % 5
        j = rnd.randrange(2, len(c) - 2)
        if kind == 0:                      # edit one field
            c[j]["entry"]["rho_milli"] = c[j]["entry"].get("rho_milli", 0) + 1
        elif kind == 1:                    # delete an entry
            del c[j]
        elif kind == 2:                    # insert a forged entry with a correct local hash
            e = {"type": "note", "run_id": "x"}
            c.insert(j, {"seq": j, "prev": c[j - 1]["hash"], "entry": e, "hash": chain_hash(c[j - 1]["hash"], e)})
        elif kind == 3:                    # swap two entries
            c[j], c[j + 1] = c[j + 1], c[j]
        else:                              # truncate the tail
            c = c[:j]
        log.write_text("\n".join(json.dumps(x) for x in c) + "\n")
        detected += bool(G.verify_registry(reg))
    assert detected == 400
    log.write_text("\n".join(good) + "\n")
    assert G.verify_registry(reg) == []
    # a recomputed chain after an edit (an operator without the checkpoint key) fails the checkpoint
    c = [json.loads(x) for x in good]
    c[5]["entry"]["verdict"] = "edited"
    prev = "0" * 64
    for x in c:
        x["prev"] = prev
        x["hash"] = chain_hash(prev, x["entry"])
        prev = x["hash"]
    log.write_text("\n".join(json.dumps(x) for x in c) + "\n")
    assert any("checkpoint" in p for p in G.verify_registry(reg))
    # a checkpoint signed by a key the registry does not trust is rejected
    log.write_text("\n".join(good) + "\n")
    cpf = sorted((reg.root / "checkpoints").glob("*.dsse"))[-1]
    cp = json.loads(cpf.read_text())
    st = G.payload(cp)
    st["predicate"]["hash"] = _h.sha256(b"other").hexdigest()
    cpf.write_text(json.dumps(G.envelope(st, G.Signer.generate("mallory"))))
    assert any("checkpoint" in p for p in G.verify_registry(reg))
    _ = copy
