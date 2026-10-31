# SPDX-License-Identifier: Apache-2.0
"""The privacy budget ledger of the organisation (design sec. 5, Props. 800-802).

The ledger is a view over the `ledger.*` entries of the registry log, so the
budget history is covered by the hash chain.  A DP run

1. **reserves** its planned (rho, delta) under the registry lock; the
   reservation is refused if it would exceed any cap of the domain;
2. writes a **touch** entry before its first read of real data;
3. **settles** with the accountant's figure, which may not exceed the
   reservation;
4. is **refunded** (released) only if it never touched data.

A run that crashed after its touch stays charged with its full reservation,
so the charged total is never below the true privacy loss and never above
the cap (Prop. 802).  Revoking a release never refunds budget.

Non-DP releases of a domain are recorded as `ledger.nondp` entries: they spend
nothing, but every DP statement of the domain then says that other releases
of the same people were made without DP (NIST SP 800-226 sec. 2.6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..errors import PrivacyError
from .accounting import dec, epsilon_of
from .policy import Policy

_EPS = 1e-12


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass
class RunState:
    run: str
    domain: str
    reserved_rho: float
    reserved_delta: float
    time: str
    touched: bool = False
    settled: tuple | None = None          # (rho, delta)
    released: bool = False

    @property
    def charged(self) -> tuple[float, float]:
        if self.released:
            return 0.0, 0.0
        if self.settled is not None:
            return self.settled
        return self.reserved_rho, self.reserved_delta

    @property
    def state(self) -> str:
        if self.released:
            return "refunded"
        if self.settled is not None:
            return "settled"
        return "touched" if self.touched else "reserved"


@dataclass
class DomainState:
    domain: str
    runs: dict = field(default_factory=dict)       # run id -> RunState
    nondp: list = field(default_factory=list)      # run ids released without DP

    def charged(self, since: datetime | None = None) -> tuple[float, float]:
        r = d = 0.0
        for s in self.runs.values():
            if since is not None and _t(s.time) < since:
                continue
            cr, cd = s.charged
            r += cr
            d += cd
        return r, d


class Ledger:
    """Budget operations on one registry under one policy."""

    def __init__(self, registry, policy: Policy):
        self.reg = registry
        self.policy = policy

    # ------------------------------------------------------------- state
    def state(self, domain: str) -> DomainState:
        st = DomainState(domain)
        for rec in self.reg.entries("ledger."):
            e = rec["entry"]
            if e.get("domain") != domain:
                continue
            k, run = e["type"], e["run"]
            if k == "ledger.reserve":
                st.runs[run] = RunState(run, domain, float(e["rho"]), float(e["delta"]), e["time"])
            elif k == "ledger.touch" and run in st.runs:
                st.runs[run].touched = True
            elif k == "ledger.settle" and run in st.runs:
                st.runs[run].settled = (float(e["rho"]), float(e["delta"]))
            elif k == "ledger.release" and run in st.runs:
                st.runs[run].released = True
            elif k == "ledger.nondp":
                st.nondp.append(run)
        return st

    def _check_caps(self, domain: str, rho: float, delta: float, st: DomainState, now: datetime) -> list[str]:
        caps = self.policy.caps(domain)
        r, d = st.charged()
        why = []
        if d + delta >= caps.delta:
            why.append(f"domain {domain!r} would spend delta {d + delta:g}, not below its delta {caps.delta:g}")
        if r + rho > caps.lifetime_rho + _EPS:
            why.append(f"domain {domain!r} would reach rho {r + rho:.6g}, above its lifetime cap {caps.lifetime_rho:.6g} "
                       f"(epsilon {caps.lifetime_epsilon:g} at delta {caps.delta:g})")
        if caps.per_release_rho is not None and rho > caps.per_release_rho + _EPS:
            why.append(f"this release would spend rho {rho:.6g}, above the per-release cap {caps.per_release_rho:.6g} "
                       f"(epsilon {caps.per_release_epsilon:g} at delta {caps.delta:g})")
        if caps.period is not None:
            pr, _ = st.charged(since=now - caps.period)
            assert caps.period_rho is not None
            if pr + rho > caps.period_rho + _EPS:
                why.append(f"domain {domain!r} would spend rho {pr + rho:.6g} this period, above the period cap "
                           f"{caps.period_rho:.6g} (a throttle, epsilon {caps.period_epsilon:g})")
        return why

    # ------------------------------------------------------------- operations
    def reserve(self, domain: str, run: str, rho: float, delta: float = 0.0, now: datetime | None = None) -> int:
        """Reserve the planned budget of a run before it reads data (Prop. 802)."""
        if domain not in self.policy.domains:
            raise PrivacyError(f"domain {domain!r} is not in the organisation policy; a DP run needs a domain")
        if not (rho >= 0 and math.isfinite(rho) and delta >= 0):
            raise PrivacyError(f"bad reservation rho={rho} delta={delta}")
        now = now or datetime.now(timezone.utc)
        with self.reg.locked():
            st = self.state(domain)
            if run in st.runs:
                raise PrivacyError(f"run {run} already holds a reservation")
            why = self._check_caps(domain, rho, delta, st, now)
            if why:
                raise PrivacyError("the privacy budget refuses this run:\n  - " + "\n  - ".join(why))
            return self.reg.append({"type": "ledger.reserve", "domain": domain, "run": run,
                                    "rho": dec(rho), "delta": dec(delta, 30)})

    def touch(self, domain: str, run: str) -> int:
        """Record that the run is about to read real data: from now on no refund."""
        with self.reg.locked():
            if run not in self.state(domain).runs:
                raise PrivacyError(f"run {run} has no reservation in domain {domain!r}; reserve before reading data")
            return self.reg.append({"type": "ledger.touch", "domain": domain, "run": run})

    def settle(self, domain: str, run: str, rho: float, delta: float = 0.0) -> int:
        with self.reg.locked():
            st = self.state(domain).runs.get(run)
            if st is None:
                raise PrivacyError(f"run {run} has no reservation in domain {domain!r}")
            if st.released or st.settled is not None:
                raise PrivacyError(f"run {run} is already {st.state}")
            if rho > st.reserved_rho + _EPS or delta > st.reserved_delta + _EPS:
                raise PrivacyError(f"run {run} spent rho {rho:.6g}, delta {delta:g}: more than it reserved "
                                   f"(rho {st.reserved_rho:.6g}, delta {st.reserved_delta:g})")
            return self.reg.append({"type": "ledger.settle", "domain": domain, "run": run,
                                    "rho": dec(rho), "delta": dec(delta, 30)})

    def release(self, domain: str, run: str) -> int:
        """Refund a reservation; refused once the run has touched data."""
        with self.reg.locked():
            st = self.state(domain).runs.get(run)
            if st is None:
                raise PrivacyError(f"run {run} has no reservation in domain {domain!r}")
            if st.touched:
                raise PrivacyError(f"run {run} read real data: its reservation cannot be refunded (Prop. 802)")
            if st.released or st.settled is not None:
                raise PrivacyError(f"run {run} is already {st.state}")
            return self.reg.append({"type": "ledger.release", "domain": domain, "run": run})

    def nondp(self, domain: str, run: str) -> int:
        return self.reg.append({"type": "ledger.nondp", "domain": domain, "run": run})

    # ------------------------------------------------------------- figures
    def summary(self, domain: str, now: datetime | None = None) -> dict:
        """The domain's spent, reserved and remaining budget in rho and epsilon."""
        caps = self.policy.caps(domain)
        st = self.state(domain)
        r, d = st.charged()
        out = {"domain": domain, "delta": caps.delta, "rho": r, "delta_spent": d,
               "epsilon": epsilon_of(r, caps.delta - d) if r > 0 and d < caps.delta else (0.0 if r == 0 else math.inf),
               "lifetime_rho_cap": caps.lifetime_rho, "lifetime_epsilon_cap": caps.lifetime_epsilon,
               "remaining_rho": max(caps.lifetime_rho - r, 0.0),
               "runs": {k: {"state": v.state, "rho": v.charged[0], "delta": v.charged[1], "time": v.time}
                        for k, v in st.runs.items()},
               "nondp_releases": list(st.nondp)}
        if caps.period is not None:
            now = now or datetime.now(timezone.utc)
            out["period_rho"] = st.charged(since=now - caps.period)[0]
            out["period_rho_cap"] = caps.period_rho
        return out

    def policy_input(self, domain: str, run_rho: float, run_delta: float = 0.0, run: str | None = None,
                     now: datetime | None = None) -> dict:
        """The `ledger` part of the policy input (design sec. 8.2) for a run of this
        domain; if `run` is already charged it is not counted twice."""
        caps = self.policy.caps(domain)
        st = self.state(domain)
        if run in st.runs:
            st.runs.pop(run)
        r, _ = st.charged()
        out = {"domain": domain, "delta": caps.delta, "run_rho": run_rho, "domain_rho_after": r + run_rho,
               "lifetime_rho_cap": caps.lifetime_rho, "lifetime_epsilon_cap": caps.lifetime_epsilon,
               "per_release_rho_cap": caps.per_release_rho if caps.per_release_rho is not None else caps.lifetime_rho,
               "per_release_epsilon_cap": caps.per_release_epsilon or caps.lifetime_epsilon}
        if caps.period is not None:
            now = now or datetime.now(timezone.utc)
            out["period_rho_after"] = st.charged(since=now - caps.period)[0] + run_rho
            out["period_rho_cap"] = caps.period_rho
        return out
