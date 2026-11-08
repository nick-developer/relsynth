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

A domain with several privacy units keeps one account per unit: every
`ledger.*` entry of such a domain names its `unit`, and each unit's caps
(its own `unit_budgets` entry, or the domain's budget) apply to that unit's
account alone (RFC-G1-01, Prop. 1704).  Entries of a one-unit domain carry no
unit, as before.  When the policy changes a domain's units, no spend is lost:
entries without a unit count against every unit's account, and entries of
several units all count against a domain that has one unit again.  A run caught
mid-flight by such a change is still touched, settled or refunded, with entries
that name its reservation's own unit and run id.

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
    unit: str | None = None               # the unit its entries name (None: written for a one-unit domain)
    key: str = ""                         # the run id its entries name

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
    unit: str | None = None
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
    def _unit(self, domain: str, unit: str | None) -> str | None:
        """The account key: None in a one-unit domain, the unit in a domain with several."""
        if domain not in self.policy.domains or not self.policy.multi_unit(domain):
            return None
        if unit is None:
            raise PrivacyError(f"domain {domain!r} has several privacy units ({', '.join(self.policy.units(domain))}): "
                               "name the unit whose account this is")
        if unit not in self.policy.units(domain):
            raise PrivacyError(f"{unit!r} is not a privacy unit of domain {domain!r}")
        return unit

    def state(self, domain: str, unit: str | None = None) -> DomainState:
        unit = self._unit(domain, unit)
        st = DomainState(domain, unit)
        ids: dict = {}                        # (unit, run id) of a reservation -> its key in st.runs
        for rec in self.reg.entries("ledger."):
            e = rec["entry"]
            if e.get("domain") != domain:
                continue
            eu = e.get("unit")
            k, run = e["type"], e["run"]
            # an account never forgets spend recorded before the policy changed the domain's units:
            # entries written without a unit (one-unit domain) count against every unit's account, and
            # after a return to one unit every unit's entries count against the domain (Prop. 1704)
            if eu is not None and unit is None:
                run = f"{run}@{eu}"
            elif eu is not None and eu != unit and k != "ledger.nondp":
                continue
            if k == "ledger.nondp":
                st.nondp.append(run)
                continue
            ident = (eu, e["run"])
            if k == "ledger.reserve":
                # each reservation keeps its own key, even if an older run id looks like a replay key
                key, i = run, 2
                while key in st.runs:
                    key, i = f"{run}#{i}", i + 1
                ids[ident] = key
                st.runs[key] = RunState(key, domain, float(e["rho"]), float(e["delta"]), e["time"], unit=eu,
                                        key=e["run"])
                continue
            rs = st.runs.get(ids.get(ident, ""))
            if rs is None:
                continue
            if k == "ledger.touch":
                rs.touched = True
            elif k == "ledger.settle":
                rs.settled = (float(e["rho"]), float(e["delta"]))
            elif k == "ledger.release":
                rs.released = True
        return st

    @staticmethod
    def _check_run_id(run: str) -> None:
        """'@' is kept for the replay key `run@unit` of a domain whose units changed, so no run id
        may hold it: two reservations then never share a key (Prop. 1704(d))."""
        if not run or "@" in run:
            raise PrivacyError(f"run id {run!r} may not be empty or contain '@'")

    @staticmethod
    def _find(sd: DomainState, run: str, unit: str | None) -> RunState | None:
        """The reservation of `run` in this account, also one written before the policy changed the
        domain's units: a reservation of a unit, seen from a domain with one unit again, is keyed
        `run@unit`.  Later entries of that run name the reservation's own unit and run id, so the
        log's lifecycle and its verification stay the same across the change (Prop. 1704)."""
        hits = [s for s in sd.runs.values() if s.key == run]
        if len(hits) > 1 and unit is not None:
            hits = [s for s in hits if s.unit == unit] or hits
        if len(hits) > 1:
            raise PrivacyError(f"run {run} reserved for several units before the policy changed the units of "
                               f"domain {sd.domain!r}: name the unit "
                               f"({', '.join(sorted(str(s.unit) for s in hits))})")
        return hits[0] if hits else None

    def _check_caps(self, domain: str, rho: float, delta: float, st: DomainState, now: datetime) -> list[str]:
        caps = self.policy.caps(domain, st.unit)
        if st.unit is not None:
            domain = f"{domain}/{st.unit}"
        r, d = st.charged()
        why = []
        if d + delta >= caps.delta:
            why.append(f"domain {domain!r} would spend delta {d + delta:g}, not below its delta {caps.delta:g}")
        if r + rho > caps.lifetime_rho + _EPS:
            why.append(f"domain {domain!r} would reach rho {r + rho:.6g}, above its lifetime cap {caps.lifetime_rho:.6g} "
                       f"(epsilon {caps.lifetime_epsilon:g} at delta {caps.delta:g})")
        if caps.per_release_rho is not None and rho > caps.per_release_rho + _EPS:
            who = f" for {st.unit}" if st.unit is not None else ""
            why.append(f"this release would spend rho {rho:.6g}{who}, above the per-release cap {caps.per_release_rho:.6g} "
                       f"(epsilon {caps.per_release_epsilon:g} at delta {caps.delta:g})")
        if caps.period is not None:
            pr, _ = st.charged(since=now - caps.period)
            assert caps.period_rho is not None
            if pr + rho > caps.period_rho + _EPS:
                why.append(f"domain {domain!r} would spend rho {pr + rho:.6g} this period, above the period cap "
                           f"{caps.period_rho:.6g} (a throttle, epsilon {caps.period_epsilon:g})")
        return why

    # ------------------------------------------------------------- operations
    @staticmethod
    def _entry(kind: str, domain: str, run: str, unit: str | None, **kw) -> dict:
        e = {"type": kind, "domain": domain, "run": run}
        if unit is not None:
            e["unit"] = unit
        return {**e, **kw}

    def reserve(self, domain: str, run: str, rho: float, delta: float = 0.0, now: datetime | None = None,
                unit: str | None = None) -> int:
        """Reserve the planned budget of a run before it reads data (Prop. 802)."""
        if domain not in self.policy.domains:
            raise PrivacyError(f"domain {domain!r} is not in the organisation policy; a DP run needs a domain")
        if not (rho >= 0 and math.isfinite(rho) and delta >= 0):
            raise PrivacyError(f"bad reservation rho={rho} delta={delta}")
        self._check_run_id(run)
        now = now or datetime.now(timezone.utc)
        with self.reg.locked():
            st = self.state(domain, unit)
            if any(s.key == run for s in st.runs.values()):
                raise PrivacyError(f"run {run} already holds a reservation")
            why = self._check_caps(domain, rho, delta, st, now)
            if why:
                raise PrivacyError("the privacy budget refuses this run:\n  - " + "\n  - ".join(why))
            return self.reg.append(self._entry("ledger.reserve", domain, run, st.unit, rho=dec(rho),
                                               delta=dec(delta, 30)))

    def reserve_units(self, domain: str, run: str, plan: dict, now: datetime | None = None) -> list[int]:
        """Reserve for several units at once ({unit: (rho, delta)}): every unit's
        caps are checked first, so a refusal reserves nothing (Prop. 1704)."""
        now = now or datetime.now(timezone.utc)
        self._check_run_id(run)
        for u, (r, d) in plan.items():           # every figure first: a bad one must not leave others reserved
            if not (r >= 0 and math.isfinite(r) and d >= 0 and math.isfinite(d)):
                raise PrivacyError(f"bad reservation for {u}: rho={r} delta={d}")
        with self.reg.locked():
            why = []
            for u, (r, d) in plan.items():
                st = self.state(domain, u)
                if any(s.key == run for s in st.runs.values()):
                    raise PrivacyError(f"run {run} already holds a reservation for {u}")
                why += self._check_caps(domain, r, d, st, now)
            if why:
                raise PrivacyError("the privacy budget refuses this run:\n  - " + "\n  - ".join(why))
            return [self.reserve(domain, run, r, d, now=now, unit=u) for u, (r, d) in plan.items()]

    def touch(self, domain: str, run: str, unit: str | None = None) -> int:
        """Record that the run is about to read real data: from now on no refund."""
        with self.reg.locked():
            rs = self._find(self.state(domain, unit), run, unit)
            if rs is None:
                raise PrivacyError(f"run {run} has no reservation in domain {domain!r}; reserve before reading data")
            return self.reg.append(self._entry("ledger.touch", domain, rs.key, rs.unit))

    def settle(self, domain: str, run: str, rho: float, delta: float = 0.0, unit: str | None = None) -> int:
        with self.reg.locked():
            st = self._find(self.state(domain, unit), run, unit)
            if st is None:
                raise PrivacyError(f"run {run} has no reservation in domain {domain!r}")
            if st.released or st.settled is not None:
                raise PrivacyError(f"run {run} is already {st.state}")
            if rho > st.reserved_rho + _EPS or delta > st.reserved_delta + _EPS:
                raise PrivacyError(f"run {run} spent rho {rho:.6g}, delta {delta:g}: more than it reserved "
                                   f"(rho {st.reserved_rho:.6g}, delta {st.reserved_delta:g})")
            return self.reg.append(self._entry("ledger.settle", domain, st.key, st.unit, rho=dec(rho),
                                               delta=dec(delta, 30)))

    def release(self, domain: str, run: str, unit: str | None = None) -> int:
        """Refund a reservation; refused once the run has touched data."""
        with self.reg.locked():
            st = self._find(self.state(domain, unit), run, unit)
            if st is None:
                raise PrivacyError(f"run {run} has no reservation in domain {domain!r}")
            if st.touched:
                raise PrivacyError(f"run {run} read real data: its reservation cannot be refunded (Prop. 802)")
            if st.released or st.settled is not None:
                raise PrivacyError(f"run {run} is already {st.state}")
            return self.reg.append(self._entry("ledger.release", domain, st.key, st.unit))

    def nondp(self, domain: str, run: str) -> int:
        return self.reg.append({"type": "ledger.nondp", "domain": domain, "run": run})

    # ------------------------------------------------------------- figures
    def summary(self, domain: str, now: datetime | None = None, unit: str | None = None) -> dict:
        """The domain's (or one unit's) spent, reserved and remaining budget in rho and epsilon."""
        st = self.state(domain, unit)
        caps = self.policy.caps(domain, st.unit)
        r, d = st.charged()
        out = {"domain": domain, **({"unit": st.unit} if st.unit else {}), "delta": caps.delta, "rho": r,
               "delta_spent": d,
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
                     now: datetime | None = None, unit: str | None = None) -> dict:
        """The `ledger` part of the policy input (design sec. 8.2) for a run of this
        domain (and unit); if `run` is already charged it is not counted twice."""
        st = self.state(domain, unit)
        caps = self.policy.caps(domain, st.unit)
        for k in [k for k, v in st.runs.items() if v.key == run]:
            st.runs.pop(k)
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
