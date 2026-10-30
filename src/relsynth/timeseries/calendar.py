# SPDX-License-Identifier: Apache-2.0
"""Operational time: the calendar intensity of a population of event sequences.

Events of one kind (orders, visits, payments) do not arrive at a constant rate:
there are more in December than in February, more on Mondays than on Sundays,
more at noon than at night, and the level drifts (growth, a regime change).
relsynth separates that calendar from the behaviour of each entity by a time
change.  Let lambda_k(t) be the intensity of events of entities in segment k and

    Lambda_k(t) = integral_{T0}^{t} lambda_k(s) ds

its cumulative intensity.  Measured in operational time u = Lambda_k(t), the
calendar is flat and what is left is the entity's own rhythm (inter-arrival,
burstiness, life span); see relsynth.timeseries.events.  The random time
change theorem (Daley and Vere-Jones 2003, Thm 7.4.I; Brown et al. 2002) says
that a point process with intensity lambda becomes a unit-rate Poisson process
in operational time, so a Poisson population keeps exactly its seasonality and
trend when its operational times are mapped back with Lambda^{-1} (Prop. 1200).

lambda_k is piecewise constant on a grid of cells (a day, or an hour for
datetime columns):

    lambda_k(cell) = P_k[b] * w[dow] / sum_{d in b} w[dow(d)] * h[hour]

* P_k  the share of segment k's events in calendar bucket b (a month by
  default): the trend and the season.  P_k ~ Dirichlet(n_kb + c * p_b),
  the posterior under a Dirichlet prior centred on the pooled profile p with
  strength c (2 by default), as in the 1.0 bucket model.
* w    the day-of-week rate: w_j ~ Gamma(1 + n_j, days_j) (events per day of
  weekday j; Gamma-Poisson posterior), normalised.
* h    the hour-of-day profile (datetime only): h ~ Dirichlet(1 + n_hour).

Within a bucket the bucket's mass is spread over its days by w, and within a
day over its hours by h, so each component keeps its own posterior.  The
normalisation puts the total of each segment at the window length in days, so
one unit of operational time is one average day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_S = 86400.0
FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def to_days(values) -> np.ndarray:
    """Days since 1970-01-01 as floats (NaN for nulls) from dates, datetimes or strings."""
    t = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    s = t.astype("datetime64[s]").astype("int64").to_numpy().astype(float)
    s[t.isna().to_numpy()] = np.nan
    return s / DAY_S


def from_days(days, datetime: bool) -> np.ndarray:
    """datetime64 values from days since 1970-01-01; dates are floored to the day."""
    d = np.asarray(days, dtype=float)
    if datetime:
        sec = np.floor(d * DAY_S)
        out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[s]")
        ok = np.isfinite(sec)
        out[ok] = sec[ok].astype("int64").astype("datetime64[s]")
        return out
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[D]")
    ok = np.isfinite(d)
    out[ok] = np.floor(d[ok]).astype("int64").astype("datetime64[D]")
    return out.astype("datetime64[s]")


class Intensity:
    """Piecewise-constant intensities lambda_k on a calendar grid and their
    cumulative maps Lambda_k (operational time) with inverses.

    Parameters are fitted from event times (days), the segment of each event
    and the observation window [t0, t1) in days.
    """

    def __init__(self, bucket="month", weekly=True, hourly=None, prior_strength=2.0):
        self.bucket = bucket
        self.weekly = weekly
        self.hourly = hourly
        self.c = float(prior_strength)

    # ------------------------------------------------------------ fitting
    def fit(self, t, seg, K, t0, t1, datetime=False):
        t = np.asarray(t, dtype=float)
        seg = np.asarray(seg, dtype=np.int64)
        ok = np.isfinite(t) & (t >= t0) & (t < t1)
        t, seg = t[ok], seg[ok]
        self.K = int(K)
        self.datetime = bool(datetime)
        self.hourly = self.datetime if self.hourly is None else bool(self.hourly and self.datetime)
        self.d0 = int(np.floor(t0))
        self.d1 = int(np.ceil(t1))
        self.t0, self.t1 = float(t0), float(t1)
        days = np.arange(self.d0, self.d1)
        self.ndays = len(days)
        dates = pd.to_datetime(days.astype("int64"), unit="D")
        per = dates.to_period(FREQ[self.bucket])
        self.periods = pd.period_range(per.min(), per.max(), freq=per.freq)
        self.bucket_of_day = (per.asi8 - self.periods[0].ordinal).astype(np.int64)
        self.B = len(self.periods)
        self.dow = ((days + 3) % 7).astype(np.int64)            # 1970-01-01 was a Thursday; Monday = 0
        self.month_of_bucket = np.asarray(self.periods.start_time.month, dtype=np.int64)
        di = np.floor(t).astype(np.int64) - self.d0
        b = self.bucket_of_day[di]
        self.counts = np.zeros((self.K, self.B))
        np.add.at(self.counts, (seg, b), 1.0)
        tot = self.counts.sum(0)
        self.pool = (tot + 1e-3) / (tot + 1e-3).sum()
        self.dow_counts = np.bincount(self.dow[di], minlength=7).astype(float)
        self.dow_days = np.bincount(self.dow, minlength=7).astype(float)
        self.hour_counts = np.bincount(np.floor((t - np.floor(t)) * 24).astype(np.int64).clip(0, 23),
                                       minlength=24).astype(float)
        self.n_events = len(t)
        self.draw(None)
        return self

    def draw(self, rng=None):
        """Parameters from their posterior (rng given) or at their posterior means."""
        a = self.counts + self.c * self.pool + 1e-6
        if rng is None:
            P = a / a.sum(1, keepdims=True)
            w = (1 + self.dow_counts) / np.maximum(self.dow_days, 1)
            h = (1 + self.hour_counts) / (1 + self.hour_counts).sum()
        else:
            P = np.stack([rng.dirichlet(r) for r in a])
            w = rng.gamma(1 + self.dow_counts) / np.maximum(self.dow_days, 1)
            h = rng.dirichlet(1 + self.hour_counts)
        if not self.weekly:
            w = np.ones(7)
        self.P, self.w = P, w / w.mean()
        self.h = h if self.hourly else np.full(24, 1 / 24)
        self._build()
        return self

    def _build(self):
        # day masses: bucket share spread over the bucket's days by weekday rate
        wd = self.w[self.dow]
        den = np.bincount(self.bucket_of_day, weights=wd, minlength=self.B)
        mass = self.P[:, self.bucket_of_day] * (wd / den[self.bucket_of_day])[None, :]
        mass = mass / mass.sum(1, keepdims=True) * self.ndays      # one unit = one average day
        self.day_mass = mass
        if self.hourly:
            mass = (mass[:, :, None] * self.h[None, None, :]).reshape(self.K, -1)
            self.width = 1 / 24
        else:
            self.width = 1.0
        self.cell_mass = np.maximum(mass, 1e-12)
        self.C = np.concatenate([np.zeros((self.K, 1)), np.cumsum(self.cell_mass, 1)], 1)

    # ------------------------------------------------------- extrapolation
    def extend_to(self, t_end):
        """Grow the grid to cover [d0, t_end): future buckets repeat the masses of
        the same calendar bucket in the last observed year (seasonal naive at the
        recent level), so operational time keeps its scale."""
        need = int(np.ceil(t_end)) - self.d1
        if need <= 0:
            return self
        last = self.day_mass
        new_days = np.arange(self.d1, self.d1 + need)
        dates = pd.to_datetime(new_days.astype("int64"), unit="D")
        dow = ((new_days + 3) % 7).astype(np.int64)
        # mean day mass per (calendar month, weekday) over the last 365 observed days
        tail = slice(max(0, self.ndays - 365), self.ndays)
        old = pd.to_datetime(np.arange(self.d0, self.d1)[tail].astype("int64"), unit="D")
        key_old = (np.asarray(old.month) * 7 + self.dow[tail])
        key_new = np.asarray(dates.month) * 7 + dow
        fill = np.zeros((self.K, need))
        lvl = last[:, tail].mean(1)
        for k in range(self.K):
            s = pd.Series(last[k, tail]).groupby(key_old).mean()
            fill[k] = s.reindex(key_new).fillna(lvl[k]).to_numpy()
        self.day_mass = np.concatenate([last, fill], 1)
        self.dow = np.concatenate([self.dow, dow])
        self.ndays += need
        self.d1 += need
        mass = self.day_mass
        if self.hourly:
            mass = (mass[:, :, None] * self.h[None, None, :]).reshape(self.K, -1)
        self.cell_mass = np.maximum(mass, 1e-12)
        self.C = np.concatenate([np.zeros((self.K, 1)), np.cumsum(self.cell_mass, 1)], 1)
        return self

    # ------------------------------------------------------------- maps
    def forward(self, t, seg):
        """Operational time Lambda_seg(t) of times t (days)."""
        t = np.asarray(t, dtype=float)
        seg = np.asarray(seg, dtype=np.int64)
        x = (t - self.d0) / self.width
        n = self.cell_mass.shape[1]
        i = np.clip(np.floor(x), 0, n - 1).astype(np.int64)
        frac = np.clip(x - i, 0, 1)
        return self.C[seg, i] + frac * self.cell_mass[seg, i]

    def inverse(self, u, seg):
        """Times t (days) with Lambda_seg(t) = u."""
        u = np.asarray(u, dtype=float)
        seg = np.asarray(seg, dtype=np.int64)
        out = np.empty(len(u))
        for k in np.unique(seg):
            m = seg == k
            Ck = self.C[k]
            uu = np.clip(u[m], 0, Ck[-1])
            i = np.clip(np.searchsorted(Ck, uu, side="right") - 1, 0, len(Ck) - 2)
            frac = (uu - Ck[i]) / self.cell_mass[k, i]
            out[m] = self.d0 + (i + np.clip(frac, 0, 1)) * self.width
        return out

    def summary(self):
        return {"bucket": self.bucket, "buckets": self.B, "days": self.ndays, "weekly": self.weekly,
                "hourly": self.hourly, "events": int(self.n_events),
                "weekday_rate": [round(float(x), 3) for x in self.w]}
