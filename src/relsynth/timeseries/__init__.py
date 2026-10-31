# SPDX-License-Identifier: Apache-2.0
"""Time series and event sequences (roadmap T12, release 3.2).

* :mod:`.calendar`  operational time: seasonal and trend intensity per segment.
* :mod:`.events`    the history of each entity's events (inter-arrival, burstiness,
  life span) given its count, as a hierarchical gamma-renewal bridge.
* :mod:`.hooks`     the sequence model inside the engine (``tables.<t>.sequence``).
* :mod:`.series`    regular multivariate series (meters, sensors) as a Bayesian
  dynamic model whose per-series parameters are entity attributes.
* :mod:`.metrics`   autocorrelation, spectra, inter-arrival, seasonality, cross-series
  dependence and forecasting TSTR, with limits calibrated on real-versus-real splits.
* :mod:`.incremental`  extend a synthetic history day by day (CDC-style test data).

See docs/timeseries.md and docs/adr/0004-time-series.md.
"""
from .calendar import Intensity
from .events import SequenceModel

__all__ = ["Intensity", "SequenceModel"]
