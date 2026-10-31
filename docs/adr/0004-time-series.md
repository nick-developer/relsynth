# ADR 0004: Time series and event sequences

- Status: proposed (roadmap thread T12, release 3.2)
- Date: 2026-10-30

## Context

relsynth 1.0 draws the date of every fact row independently from a seasonal profile of its segment. The calendar
(month, weekday) is right, but the history of an entity is not: a customer's three orders are three independent
dates, so inter-arrival times, burstiness, the active life span and churn are lost. Regular measurements (a meter's
hourly load, a sensor's readings) have no model at all; a table of them is a fact table whose readings are drawn as
independent attributes, which loses autocorrelation, daily and weekly cycles and the dependence between series.

The roadmap (T12) asks for both kinds of time data, for TimescaleDB as a source and target, for metrics that
measure them, for an incremental (day-by-day) mode, and for a choice between three families of models.

## Options

1. **State-space models** (structural time series, Harvey 1989; Durbin and Koopman 2012): level, trend and seasonal
   components with Gaussian or count observations. Exact Kalman filtering, interpretable components, cheap posterior
   draws. Fits regular series well. For irregular event times they need a discretisation (counts per day), which
   loses the order and spacing of events within an entity.
2. **Point processes**: renewal processes, Cox processes and Hawkes processes (Hawkes 1971; Daley and Vere-Jones
   2003). They model event times directly. A renewal process with gamma gaps covers bursty (shape < 1), Poisson
   (shape 1) and regular (shape > 1) entities. Hawkes self-excitation explains bursts by past events. Calendar
   effects enter by a time change (random time change theorem). Likelihoods are explicit and Bayesian inference is
   standard.
3. **Neural sequence generators**: TimeGAN (Yoon, Jarrett and van der Schaar 2019), DoppelGANger (Lin et al. 2020),
   diffusion models for time series (Tashiro et al. 2021). They are flexible and good at long multivariate series.
   On the other hand they need a GPU and a lot of data per pattern, they have no calibrated posterior, and they are
   hard to make respect relational rules and key structure. They also memorise rare sequences unless trained with
   DP-SGD, which costs accuracy (Lin et al. 2020 report this trade-off).

## Decision

1. **Event sequences use a point-process model**: operational time per segment (piecewise-constant intensity from
   month, weekday and hour profiles; Prop. 1200), a gamma renewal bridge for the gaps given the first and last event
   (exact, Prop. 1201) with an entity random effect on the shape, and a life-time copula for when an entity starts
   and how long it stays active, given its count. The engine keeps deciding *how many* events an entity has and
   *what* each holds. The sequence model only redraws *when*, as a hook after the table is finished, so keys,
   counts, links, rules and the other tables are unchanged (Prop. 1203). Hawkes excitation is not used: given the
   counts, the renewal bridge with a random shape already reproduces burstiness on all real tables tried, and a
   Hawkes bridge has no closed form.
2. **Regular series use a state-space-style decomposition** fitted per entity: level, daily, weekly and yearly
   profiles (principal component scores), and an AR plus seasonal-AR residual with a common factor for the
   dependence between series. Entity parameters come from a Bayesian Gaussian mixture. This keeps every draw
   stationary and inside the real range (Prop. 1205) and runs on a CPU.
3. **Neural generators stay optional plug-ins** (T11, gated): the generator contract of ADR 0002 can host a
   TimeGAN- or diffusion-style series generator later, behind the same metrics.
4. **Incremental mode** continues each entity's renewal process from the base dataset with counter-based random
   numbers, so a day-by-day run equals one long run (Prop. 1206).
5. **TimescaleDB** is the PostgreSQL connector with hypertable detection (source) and `create_hypertable` on load
   (target). InfluxDB and other time-series stores are left for the user to decide (T12 report).
6. **Metrics**: inter-arrival (gap Wasserstein and KS, burstiness, gap autocorrelation), life time (span, first
   event), calendar (weekday, month, trend, hour) and counts for sequences; ACF, log spectral density, cross-series
   correlation and values for series; forecasting TSTR (train on synthetic, test on real). Limits are calibrated on
   real-versus-real splits (Prop. 1204), with a predictive correction for posterior draws and a block bootstrap for
   shared shocks.

## Consequences

- Specs without `sequence` or `series` produce the same output as before.
- The sequence model costs a fit per sequence table (Contoso Orders: about 3 minutes more on 4 cores, mostly the
  moment calibration of the calendar).
- A neural series generator can be added later without changing the spec keys or the metrics.
- The count law of each entity stays the engine's. Where it is off (a heavier tail of active customers), the
  inter-arrival metrics inherit the error, and it should be fixed in the count model.

## References

Brown, Barbieri, Ventura, Kass and Frank (2002); Daley and Vere-Jones (2003); Durbin and Koopman (2012); Harvey
(1989); Hawkes (1971); Künsch (1989); Lin, Jain, Wang, Fanti and Sekar (2020); Lukacs (1955); Rubin (1981);
Tashiro, Song, Song and Ermon (2021); Yoon, Jarrett and van der Schaar (2019).
