# Time series and event sequences

relsynth has two models of time data. Both are switched on per table in the spec and leave every other table as
it was. The design is in [ADR 0004](adr/0004-time-series.md), and the proofs are Propositions 1200-1206 in
`docs/proofs` (section "Time series and event sequences").

| Data | Example | Spec key | What is modelled |
|---|---|---|---|
| Events of an entity | a customer's orders, a patient's visits | `sequence` | when each event happens, given how many the entity has |
| Regular readings | a meter's hourly load, a sensor's minute values | `series` | the whole series per entity: level, cycles, autocorrelation, shared shocks |

## Event sequences

```yaml
tables:
  Orders:
    primary_key: OrderKey
    columns: {OrderKey: id, CustomerKey: id, OrderDate: date, Amount: float}
    sequence:
      entity: CustomerKey      # the foreign key of the entity whose history this is
      time: OrderDate          # a date or datetime column, or a lookup fk.col (DateKey.Date)
```

Without `sequence`, every order date is drawn on its own from the seasonal profile of the customer's segment. With
it, the engine still decides how many orders each customer has and what is in them, and then the dates of all of a
customer's orders are redrawn together:

- **calendar**: one intensity per segment from month, weekday and (datetime) hour profiles, read as a rate per active
  entity and matched to the real month totals;
- **life time**: when the entity starts and how much of the rest of its window it stays active, given its number of
  events (a Gaussian copula per count group: 1, 2, 3-4, 5-8, ...);
- **inter-arrival**: the gaps between the first and last event are those of a gamma renewal process, with a shape per
  entity (below 1 bursty, 1 random, above 1 regular).

Options (all optional):

| Option | Default | Meaning |
|---|---|---|
| `window` | `auto` | `auto` reads the entity's window from check rules such as `days(OrderDate) < days(CustomerKey.EndDT)`; `{start: col, end: col}` names columns of the entity; `none` uses the data's time range |
| `bucket` | `month` | calendar bucket of the trend and season: day, week, month, quarter, year |
| `weekly` | `true` | weekday profile |
| `hourly` | `auto` | hour profile (datetime columns only) |
| `shrinkage` | `1.0` | prior scale of a segment's gap parameters around the pooled fit (smaller pools more) |
| `min_entities` | `30` | segments with fewer entities use the pooled parameters |
| `redraws` | `10` | an entity whose new dates break a check rule is redrawn up to this many times, then keeps the engine's dates |

Guarantees: keys, counts, links and the other tables are identical with and without `sequence`, and no check rule
that held before is broken (Prop. 1203). The sample log has `sequences: {table: {entities, events,
redrawn_entities, kept_engine_times}}`.

## Regular series

```yaml
tables:
  Meter: {primary_key: MeterKey, columns: {MeterKey: id, Region: categorical}}
  Load:
    role: fact
    columns: {MeterKey: id, Hour: datetime, kW: float}
    series:
      entity: MeterKey
      time: Hour
      values: [kW]
relationships: [{child: Load, child_key: MeterKey, parent: Meter}]
```

A series table is generated after the relational model, so nothing may reference it. Each synthetic entity row
gets one full series on the real grid (step inferred from the data). Per value column, `log(1 + y)` is split into a
level, daily, weekly and (one year of data or more) yearly profiles, and a residual with AR(`ar`) noise, a seasonal
AR at one day, and a common factor shared by all series (AR(`factor_ar`)). Entity parameters are drawn from a
Bayesian Gaussian mixture and kept inside the real range. Options: `ar` (2), `factor_ar` (2), `day`, `week` (true),
`year` (`auto`).

## Incremental runs

`relsynth extend` adds the events after a dataset, period by period, as a feed would:

```bash
relsynth extend spec.yaml --until 2025-03-01 --step day --out deltas/   # deltas/delta-2025-01-01/Orders.csv, ...
```

Each entity continues its own renewal process: its rate is updated by its base history, and the first new order
takes into account how long ago its last one was. Random numbers are tied to (seed, entity, event number), so
running day by day gives the same rows as one run to the end date, whatever the step (Prop. 1206). New rows copy
the attributes of an event of the same segment, with new keys, shifted dates and the rules checked. The Python API
is `relsynth.extend(model, until, base=..., start=..., step=...)`.

## TimescaleDB

The PostgreSQL connector reads TimescaleDB like PostgreSQL; `introspect` notes each hypertable and its time
column. A PostgreSQL target with the TimescaleDB extension creates hypertables on load
(`create_hypertable(..., if_not_exists => TRUE)`) for the source's hypertables and for every table whose
`series`, `sequence` or fact time column is a date or datetime. `target.timescale.hypertables` can be `none` or an
explicit `{table: time column}` map, and `target.timescale.chunk_interval` sets the chunk size (e.g. `7 days`).
Tables whose key does not include the time column stay plain tables (TimescaleDB requires the time column in every
unique index). See [connectors](connectors.md#timescaledb).

## Metrics

`relsynth.timeseries.metrics` compares real and synthetic data:

- sequences (`sequence_report(spec, real, synthetic)`): gap Wasserstein and KS, same-day share, burstiness, gap
  autocorrelation, active span, first event, weekday, month, trend and hour profiles, counts;
- series (`series_checks(Y_real, Y_syn, step)`): mean ACF, log spectral density (Welch), cross-series correlation,
  values, and forecasting TSTR: a ridge day-ahead model trained on synthetic series and tested on the last 20% of
  the real ones, against the same model trained on real data (gap = TSTR / TRTR - 1).

Limits are calibrated at 99% on splits of the real entities (Prop. 1204). For relsynth's datasets, which are
posterior predictive draws, `predictive=True` (the default of `sequence_report`) adds the parameter uncertainty.
Series limits use a weekly block bootstrap so that shocks shared by all series (weather, holidays) count as the
variation they are.

The benchmark runner is `benchmarks/timeseries/run_timeseries.py` (UCI electricity load and the sequence tables of
the real databases).
