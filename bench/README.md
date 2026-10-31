# bench

The benchmark harness: `python -m bench all` runs every suite and compares the
results with `baseline.json`. The full description, the regression rule and how
to change the baseline are in [docs/development/benchmarks.md](../docs/development/benchmarks.md).

| Path | Content |
|---|---|
| `__main__.py` | the command line |
| `datasets.py` | the golden databases: pinned downloads, builders, spec and checks of each |
| `runner.py` | runs relsynth per dataset and seed, writes a history record |
| `assess.py` | the 2.0 quality report and the privacy attacks of one run, as checks |
| `regress.py` | baseline and regression rule (R1–R3) |
| `specs/` | the frozen spec of each database |
| `builders/`, `checks/` | the layer-2 build scripts and domain checks, kept as they were written |
| `history/` | history records (aggregates only), committed with each baseline change |
| `baseline.json` | the current baseline |
| `.cache/` | downloads and built databases (not committed) |

## History record format (schema 1)

```json
{"schema": 1, "relsynth": "1.0.0", "git": {"sha": "...", "branch": "...", "dirty": false},
 "machine": {"node": "...", "platform": "...", "cpu": "...", "cpus": 4, "python": "3.13.0",
             "numpy": "...", "pandas": "...", "scipy": "...", "ci": false},
 "started": "...", "finished": "...", "label": "...", "seeds": [1, 2, 3, 4, 5],
 "suites": ["scorecard", "quality", "privacy", "repro"],
 "runs": [{"dataset": "chinook", "seed": 1, "seconds": 25.6, "rows": {...}, "scorecard": {"passed": 141, "total": 142},
           "cost": {"cpu_seconds": 40.2, "peak_mib": 339.4}, "quality": {"seconds": 55.8, ...},
           "checks": [{"id": "scorecard:<area>|<metric>", "pass": true, "value": 0.03, "limit": 0.1},
                      {"id": "domain:<check>", "pass": null, "value": 2.5, "real": 2.4},
                      {"id": "quality:<check id>", "pass": true, "value": 0.01, "limit": 0.05, "alpha": 0.01},
                      {"id": "area:fidelity", "pass": true, "value": 0.98, "limit": 0},
                      {"id": "goal:referential_integrity", "pass": true, "value": 1.0, "real": 1.0},
                      {"id": "privacy:<criterion>|<attack>", "pass": true, "value": 0.0, "limit": 0.1, "upper": 0.01,
                       "status": "pass"},
                      {"id": "metric:<name>", "pass": null, "value": 0.91, "direction": "higher"}]},
          {"dataset": "bi_corep_lr", "skipped": "RELSYNTH_BENCH_BI_DIR is not set ..."},
          {"dataset": "fake", "seed": null, "checks": [{"id": "science:C00", "pass": true}]}]}
```
