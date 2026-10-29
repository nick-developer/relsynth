# bench

The benchmark harness: `python -m bench all` runs every suite and compares the
results with `baseline.json`. The full description, the regression rule and how
to change the baseline are in [docs/development/benchmarks.md](../docs/development/benchmarks.md).

| Path | Content |
|---|---|
| `__main__.py` | the command line |
| `datasets.py` | the golden databases: pinned downloads, builders, spec and checks of each |
| `runner.py` | runs relsynth per dataset and seed, writes a history record |
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
 "runs": [{"dataset": "chinook", "seed": 1, "seconds": 25.6, "rows": {...}, "scorecard": {"passed": 141, "total": 142},
           "checks": [{"id": "scorecard:<area>|<metric>", "pass": true, "value": 0.03, "limit": 0.1},
                      {"id": "domain:<check>", "pass": null, "value": 2.5, "real": 2.4}]},
          {"dataset": "bi_corep_lr", "skipped": "RELSYNTH_BENCH_BI_DIR is not set ..."},
          {"dataset": "fake", "seed": null, "checks": [{"id": "science:C00", "pass": true}]}]}
```
