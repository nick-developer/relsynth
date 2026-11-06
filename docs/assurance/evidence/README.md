# Evidence files

Aggregate results that claims in the [claim register](../claims.yaml) cite as `report` evidence. They hold scores, rates and intervals only: no rows, no keys and no person labels. Each was copied unchanged from the roadmap report named below, which also has the scripts and settings that produced it.

| File | What it measures | Produced by |
|---|---|---|
| `T06/repro_anonymeter.json` | Singling-out, linkage and inference risks reproduced against Anonymeter, and the bias of its size correction under the null | Roadmap T06, `benchmarks/privacy/reproduce_anonymeter.py`, 2026-10-27 |
| `T06/repro_mia.json` | Shadow-model and DOMIAS membership inference on a raw release and on a population generator | Roadmap T06, `benchmarks/privacy/reproduce_mia.py`, 2026-10-27 |
| `T06/guards_adult.json` | Which planted leaks (copied records, 1% to 20%) the distance guard, the exact-match rate, singling out and DOMIAS detect, on Adult | Roadmap T06, `benchmarks/privacy/calibrate_guards.py`, 2026-10-27 |
| `T07/utility_vs_epsilon.jsonl` | Share of scorecard checks passing for DP runs at ε = 0.5 to 8 on Chinook, Contoso and AdventureWorks DW | Roadmap T07, 2026-10-31 |
| `T07/audit_contoso_eps1.jsonl` | The attack suite against the DP pipeline on Contoso at ε = 1, with the TPR bound ε implies | Roadmap T07, 2026-10-31 |

Datasets: UCI Adult (CC BY 4.0), Chinook (MIT), Contoso Data Generator V2 (MIT, SQLBI) and AdventureWorks DW (MIT, Microsoft sample code), all marked allowed in [docs/datasets.md](../../datasets.md).

A new file belongs here only when a documented claim needs it. Re-running a benchmark replaces the file and may change the documents that quote its numbers; `tools/claims_check.py` then points at every claim whose quote no longer matches.
