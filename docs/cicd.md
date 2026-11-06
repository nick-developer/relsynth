# CI/CD

relsynth is built to run unattended: every command has a JSON mode and a meaningful exit code, credentials come from the environment, and a quality gate turns a drop in fidelity into a failed job.

## Typical jobs

| Job | Trigger | Commands |
|---|---|---|
| Refresh a synthetic test database | Nightly schedule | `relsynth run spec.yaml --target-url-env TEST_DB_URL --fail-under 0.98` |
| Validate a spec change | Change to `spec.yaml` | `relsynth check spec.yaml --connect`, then `relsynth run spec.yaml --no-write --fail-under 0.98` |
| Publish a shareable extract | Release, or on demand | `relsynth run spec.yaml --target csv:out/ --datasets 3` |
| Publish the model, not the data | Release | `relsynth export-model spec.yaml --out model/` |
| Watch the source for drift | Weekly | `relsynth verify spec.yaml --json` (model checks R1–R7), `relsynth profile --source ...` |
| Seed a database for integration tests | Each test run | `relsynth run spec.yaml --target "postgresql://..." --scale 0.05 --set synthesis.posterior.mode=mean` |

Ready-made pipelines:

| System | File |
|---|---|
| GitHub Actions | [examples/ci/github-actions.yml](../examples/ci/github-actions.yml) |
| GitLab CI | [examples/ci/gitlab-ci.yml](../examples/ci/gitlab-ci.yml) |
| Azure DevOps | [examples/ci/azure-pipelines.yml](../examples/ci/azure-pipelines.yml) |
| Jenkins (Docker agent) | [examples/ci/Jenkinsfile](../examples/ci/Jenkinsfile) |

## Exit codes

| Code | Meaning | Typical reaction |
|---|---|---|
| 0 | Success | |
| 1 | Quality gate failed: the share of passed scorecard checks is below `--fail-under` | Run `relsynth doctor` on the bundle, and publish its report |
| 2 | Invalid spec or arguments, including a spec that is not valid YAML | Fix the spec; nothing was read or written |
| 3 | Connection, permission or file error (also a target table that exists with `if_exists: fail`) | Check secrets, network and grants |
| 4 | Engine failure while fitting or sampling | Open an issue with the diagnostics bundle |
| 130 | Interrupted | |

`relsynth run` still writes its diagnostics bundle when a step fails, so a job can always publish it.

## Machine-readable output

`--json` prints exactly one JSON document on stdout. Progress and log lines go to stderr. Every document has `command`, `version` and `exit_code`; an error adds `error` (password masked) and `error_type`. For `run`:

```json
{
 "command": "run",
 "version": "1.0.0",
 "bundle": "diagnostics_bundle.json",
 "seconds": 4.9,
 "rows": {"customer": 1500, "product": 300, "sales": 14887},
 "pushdown": {"mode": "auto", "sample_rows": 500000, "pushed": {}, "loaded": ["customer", "product", "sales"], "reasons": {}},
 "errors": [],
 "written": [{"kind": "csv", "path": "/work/out"}],
 "scorecard": {"passed": 36, "total": 36, "score": 1.0, "failed": []},
 "gate": {"fail_under": 0.98, "passed": true},
 "exit_code": 0
}
```

Read it with `jq`, for example `jq -r '.scorecard.score' run.json`.

## Secrets

- Store passwords as masked, protected CI variables. Map them to the variables the spec names (`password_env: PGPASSWORD`), or to `--source-password-env` and `--target-url-env`.
- Never write a password into `spec.yaml`: a spec saved by relsynth has its URLs masked with `***` and reads them from `url_env`.
- Error messages, JSON output, logs and bundles mask passwords, so job logs can be public within your organisation.
- Use a read-only account for the source. relsynth only issues `SELECT` against a source.
- For the target, use an account that can create and drop tables in one schema (`if_exists: replace`), or only insert and delete (`structure: existing` with `if_exists: truncate`).

## Reproducibility

- `synthesis.seed` fixes everything. The same spec, data, seed and relsynth version give the same synthetic database, on any of the supported databases (rows are put in a canonical order on load).
- `--datasets N` writes N independent posterior draws. Dataset 0 goes to the target; dataset *i* goes to `<path>_<i+1>` for folders and SQLite files, or to schema `<schema>_<i+1>` for servers. Dataset *m* is the same in every run with the same seed.
- `--set synthesis.posterior.mode=mean` uses posterior means instead of a draw. The output is then stable across small changes to the source, which suits fixtures for integration tests.
- Pin the version in CI (`relsynth==2.0.0`), because a new version may change draws even when the model is the same.

## Containers

The [Dockerfile](../Dockerfile) builds an image with every driver, including ODBC Driver 18. It runs as an unprivileged user, with the working folder at `/work`:

```bash
docker build -t relsynth:2.0.0 .
docker run --rm -v "$PWD:/work" -e PGPASSWORD -e TEST_DB_URL relsynth:2.0.0 \
  run /work/spec.yaml --target-url-env TEST_DB_URL --fail-under 0.98 --json
```

## Run time and resources

- A run fits the model once and samples once per dataset. Fitting dominates, and it grows with the number of entity rows and links rather than with fact rows when large facts are pushed down (see [pushdown.md](pushdown.md)).
- For a nightly job, set the runner's memory to about twice the size of the loaded tables, and use `synthesis.pushdown` for facts above about a million rows.
- `--no-privacy` skips the slowest part of evaluation on wide tables. Keep it on for data that leaves the organisation.
- `relsynth plan spec.yaml` shows, without fitting, which tables will be loaded and which pushed down.

## Quality gates

`--fail-under 0.98` fails the job when fewer than 98% of scorecard checks pass. Every check reports its significance level `alpha` under the null that the synthetic data is a correct posterior predictive draw. Their sum, `expected_false_failures` in the bundle, is the number of checks expected to fail by chance alone. A gate at `1.0` is therefore too strict for large schemas; set it a little below `1 - expected_false_failures / total`. A typical value is 0.97–0.99.

In Python:

```python
import relsynth as rs
result = rs.run("spec.yaml", out="csv:out/", fail_under=0.98)   # raises QualityGateError (exit code 1 in the CLI)
```
