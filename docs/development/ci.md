# Continuous integration

Every pull request and every push to `main` runs [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). The job names below are the **required status checks** of the `main` ruleset ([repository settings](repository-settings.md)), so a change merges only when all of them pass.

| Job | What it proves | Time |
|---|---|---|
| `lint` | ruff (style, bugs and the `S` security rules), bandit, pip-audit on the installed dependency closure, SPDX header on every Python file, dependency licences (report in the job summary), one version everywhere (`tools/release.py check`: package, README, docs, example pipelines, images, Helm chart, OpenAPI) | 1 min |
| `proofs` | `tools/check_proofs.py` (every `sec_*.tex` included, no proposition number claimed twice), then the PDF built with latexmk and no undefined or duplicate references; the PDF is an artifact | 3 min |
| `types` | mypy reports no error outside `tools/mypy-baseline.txt` (the ratchet below) | 1 min |
| `dco` | every commit of the pull request is signed off by its author | seconds |
| `unit (os, python)` | `tests/unit` on Python 3.10–3.13 (Linux) and 3.12 on Windows and macOS, with branch coverage; the Linux 3.12 job writes coverage per module to the job summary | 3–6 min |
| `e2e` | every CLI command as a pipeline runs it (`tests/e2e`) | 2 min |
| `integration` | live PostgreSQL 16, SQL Server 2022 and Oracle 23 Free in service containers (`tests/integration`) | 4 min |
| `science (i/4)` | the statistical check suite (C00–C82, C600s, C800s, C900s, C1200s, C1600s) in four shards of about equal run time | about 15 min each |
| `agents` | the review agent on Chinook, AdventureWorks DW and Contoso with the catalog hidden: the rules backend must reach accuracy 0.95 against the declared catalog with at most one wrong answer not flagged for a person (`python -m bench.agents`); then a small local model under Ollama is measured, not gated ([agents](../agents.md#evaluation)) | 5 min |
| `build` | sdist and wheel, `twine check`, a clean-environment install with every extra, and a CycloneDX SBOM of that install | 1 min |
| `sign` | dry run of the release signature: the wheel, sdist and SBOM are signed with Sigstore under this workflow's identity and verified (pull requests from this repository and pushes; not on forks) | 1 min |

The suite also runs weekly, so that new dependency releases and new advisories in pip-audit are caught when nothing changed in the repository. Actions are pinned to commit SHAs, and Dependabot ([`.github/dependabot.yml`](../../.github/dependabot.yml)) proposes updates weekly. Code scanning runs in two more workflows; see [Security scanning](#security-scanning).

More workflows run outside the required gate:
- [`service-e2e.yml`](../../.github/workflows/service-e2e.yml), when the service, the Helm chart or the images change, weekly and by hand: the service end to end on docker compose and on a real Kubernetes cluster (kind), with SSO, a run, approval, download, audit, tenant isolation, `helm test`, Pod Security `restricted`, and a worker lost mid-run ([deployment](../service/deployment.md)).
- [`mutation.yml`](../../.github/workflows/mutation.yml), weekly: the sampled mutation score of the core engine ([testing](../testing.md#mutation-testing)).
- [`pypi_upload.yml`](../../.github/workflows/pypi_upload.yml), on a published release: the whole gate, then build, SBOM, signatures, provenance and the upload ([supply chain](supply-chain.md)).

## Security scanning

Every finding goes to Security > Code scanning (SARIF) and annotates the pull request.

| Workflow / job | What it checks | Fails the pull request on |
|---|---|---|
| `codeql` (`codeql.yml`) | CodeQL for Python, GitHub Actions and the UI's JavaScript with the `security-extended` and `security-and-quality` suites ([config](../../.github/codeql/codeql-config.yml); tests are left out) | alerts at or above the severity set in the branch rule |
| `security / semgrep` | Semgrep rule packs `p/python`, `p/security-audit`, `p/secrets`, `p/dockerfile`, `p/github-actions` | new ERROR findings only (`--baseline-commit`), so older findings are triaged in code scanning |
| `security / gitleaks` | secrets in the commits of the pull request or push; the whole history on the weekly run ([allowlist](../../.gitleaks.toml): test fixtures and the documented demo credentials) | any secret |
| `security / zizmor` | workflow security: template injection, unpinned actions, credential persistence, permissions ([policy](../../.github/zizmor.yml)) | high-severity, high-confidence findings |
| `security / dependency-review` | dependencies the pull request adds or changes | a known vulnerability of high or critical severity |
| `lint` (ci.yml) | ruff `S` rules, bandit, pip-audit | any finding |

`codeql.yml` is an advanced setup: the repository's CodeQL default setup must be switched off (Settings > Code security > CodeQL analysis > Switch to advanced), or GitHub rejects the workflow's results. All five run weekly as well.

[CodeRabbit](https://coderabbit.ai) reviews pull requests with the instructions in [`.coderabbit.yaml`](../../.coderabbit.yaml) (tenant isolation and secrets in the service, privacy-critical code against the proofs and the claim register, workflow hardening) once its GitHub App is installed on the repository. It comments; it does not approve or merge.

## The type ratchet

relsynth 1.0 was written without a type checker; mypy reports 77 errors on it, nearly all attributes set dynamically on model objects. Fixing them at once would touch the engine for no change in behaviour. Instead, `tools/mypy_ratchet.py` fails when mypy reports an error (file, code and message, ignoring line numbers) that the baseline does not hold. A change that fixes errors runs `python tools/mypy_ratchet.py --update` and commits the smaller baseline.

## Coverage

Coverage is measured on statements and branches and shown per module; there is no global percentage gate. Following Ivanković et al. ("Code coverage at Google", ESEC/FSE 2019), the useful signal is whether the lines a change touches are covered, which reviewers read in the per-module table, and a single target invites tests written to the number rather than to the behaviour. The statistical suite, which tests the engine most thoroughly, runs in a subprocess and is not part of the coverage figure.

## Running the gate locally

```bash
pip install -e ".[dev]" datasketches
ruff check src tests tools && bandit -q -c pyproject.toml -r src && pip-audit --skip-editable
python tools/check_spdx.py && python tools/check_licenses.py && python tools/mypy_ratchet.py
pytest tests/unit tests/e2e
docker compose -f docker-compose.test.yml up -d --wait && pytest tests/integration   # see docs/testing.md for the URLs
pytest -m science                                                                     # or RELSYNTH_SCIENCE_SHARD=1/4
```
