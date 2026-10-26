# Contributing to relsynth

Thank you for helping. relsynth is an open-source project under the [Apache License 2.0](LICENSE). This guide says how a change gets from an idea to a release.

## Ground rules

- Be kind: the [Code of Conduct](CODE_OF_CONDUCT.md) applies everywhere the project lives.
- Report security problems privately, never in a public issue: see [SECURITY.md](SECURITY.md).
- Never commit data you may not redistribute, or synthetic data fitted to it. The [dataset register](docs/datasets.md) says which datasets may be committed, which may only be used for local benchmarks, and which may not be used at all.

## Developer setup

```bash
git clone https://github.com/nick-developer/relsynth && cd relsynth
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests && pytest     # lint, unit and end-to-end tests (a few minutes)
```

The full suite, including the live database servers and the statistical check suite, is described in [docs/testing.md](docs/testing.md).

## Workflow

`main` is protected and always releasable. Nobody pushes to it, maintainers included.

1. Open an issue first for anything larger than a fix, so the design can be agreed before the code.
2. Branch from the latest `main`:

   | Kind of work | Branch name |
   |---|---|
   | Roadmap thread deliverable | `roadmap/<thread id>-<short-name>`, for example `roadmap/T02-ir-schema` |
   | Bug fix | `fix/<issue>-<short-name>` |
   | Anything else | `feat/…`, `docs/…`, `ci/…`, `refactor/…`, `test/…` |

3. Commit in small, reviewable steps (see below), each signed off.
4. Open a pull request to `main` and fill in the template, including the test evidence.
5. CI must be green and a code owner must approve. Maintainers merge with **squash** or **rebase**, never a merge commit, so history stays linear.
6. Delete the branch after the merge.

Never force-push a branch somebody else is working on, and never rewrite `main`.

## Commit messages

relsynth follows [Conventional Commits 1.0](https://www.conventionalcommits.org/en/v1.0.0/):

```
feat(pushdown): fit occupancy facts from GROUP BY statistics

The planner loaded occupancy facts into memory even when every statistic
they need is a GROUP BY count. Prop. 106 shows the counts are sufficient.

Refs #42
Signed-off-by: Ada Lovelace <ada@example.org>
```

- Subject: a type, an optional scope, then an imperative summary of at most 72 characters, no full stop.
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `build`, `chore`, `revert`. A breaking change adds `!` after the type (`feat!:`) and a `BREAKING CHANGE:` footer that says what users must do.
- Body: why the change is needed and what it relies on. The diff already says what changed.

## Developer Certificate of Origin

Every commit must carry a `Signed-off-by:` line whose name and e-mail match the commit author. By adding it you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/): that you wrote the change, or otherwise have the right to submit it under the project's licence. `git commit -s` adds the line. CI rejects a pull request with an unsigned commit; `git rebase --signoff main` fixes a branch.

There is no contributor licence agreement.

## Licence headers

Each Python file starts with

```python
# SPDX-License-Identifier: Apache-2.0
```

(after a shebang or encoding line, if any). `python tools/check_spdx.py` checks every tracked file and `--fix` adds the line. Copyright is held by the contributors and recorded in git history, so files carry no per-file copyright line. Code copied from elsewhere keeps its own SPDX identifier and copyright line and is listed in [NOTICE](NOTICE); its licence must be compatible with Apache-2.0.

## Dependencies

- New runtime dependencies need a reason in the pull request; prefer an optional extra.
- Allowed licences: permissive ones (MIT, BSD, Apache-2.0, ISC, PSF, Zlib, UPL, CC0) everywhere; weak copyleft (LGPL, MPL-2.0) only in optional extras that the user installs; strong copyleft (GPL, AGPL) and unknown licences not at all.
- `python tools/check_licenses.py` enforces this in CI. `pip-audit` checks for known vulnerabilities.

## Tests and proofs

- Every change keeps the existing suites green: unit, end-to-end, live servers, the statistical check suite (`pytest -m science`, C00–C78) and the four real-database regressions (bi/COREP, Chinook, AdventureWorks DW, Contoso). [docs/testing.md](docs/testing.md) says how to run each.
- A bug fix comes with a test that fails without it.
- Every new statistical or privacy claim gets a numbered proposition with a proof in `docs/proofs/` and a check that tests it.
- Base design choices on peer-reviewed work or official standards and cite them. Say so when something is a guess.
- Coverage is reported per module and branch, but there is no global percentage target: review the uncovered lines of the code you touched (Ivanković et al., "Code coverage at Google", ESEC/FSE 2019).

## Changelog

User-visible changes get a line under `## Unreleased` in [CHANGELOG.md](CHANGELOG.md), in the section that fits (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`), following [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/). Internal refactors, CI and test-only changes do not need an entry.

## Versions and releases

relsynth uses [Semantic Versioning 2.0](https://semver.org/). The public API is the CLI (commands, options, JSON output, exit codes), the Python API exported from `relsynth`, and the spec format. Because a new version may change the synthetic draws for the same seed, the release notes say when draws change; that alone is not a breaking change. Maintainers follow the [release checklist](docs/development/release-checklist.md).
