## What and why

<!-- What does this change do, and why is it needed? Link the issue or roadmap thread (for example "Roadmap T02"). -->

## How it was tested

<!-- Paste the evidence: commands run and their results. Delete lines that do not apply. -->

- [ ] `ruff check src tests` and the unit and end-to-end tests pass
- [ ] Live servers (PostgreSQL, SQL Server, Oracle): `pytest tests/integration`
- [ ] Statistical check suite: `pytest -m science` (C00–C78), result:
- [ ] Real-database regressions (bi/COREP, Chinook, AdventureWorks DW, Contoso), result:

## Statistics and privacy

- [ ] No new statistical or privacy claim, **or** each new claim has a numbered proposition, a proof in `docs/proofs/` and a check
- [ ] Synthetic draws for the same seed are unchanged, **or** the change is explained in CHANGELOG.md

## Checklist

- [ ] Every commit follows Conventional Commits and is signed off (`git commit -s`)
- [ ] CHANGELOG.md has an entry under `Unreleased` (user-visible changes only)
- [ ] Documentation updated (README, `docs/`, generated references)
- [ ] No data, credentials or synthetic data fitted to restricted data is committed (see `docs/datasets.md`)
- [ ] New dependencies have an allowed licence and a reason above
