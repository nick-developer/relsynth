# Self-tuning: `relsynth autotune`

`relsynth autotune` runs the engine on several seeds, finds the checks that fail, and changes the engine settings
(and, if you allow it, the spec) until your objectives hold. It keeps a history of every run, so a second call goes
on from where the first stopped. When an objective cannot be met, it writes a report that says why.

The background, the literature and the feasibility tests behind this design are in
[research/self-tuning.md](research/self-tuning.md).

```
relsynth autotune spec.yaml [--objectives objectives.yaml] [--runs 40] [--seeds 1,2,3] [--holdout 11,12]
                            [--set KEY=VALUE] [--settings FILE] [--history DIR] [--fresh] [--workers N]
                            [--out spec.tuned.yaml] [--report DIR/report.md]
```

It writes:

- `<spec>.tuned.yaml`: the spec with the tuned settings in its `synthesis` block (and the accepted spec fixes);
- `<spec>.tune/report.md` and `report.json`: the constraints report;
- `<spec>.tune/history.jsonl`: one line per run (a candidate on one seed), aggregates only: check values, limits
  and pass flags, never rows or category values.

Exit code 0 means every objective is met on the search seeds and confirmed on the held-out seeds; 1 means not met
or not confirmed (the report says why); 2 means the spec or the objectives are invalid.

## Objectives

Every key is optional. Without a file, every hard constraint and every statistical family of the scorecard is an
objective.

```yaml
constraints:                     # hard: must hold on every seed
  - referential_integrity        # every foreign key resolves
  - rules                        # no check rule violated beyond the source's own rate
  - unique_keys
  - trees                        # no cycles
  - privacy                      # no privacy check fails (and the attack suite when privacy.attacks is on)
  - domain                       # every identity of domain_checks holds
  - {check: "scorecard:sales|discriminator_auc"}   # any check id (fnmatch pattern) as a hard constraint
targets:                         # statistical: calibrated (see below)
  - all                          # every statistical family of the scorecard (the default)
  - {family: association_max_diff}
  - {scorecard_pass_rate: 0.98}  # share of passed scorecard items, on every seed
seeds: {search: [1, 2, 3], holdout: [11, 12]}
budget: {runs: 40, hours: 6}     # a run is one candidate fitted on one seed
search: auto                     # or a list of setting keys, or {key: {low, high}} / {key: [choices]}
spec_patches: true               # may apply `relsynth doctor` fixes (to the tuned spec only)
quality: false                   # also run the 2.0 quality report; its calibrated checks become targets
privacy: {attacks: false, config: privacy.yaml, review: undecided}
domain_checks: checks.py:checks  # function(real, synthetic) -> {name: (real value, synthetic value)}
null_quantile: 0.99
```

### Hard constraints and statistical targets

A **hard constraint** must hold exactly: none of its checks may fail on any seed. Zero violations on n seeds does
not prove a rate of zero, so the report gives the 95% upper bound of the violation rate (about 3/n) and the number
of seeds you would need for a 5% or 1% bound.

A **statistical check** fails sometimes even when the model is right: each check has a level α, so a perfect model
has about Σα false failures per seed. On AdventureWorks Σα is 4.4, and the chance that every check passes on one
seed is about 1%. So a target of zero failed statistical checks is not reachable. A statistical family is met
when its failures on each seed are at most the `null_quantile` (0.99 by default) of its null count, a Poisson
variable with mean Σα. The report prints the expected false failures and the chance that every check passes.

### Privacy

A privacy check whose confidence interval straddles its limit has status `review`: the test cannot decide. That is
not a fault of the generator, and more seeds or other settings rarely change it. `privacy.review` says how such
a check counts: `undecided` (the default; listed separately and not counted as met or violated), `pass` or
`fail`.

`autotune` refuses a spec in differential-privacy mode. Choosing settings by looking at the real data many times is
itself a query on that data, and the DP guarantee would no longer hold (Papernot and Steinke, 2022).

## How the search works

1. **The spec as given** runs on the search seeds. Its failing checks are classed as *systematic* (they fail on
   more seeds than their level explains: a binomial test at p < 0.01) or *sporadic* (consistent with noise).
   Only systematic failures and hard violations drive the search.
2. **Spec fixes** from `relsynth doctor` are tried first when `spec_patches` is on.
3. **Guided moves.** Each setting in the registry records which checks it affects and in which direction. For each
   failing family the loop steps the linked settings on their scale (log or linear), and tries going back to the
   default for any setting you changed. The history teaches it which settings helped before; those are tried
   first, and a step that helped is expanded, one that overshot is contracted.
4. **Model-based proposals** around the best runs so far, once the guided moves are spent.
5. **Successive halving.** A candidate runs on one seed first and goes on to the other seeds only if it is not
   worse.
6. **Acceptance on paired seeds.** Every candidate runs on the same seeds. A candidate is accepted when it has
   fewer hard violations, or as many and fewer excess statistical failures per seed by at least 0.5 with the
   standard error below the gain.
7. **Confirmation.** The tuned settings and the spec as given are compared on the held-out seeds, which the
   search never saw, so a lucky seed cannot be mistaken for a fix. Budget is reserved for this step.

## The constraints report

The report always comes out, whether the objectives are met or not. For each constraint and family it gives the
evidence and, when it is not met, the reason:

| Reason | Meaning | What to do |
|---|---|---|
| no setting | No tunable setting is linked to the failing checks. | A spec change or an engine change. |
| insensitive | The linked settings were moved and the checks did not change. | The misfit is in the model class: change the spec (`relsynth doctor`) or report it. |
| conflict | Fixing one objective breaks another on the same seeds. | Choose between the objectives; the report lists both. |
| undecided | The privacy test cannot decide (status `review`). | More rows for the attack, or accept the limit. |
| budget | The runs or hours ran out before the search was done. | Run `autotune` again: it continues from the history. |

## Parallel seeds

`--workers N` runs the seeds of one candidate in N processes at once (`autotune(..., workers=N)` in Python). The
records are the same as with one worker, since each run depends only on its candidate and its seed, so the search
takes the same path in less time. Each worker holds one fit in memory (about 2 GB on AdventureWorks). With several
workers, turn successive halving off (`halving: false` in the objectives) so that every seed of a candidate runs at
once.

## On the bench databases (GitHub CI)

`tools/autotune_bench.py --dataset adventureworks` builds the database, writes the objectives (the hard
constraints, the dataset's domain identities from `bench/autotune_domain.py`, every scorecard family) and runs
autotune. The workflow `autotune.yml` runs it on GitHub-hosted runners: start it by hand with the dataset, hours
and seeds, or push a commit whose message contains `[autotune]`. It uploads the history, and `resume_from_run`
goes on from an earlier run's history, since one job may run six hours.

## History and resuming

Runs are keyed by a fingerprint of the spec (without its seed) and the objectives. A second `autotune` with the
same spec and objectives reuses every run it already has and starts from the last accepted settings; change the
spec or the objectives and the old runs stay in the file but are not reused. `--fresh` forgets this fingerprint's
runs.

## Example: Chinook from a bad setting

The CI job `settings-autotune` starts Chinook from two planted settings (`facts.count_model: poisson`,
`segments.default_cap: 1`):

```
relsynth autotune examples/chinook_model/spec.yaml --set source.path=Chinook.sqlite \
    --set synthesis.facts.count_model=poisson --set synthesis.segments.default_cap=1 \
    --runs 24 --seeds 1,2 --holdout 11
```

The spec as given has one systematic statistical failure per seed. The loop tries the doctor's fixes and
`segments.default_cap: 8` (no change), then `facts.count_model: auto`, which removes the failure. The held-out
seed confirms it. That takes 10 runs and about 70 seconds.

## In Python

```python
from relsynth.autotune.loop import autotune
from relsynth.autotune.objectives import parse

res = autotune("spec.yaml", parse({"constraints": ["referential_integrity", "rules"], "budget": {"runs": 30}}))
res.met, res.confirmed, res.delta      # True, True, {'copula.sample_rows': 3000}
res.spec                               # the tuned spec as a dict
print(res.report_markdown)
```
