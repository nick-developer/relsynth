# Notes for AI coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first: branch names, Conventional Commits, sign-off and the
byte-identity rule apply to agents as they do to people. This file adds the project rules that
every change must follow, and the pull-request workflow that agents (Claude Code and others) must
follow, so that CodeRabbit reviews each change once and review rounds stay short.

## Project rules for every change

These four rules are set by the project owner and are **mandatory** for every change or addition,
by people and agents alike. A pull request that breaks one is not ready to merge.

### 1. Pre-merge security and code audit

At the final stage of every change, before its pull request is merged to `main`, run a detailed
security and code audit against the code, and resolve **every finding above "warning"** before the
merge. The audit has two parts:

- **The CI scanners, all green on the final head commit:** the `ci` lint job (ruff with the `S`
  security rules, bandit, pip-audit), `codeql` (security-extended and security-and-quality
  suites), and `security` (Semgrep with the project rules in `.semgrep/`, gitleaks, zizmor,
  dependency review). `codeql` and `security` run only on pull requests to `main`; for a stacked
  pull request, run the audit when it is retargeted to `main`.
- **A review of the full diff against `main`:** run `/security-review` (Claude Code) on the branch,
  plus `/code-review` for correctness, and read the CodeQL and Semgrep alerts the pull request
  adds under Security > Code scanning.

"Above warning" means: CodeQL `error` or security severity high/critical; Semgrep `ERROR`;
bandit `HIGH` severity; any pip-audit or dependency-review vulnerability with a fixed version;
any gitleaks secret; zizmor high; and any `/security-review` or `/code-review` finding rated
high, critical or important. These are fixed, not suppressed: a `# nosec` or `# nosemgrep` is
allowed only for a proven false positive and carries its reason. Warnings and lower may be
declined with a one-line reason. List the audit result (tools run, findings fixed, findings
declined and why) in the pull request description before it is marked ready, and run the audit
again if commits land after that.

### 2. New datasets: check the spec first, then fix the engine generically

When a new dataset or database is added and synthesis shows problems, first check the
specification file that describes the source: is it complete and correct (tables, keys,
relationships, types, rules, per-table settings)? Fix the spec where it is wrong or incomplete.
Only when the spec is sound, or editing it cannot fix the problem, analyse the engine. Engine
changes **must be generic**: they must apply to a type of data or a class of datasets, never only
to the dataset under test, and never be tuned to improve that dataset's scores. Each engine change
names the class of data it addresses and is tested on more than the dataset that revealed it.

### 3. Aim: support practically every structured dataset

The overall aim is an engine that supports practically every structured dataset a user can provide,
while keeping its relationships, dependencies, business and technical rules, and every
specification listed in [README.md](README.md). Prefer changes that widen what the engine handles
correctly over changes that work around one schema; a source the engine cannot yet handle should
fail with a clear error, not produce silently wrong data.

### 4. User-adjustable, with maximum privacy by default

The engine must be adjustable to each user's needs: the user decides the level of privacy, the level
of detail (exact distributions or slight deviations from them) and the accuracy, through the spec,
settings and command-line options, without code changes. The **defaults** aim for the maximum
privacy possible while keeping the underlying data's details and accuracy as far as possible for
data analysis, AI training and inference. A change that alters a default states in its pull request
how it affects privacy and fidelity, with evidence (privacy checks and quality report).

## Pull-request workflow

1. **Open every pull request as a draft.** CodeRabbit skips drafts (`.coderabbit.yaml`,
   `reviews.auto_review.drafts: false`), so work in progress costs no review.
2. **Finish the coding while it is a draft.** Push as often as needed. Run the fast local checks
   for what you changed (ruff, mypy, the tests of the touched modules) before each push.
3. **Mark it ready for review only when** the coding is complete and every other required check
   (`ci` always; `codeql` and `security` when the pull request targets `main`, since they run only
   there; and any workflow the change triggers) is green, and the pre-merge audit of rule 1 is
   done with every finding above warning resolved and its result in the pull-request
   description. Fix red checks while still in draft; do not mark ready to "see what CodeRabbit
   says".
4. **Wait for CodeRabbit to finish.** Marking a pull request to `main` ready starts one full review.
   CodeRabbit reviews only pull requests to `main` on its own (`base_branches`), so for a stacked
   pull request, or if no review has started a few minutes after marking ready, post one top-level
   `@coderabbitai review` comment instead. The review is finished when the `CodeRabbit` commit
   status is no longer pending and its review (walkthrough plus inline comments) is posted. Do not
   poll in a loop: let the PR activity wake you, or check once after a few minutes.
5. **Leave one top-level comment** on the pull request, and nothing else:

   ```
   @claude address all critical and warning comments from coderabbit
   ```

   Then work through them in **one batch**: every CodeRabbit comment marked critical, major,
   "Potential issue" or warning, and every failing CodeRabbit pre-merge check. Minor comments and
   nitpicks are optional; take them only when they are cheap and clearly right. Push the fixes as
   one commit (or a few, by topic), not one commit per comment.
6. **Never reply to individual CodeRabbit inline comments** and never tag `@claude` or
   `@coderabbitai` on an inline thread. Do not add more top-level comments either.
7. **Record the outcome in the fix commit.** Its message lists what was fixed and which findings
   were declined, with a one-line reason each, so the pull request keeps a single agent comment.
8. **No automatic re-review.** Incremental reviews are off
   (`reviews.auto_review.auto_incremental_review: false`), so the fix push is not reviewed again.
   Ask for a second pass with one top-level `@coderabbitai review` comment only when the fixes were
   large or touched security or privacy code. Do not run more than two CodeRabbit rounds on a PR;
   leave anything left over to the code owner.

## Keeping runs short

- Re-run only the checks that failed while fixing them; run the full suite once before marking ready.
- Never close and reopen a pull request, push empty commits, or skip a test to get a green check.
- Stacked pull requests are merged by the maintainer by hand; an agent never merges.
