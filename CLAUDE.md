# Notes for AI coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first: branch names, Conventional Commits, sign-off and the
byte-identity rule apply to agents as they do to people. This file adds the pull-request workflow
that agents (Claude Code and others) must follow, so that CodeRabbit reviews each change once and
review rounds stay short. The project rules below come from the maintainer (2026-11-19) and apply
to every change or addition, by agents and people alike.

## Project rules

1. **Aim: practically every structured dataset.** The engine should accept practically any structured
   dataset it is given and keep its relationships, dependencies, business and technical rules and every
   property README.md promises (keys resolve, rules hold, statistics match). Judge a change by how far it
   widens that coverage without breaking what already works.
2. **Adjustable, with privacy-first defaults.** Users choose the level of privacy, of detail (exact
   distributions or slight deviations from them) and of accuracy through the spec and settings. The
   defaults give the most privacy possible while keeping as much of the data's detail and accuracy as
   data and AI analysis and inference need. A new option comes with a documented default that follows
   this rule; a change of default needs the maintainer's approval (CONTRIBUTING.md, byte identity).
3. **New datasets: spec first, then generic engine changes.** When a new dataset or database gives
   poor synthesis results, first check the specification that describes the source (keys, links, types,
   rules, privacy units, settings) for integrity and completeness, and fix it there if you can. Change the
   engine only when the spec is right and cannot be improved. Engine changes must be generic: they fix a
   kind of data or structure (a distribution shape, a key pattern, a rule type), never one dataset's
   names or values, and never only to raise one benchmark's score. Say in the pull request which kind of
   data the change serves and show it on more than one dataset where the benchmark has one.
4. **Security and code audit before every merge to `main`.** At the final stage of a pull request,
   after the coding is complete and before it is marked ready (step 3 below), run a full audit of the
   change and resolve every finding at **warning level or above** before the pull request is merged:
   - the `ci`, `codeql` and `security` workflows green, which run bandit, pip-audit, CodeQL
     (security-extended and security-and-quality), Semgrep, gitleaks, zizmor and dependency review, and
     no new code-scanning alert of severity warning or higher on the pull request;
   - an agent security review of the whole diff (Claude Code: `/security-review`) and a code review at
     high effort (`/code-review high`), with every finding of warning severity or above fixed;
   - the CodeRabbit round of step 5 below, with every critical, major and warning comment fixed.

   A finding may be declined only when it is shown to be a false positive; the pull request says why.
   List the audit in the pull request body under "Security and code audit": each tool, its result, what
   was fixed and what was declined with the reason. Nothing at warning level or above stays open at merge.

## Pull-request workflow

1. **Open every pull request as a draft.** CodeRabbit skips drafts (`.coderabbit.yaml`,
   `reviews.auto_review.drafts: false`), so work in progress costs no review.
2. **Finish the coding while it is a draft.** Push as often as needed. Run the fast local checks
   for what you changed (ruff, mypy, the tests of the touched modules) before each push.
3. **Mark it ready for review only when** the coding is complete, the audit of project rule 4 is done
   with its findings fixed, and every other required check
   (`ci` always; `codeql` and `security` when the pull request targets `main`, since they run only
   there; and any workflow the change triggers) is green. Fix red checks while still in draft; do
   not mark ready to "see what CodeRabbit says".
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
