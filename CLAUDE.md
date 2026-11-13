# Notes for AI coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first: branch names, Conventional Commits, sign-off and the
byte-identity rule apply to agents as they do to people. This file adds the pull-request workflow
that agents (Claude Code and others) must follow, so that CodeRabbit reviews each change once and
review rounds stay short.

## Pull-request workflow

1. **Open every pull request as a draft.** CodeRabbit skips drafts (`.coderabbit.yaml`,
   `reviews.auto_review.drafts: false`), so work in progress costs no review.
2. **Finish the coding while it is a draft.** Push as often as needed. Run the fast local checks
   for what you changed (ruff, mypy, the tests of the touched modules) before each push.
3. **Mark it ready for review only when** the coding is complete and every other required check
   (`ci`, `codeql`, `security`, and any workflow the change triggers) is green. Fix red checks while
   still in draft; do not mark ready to "see what CodeRabbit says".
4. **Wait for CodeRabbit to finish.** Marking ready starts one full review. It is finished when the
   `CodeRabbit` commit status is no longer pending and its review (walkthrough plus inline comments)
   is posted. Do not poll in a loop: let the PR activity wake you, or check once after a few minutes.
5. **Leave one top-level comment** on the pull request, and nothing else:

   ```
   @claude address all critical and warning comments from coderabbit
   ```

   Then work through them in **one batch**: every CodeRabbit comment marked critical, major,
   "Potential issue" or warning, and every failing CodeRabbit pre-merge check. Minor comments and
   nitpicks are optional; take them only when they are cheap and clearly right. Push the fixes as
   one commit (or a few, by topic), not one commit per comment.
6. **Never reply to individual CodeRabbit inline comments** and never tag `@claude` or
   `@coderabbitai` on an inline thread. If a suggestion is wrong, say so once, in the summary below.
7. **Summarise once.** After the batch is pushed, edit or add a single top-level comment listing
   what was fixed and which findings were declined, with a one-line reason each.
8. **No automatic re-review.** Incremental reviews are off
   (`reviews.auto_review.auto_incremental_review: false`), so the fix push is not reviewed again.
   Ask for a second pass with one top-level `@coderabbitai review` comment only when the fixes were
   large or touched security or privacy code. Do not run more than two CodeRabbit rounds on a PR;
   leave anything left over to the code owner.

## Keeping runs short

- Re-run only the checks that failed while fixing them; run the full suite once before marking ready.
- Never close and reopen a pull request, push empty commits, or skip a test to get a green check.
- Stacked pull requests are merged by the maintainer by hand; an agent never merges.
