# Weekly security patching with Claude

[`security-patch.yml`](../../.github/workflows/security-patch.yml) runs every Tuesday at 05:23 UTC, after Dependabot's weekly run and the Monday scan in `security.yml`, and by hand from `main` (Actions > security-patch > Run workflow; tick *dry run* to only count the findings).

1. **Collect and patch** gathers the open findings: Dependabot alerts (when a token may read them, see below), code-scanning alerts above warning (CodeQL, Semgrep, zizmor), and pip-audit of two resolutions of `pyproject.toml` with all extras: the **lowest** versions its ranges allow (what Dependabot flags for a library: a vulnerable lower bound) and the **newest** ones. [`security_findings.py`](../../.github/scripts/security_findings.py) keeps only identifiers, packages, versions and paths and cuts every text short; findings in paths the agent may not change (workflows, tests, tools, scanner configuration) are counted separately for the maintainer and do not start Claude. When there is a finding, no earlier `security-patch/` pull request of this repository is open, and a Claude credential is set, Claude ([claude-code-action](https://github.com/anthropics/claude-code-action)) makes the smallest fix (a raised version floor, a recompiled hash-pinned file, or the flagged lines), runs ruff, bandit, pip-audit and the unit tests, commits locally and writes the pull request text.
2. **Publish** runs no agent. It applies Claude's commits to the same commit of `main`, refuses the patch if it touches anything under `.github/` except the hash-pinned requirements, scanner or agent configuration, tests, the gate scripts in `tools/` or the `[tool.*]` settings in `pyproject.toml` (a moved file counts under its old path), then pushes a `security-patch/<date>-<run>` branch and opens **one draft pull request** that lists what was fixed, what was not and why, and the checks that ran.

The pull request then follows [CLAUDE.md](../../CLAUDE.md) like any other: CI green, the pre-merge audit, mark ready, CodeRabbit, one `@claude address all critical and warning comments from coderabbit` comment, and a merge by the maintainer. The job never merges, never marks ready, and never pushes outside its own branch.

## What keeps it safe

- **Behaviour does not change.** The prompt allows only the smallest fix: no refactoring, no edits to workflows or tests (the publish job enforces this), no change to synthetic output for a seed. A fix that needs a breaking upgrade or a design change is listed under *Not fixed* instead.
- **Fix, never suppress.** `# nosec` / `# nosemgrep` only for a proven false positive, with its reason (project rule 1).
- **No write token near the agent.** Claude runs with the read-only workflow token (`contents`, `security-events`, `pull-requests`: read) and no `id-token` permission, so even code it runs (tests, package builds) cannot push, open pull requests or mint tokens; the Anthropic credential is scrubbed from its commands where the runner allows. The Dependabot alerts token is revoked before Claude starts. Pushing and the pull request happen in the separate publish job, with a short-lived token of the Claude GitHub App obtained by the same OIDC exchange claude-code-action uses, so the draft triggers the usual CI (one made with the workflow token would not; that is the fallback, with a warning). Claude's tools are an allowlist (local git, pip, uv, ruff, mypy, bandit, pytest and three repository checks) without web fetch or search, and it never reads pull request titles or bodies, which anyone can write.
- **No alert details in public places.** In a public repository code-scanning and Dependabot alerts are visible only to maintainers, but run logs, summaries and artifacts are public, so the job prints only counts per source and the work list never leaves the job that collects it.
- **No duplicate work.** Open Dependabot pull requests are passed to Claude so it does not redo them, and nothing runs while an earlier `security-patch/` pull request of this repository is open (a fork branch with that name does not count).
- **Every action pinned by commit SHA** (zizmor `hash-pin` policy).

Turn it off with the repository variable `SECURITY_PATCH_ENABLED=false`.

## Giving Claude access to Dependabot alerts

The workflow token and the Claude GitHub App cannot read Dependabot alerts (the REST API answers 403): GitHub grants that only through the *Dependabot alerts* permission of a GitHub App of your own or a personal access token. Without one, the job still covers the Python dependencies through pip-audit, which reads PyPI's vulnerability data (the PyPA advisory database, through OSV) and so sees most of the same advisories; the actions and base images are kept current by Dependabot's own pull requests.

| Option | What to set | Pros | Cons |
|---|---|---|---|
| **A. GitHub App of your own (recommended)** | Create an app (Settings > Developer settings > GitHub Apps) with only *Repository permissions > Dependabot alerts: Read-only*, no webhook, install it on this repository only. Variable `SECURITY_ALERTS_APP_CLIENT_ID` (the app's Client ID, on its settings page), secret `SECURITY_ALERTS_APP_KEY` (its private key). | Token minted per run, lives one hour, revoked at the end of the job; not tied to a person; no expiry to renew. | Five minutes of setup. |
| B. Fine-grained personal access token | *Only select repositories*: relsynth; *Repository permissions > Dependabot alerts: Read-only*. Secret `DEPENDABOT_ALERTS_TOKEN`. | Two minutes of setup. | Long-lived, tied to your account, expires (renew it); a leaked token reads your alerts until revoked. |
| C. pip-audit only | Nothing. | No secret at all. | Sees Python dependencies only, not Dependabot's view of actions and images. |
| D. Interactive Claude sessions | Same token as A or B in the Claude Code cloud environment's settings (environment variable), or a GitHub MCP server with that token. | Lets a chat session read the alerts directly. | A token in a general-purpose environment is broader exposure than one job step; prefer A for the scheduled job. |

Also switch on **Settings > Code security > Dependabot security updates**, so Dependabot opens its own pull requests for alerts it can fix by a version bump; the job then fixes only what Dependabot cannot (lower bounds in `pyproject.toml`, hash-pinned scanner files, code-scanning alerts).

## Claude credential

Set one of these as a repository secret (workload identity federation is not offered: it would need `id-token: write` in the job where Claude runs):

- `CLAUDE_CODE_OAUTH_TOKEN`: from `claude setup-token` on a machine signed in to your Claude plan; runs count against that plan.
- `ANTHROPIC_API_KEY`: a Claude API key; runs are billed to the API account.

The Claude GitHub App must be installed on the repository (it already is, for `@claude`).
