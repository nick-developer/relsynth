# Weekly security patching with Claude

[`security-patch.yml`](../../.github/workflows/security-patch.yml) runs every Tuesday at 05:23 UTC, after Dependabot's weekly run and the Monday scan in `security.yml`, and by hand (Actions > security-patch > Run workflow; tick *dry run* to only list the findings).

1. **collect** gathers the open findings with read-only tokens: Dependabot alerts (when a token may read them, see below), code-scanning alerts above warning (CodeQL, Semgrep, zizmor), and pip-audit of two resolutions of `pyproject.toml` with all extras: the **lowest** versions its ranges allow (what Dependabot flags for a library: a vulnerable lower bound) and the **newest** ones. [`security_findings.py`](../../.github/scripts/security_findings.py) keeps only identifiers, packages, versions and paths, cuts every text short and writes the list to the job summary.
2. **patch** runs only when there is a finding, no earlier `security-patch/` pull request is still open, and a Claude credential is set. Claude ([claude-code-action](https://github.com/anthropics/claude-code-action)) makes the smallest fix (a raised version floor, a recompiled hash-pinned file, or the flagged lines), runs ruff, bandit, pip-audit and the unit tests, and opens **one draft pull request** from a `security-patch/<date>` branch that lists what was fixed, what was not and why, and the checks it ran.

The pull request then follows [CLAUDE.md](../../CLAUDE.md) like any other: CI green, the pre-merge audit, mark ready, CodeRabbit, one `@claude address all critical and warning comments from coderabbit` comment, and a merge by the maintainer. The job never merges, never marks ready, and never pushes outside its own branch.

## What keeps it safe

- **Behaviour does not change.** The prompt allows only the smallest fix: no refactoring, no edits to workflows or tests, no change to synthetic output for a seed. A fix that needs a breaking upgrade or a design change is listed under *Not fixed* instead.
- **Fix, never suppress.** `# nosec` / `# nosemgrep` only for a proven false positive, with its reason (project rule 1).
- **Least privilege.** `collect` has read permissions only. `patch` has `contents: read` and `id-token: write`: its pushes and the pull request use the short-lived, repository-scoped token of the Claude GitHub App, so the draft triggers the usual CI (a pull request made with the workflow token would not). Claude's shell commands are limited to an allowlist (git on its own branch, `gh pr create --draft`, pip, uv, ruff, mypy, bandit, pytest and three repository checks), and web fetch and search are off, so text from the internet cannot steer it.
- **Data, not instructions.** The work list carries no long third-party text; the prompt tells Claude to treat it as data. It travels as a job output, not an artifact, because artifacts of a public repository can be downloaded by anyone.
- **No duplicate work.** Open Dependabot pull requests are passed to Claude so it does not redo them, and nothing runs while an earlier `security-patch/` pull request is open.
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

Set one of these, as a repository secret or variables:

- `CLAUDE_CODE_OAUTH_TOKEN`: from `claude setup-token` on a machine signed in to your Claude plan; runs count against that plan.
- `ANTHROPIC_API_KEY`: a Claude API key; runs are billed to the API account.
- Workload identity federation (no stored secret): variables `ANTHROPIC_FEDERATION_RULE_ID` and `ANTHROPIC_ORGANIZATION_ID`, with a federation rule for this repository's GitHub OIDC tokens in the Claude Console.

The Claude GitHub App must be installed on the repository (it already is, for `@claude`).
