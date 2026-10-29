# Repository settings (for a repository admin)

These settings cannot be stored in the repository; an admin applies them once under **Settings**. They make the rules of [CONTRIBUTING.md](../../CONTRIBUTING.md) enforceable.

## General

- **Pull requests:** allow **rebase merging** and **squash merging**; untick **Allow merge commits** (linear history). Tick **Always suggest updating pull request branches** and **Automatically delete head branches**.
- **Features:** Issues on; Discussions optional.

## Branch ruleset for `main`

**Settings → Rules → Rulesets → New branch ruleset**

| Field | Value |
|---|---|
| Name | `main` |
| Enforcement status | Active |
| Target branches | Include default branch |
| Bypass list | *Repository admin*, mode **For pull requests only** (see the note below) |
| Restrict deletions | on |
| Require linear history | on |
| Require a pull request before merging | on: required approvals **1**, dismiss stale approvals on new commits, require review from Code Owners, require conversation resolution; allowed merge methods **Rebase, Squash** |
| Require status checks to pass | on, "Require branches to be up to date before merging"; checks listed below |
| Block force pushes | on |
| Require signed commits | off (the DCO sign-off is checked instead; signing can be added later) |

**Required status checks** (the job names in `ci.yml`, source GitHub Actions):

```
lint
types
dco
unit (ubuntu-latest, 3.10)
unit (ubuntu-latest, 3.11)
unit (ubuntu-latest, 3.12)
unit (ubuntu-latest, 3.13)
unit (windows-latest, 3.12)
unit (macos-latest, 3.12)
e2e
integration
science (1/4)
science (2/4)
science (3/4)
science (4/4)
build
sign
bench
```

`bench` exists once the benchmark-harness pull request is merged; add it then.

**Why admins may bypass the review rule.** GitHub never lets the author of a pull request approve it. While the project has one maintainer, every pull request (including those opened by Claude through the maintainer's account) is authored by that maintainer, so a strict "one approval" rule would block every merge. With the bypass limited to pull requests, an admin can still merge only through a pull request whose checks pass, never by pushing to `main`. Remove the bypass when a second maintainer joins.

## Tag ruleset for releases

**New tag ruleset**: target `v*`; restrict creations, updates and deletions to *Repository admin*. Release tags then cannot be moved or deleted after a release.

## Code security

**Settings → Code security**

- **Private vulnerability reporting:** enable (SECURITY.md points reporters to it).
- **Dependabot alerts** and **Dependabot security updates:** enable. Version updates come from `.github/dependabot.yml`.
- **Secret scanning** and **Push protection:** enable.
- **Code scanning:** CodeQL default setup (already active).

## Environments

**Settings → Environments → `pypi`**: required reviewer = the maintainer, deployment branches and tags = `v*` tags only. On pypi.org the trusted publisher stays as it is: repository `nick-developer/relsynth`, workflow `pypi_upload.yml`, environment `pypi`.

## Actions

**Settings → Actions → General**: allow GitHub Actions and actions by verified creators plus `pypa/*` and `sigstore/*`; workflow permissions **Read repository contents** (each workflow asks for more where it needs it); do not allow Actions to create or approve pull requests.
