# Release checklist

relsynth follows [Semantic Versioning](https://semver.org/). **Every release is published to PyPI and updates README.md and the relevant documentation**; `tools/release.py check` fails CI when any file still names an older version. A release is cut from `main` by a maintainer; every step below is in a pull request or in the GitHub release page, never a push to `main`.

## 1. Decide the version

- **Major**: a breaking change to the CLI (commands, options, JSON fields, exit codes), the public Python API or the spec format, or a changed default that alters privacy behaviour.
- **Minor**: new features, new connectors, new spec options with backward-compatible defaults.
- **Patch**: fixes only.
- Draws for the same seed may change in any release; the notes say so when they do.

## 2. Prepare (pull request `release/vX.Y.Z`)

- [ ] `main` is green on every required check, including `science (1/4)`…`(4/4)` and `bench`
- [ ] Benchmark history shows no unexplained regression since the previous release (`python -m bench compare`, from the benchmark harness)
- [ ] The weekly mutation score has not dropped below its baseline without a reason
- [ ] `python tools/release.py bump X.Y.Z` writes the version into `src/relsynth/_version.py`, the README, `docs/cicd.md`, the example pipelines, both Dockerfiles, the Helm chart and the OpenAPI document, and renames `## Unreleased` in CHANGELOG.md to `## X.Y.Z (date)` under a fresh `## Unreleased`; `python tools/release.py check --release` must then pass (CI and the release workflow run it)
- [ ] CHANGELOG.md: check every user-visible change is listed; say whether draws change for the same seed
- [ ] README.md describes every user-visible feature of the release (what it does, install extras, commands, validation numbers, limitations, documentation map), and the guides in `docs/` are updated for it
- [ ] Generated references are current: `python docs/make_cli_reference.py`, `python docs/make_api_reference.py`, `python docs/make_spec_reference.py`
- [ ] Proofs: every new proposition is in `docs/proofs/`, `python tools/check_proofs.py` passes, and `docs/proofs/relsynth-proofs.pdf` is replaced by the `proofs-pdf` artifact of the CI job `proofs` (or a local `latexmk -pdf relsynth-proofs.tex`)
- [ ] `docs/datasets.md` rows used by this release are re-checked
- [ ] NOTICE lists any new third-party material
- [ ] Merge the pull request (rebase)

## 3. Release

- [ ] Draft a GitHub release with tag `vX.Y.Z` on the merge commit; paste the changelog section as notes (`python tools/release.py notes X.Y.Z`)
- [ ] Publish it. `pypi_upload.yml` runs the full gate, builds, writes the SBOM, signs, attests, attaches the files and uploads to PyPI after the `pypi` environment approval
- [ ] Approve the `pypi` deployment once the gate is green

## 4. Verify

- [ ] The release page has the wheel, sdist, `relsynth.cdx.json`, `*.sigstore.json` and `SHA256SUMS`
- [ ] Verify one artefact as in [supply-chain.md](supply-chain.md#verifying-a-release) (Sigstore and `gh attestation verify`)
- [ ] `pip install relsynth==X.Y.Z` in a clean environment; `relsynth version --json` shows the version
- [ ] PyPI shows the attestations of the files

## 5. Afterwards

- [ ] Announce (release notes link) and close the milestone
- [ ] Security fixes: publish the GitHub security advisory and request the CVE
- [ ] The minor release before the previous one leaves security support 6 months later (SECURITY.md)
