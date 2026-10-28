# Release checklist

relsynth follows [Semantic Versioning](https://semver.org/). A release is cut from `main` by a maintainer; every step below is in a pull request or in the GitHub release page, never a push to `main`.

## 1. Decide the version

- **Major**: a breaking change to the CLI (commands, options, JSON fields, exit codes), the public Python API or the spec format, or a changed default that alters privacy behaviour.
- **Minor**: new features, new connectors, new spec options with backward-compatible defaults.
- **Patch**: fixes only.
- Draws for the same seed may change in any release; the notes say so when they do.

## 2. Prepare (pull request `release/vX.Y.Z`)

- [ ] `main` is green on every required check, including `science (1/4)`…`(4/4)` and `bench`
- [ ] Benchmark history shows no unexplained regression since the previous release (`python -m bench compare`, from the benchmark harness)
- [ ] The weekly mutation score has not dropped below its baseline without a reason
- [ ] `src/relsynth/_version.py` holds the new version
- [ ] CHANGELOG.md: rename `## Unreleased` to `## X.Y.Z (YYYY-MM-DD)`, add a fresh empty `## Unreleased`, and check every user-visible change is listed; say whether draws change for the same seed
- [ ] Generated references are current: `python docs/make_cli_reference.py`, `python docs/make_api_reference.py`, `python docs/make_spec_reference.py`
- [ ] Proofs: every new proposition is in `docs/proofs/` and the PDF is rebuilt
- [ ] `docs/datasets.md` rows used by this release are re-checked
- [ ] NOTICE lists any new third-party material
- [ ] Merge the pull request (rebase)

## 3. Release

- [ ] Draft a GitHub release with tag `vX.Y.Z` on the merge commit; paste the changelog section as notes
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
