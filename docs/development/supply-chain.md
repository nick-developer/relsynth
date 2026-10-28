# Supply chain: SBOM, signatures and provenance

Every release of relsynth ships, next to the wheel and sdist on the GitHub release page:

| File | What it is | Standard |
|---|---|---|
| `relsynth.cdx.json` | Software bill of materials of the wheel installed with every extra (all Python dependencies, versions, licences, package URLs) | CycloneDX 1.6 |
| `*.sigstore.json` | Sigstore bundle of each wheel, sdist and the SBOM: keyless signature by the release workflow's GitHub identity, with its Rekor transparency-log entry | Sigstore |
| `SHA256SUMS` | Checksums of the wheel and sdist | |
| GitHub attestations | SLSA v1 build provenance of the wheel and sdist, and the SBOM bound to the wheel | SLSA v1.0, in-toto |
| PyPI attestations | PEP 740 attestations created by trusted publishing | PEP 740 |

The release workflow is [`pypi_upload.yml`](../../.github/workflows/pypi_upload.yml). It runs the complete CI gate first, then builds once, generates the SBOM from a clean install, signs, attests, attaches the files to the release and uploads to PyPI with trusted publishing (no long-lived token). The same build, SBOM and signature steps run as a dry run in CI on every change (`build` and `sign` jobs), so a broken supply-chain step is found before a release.

## Verifying a release

```bash
pip download relsynth==2.0.0 --no-deps -d dl && cd dl
# 1. the Sigstore signature, made by this repository's release workflow
python -m pip install sigstore
python -m sigstore verify identity relsynth-2.0.0-py3-none-any.whl \
  --bundle relsynth-2.0.0-py3-none-any.whl.sigstore.json \
  --cert-identity "https://github.com/nick-developer/relsynth/.github/workflows/pypi_upload.yml@refs/tags/v2.0.0" \
  --cert-oidc-issuer https://token.actions.githubusercontent.com
# 2. the SLSA build provenance and the SBOM attestation
gh attestation verify relsynth-2.0.0-py3-none-any.whl --repo nick-developer/relsynth
gh attestation verify relsynth-2.0.0-py3-none-any.whl --repo nick-developer/relsynth \
  --predicate-type https://cyclonedx.org/bom
```

## SLSA level

- **Met: SLSA v1.0 Build Level 2.** The build runs on GitHub-hosted runners from a versioned workflow, and the provenance is generated and signed by the build platform (GitHub artifact attestations, Sigstore-backed), so it cannot be forged by the package's own build steps.
- **Target: Build Level 3** for the 2.0 release. Level 3 needs the provenance signing to be isolated from the build steps that a contributor controls. GitHub reaches it when the build and attestation run in a reusable workflow that the release workflow calls, and the verifier pins that workflow (`gh attestation verify --signer-workflow`). The step left is to move the `build` job of `pypi_upload.yml` into such a reusable workflow; the steps themselves do not change.
- Source track: protected `main` with required review and status checks ([repository settings](repository-settings.md)) and signed-off commits.

References: [SLSA v1.0 levels](https://slsa.dev/spec/v1.0/levels), [CycloneDX](https://cyclonedx.org/specification/overview/), [Sigstore](https://docs.sigstore.dev/), [GitHub artifact attestations](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations).
