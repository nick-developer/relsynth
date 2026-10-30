# Governance: a user guide

relsynth can record every run in a **registry**: a folder with signed run manifests, the organisation's privacy budget, approvals and releases, kept in a hash chain so that any later change is detected. The design and its reasons are in [design.md](design.md); this page shows how to use it. Install the extra first:

```
pip install 'relsynth[gov]'
```

## 1. Keys and the registry

Every person or service that signs gets an Ed25519 key. The privacy owner (usually the DPO) creates the registry; their key is its root of trust.

```
relsynth gov keys new dpo.key --name "Data Protection Officer"
relsynth gov keys new ana.key --name "Ana Analyst"
relsynth gov init registry/ --owner-key dpo.key --policy-out policy.yaml
relsynth gov keys trust ana.pub.json --roles operator --owner-key dpo.key --registry registry/
```

Private keys are written with mode 0600. Set `RELSYNTH_KEY_PASSPHRASE` to encrypt them. Give the root key id printed by `init` to anyone who will verify your release packages.

## 2. The organisation policy

`init` writes a policy skeleton. It has **no** privacy budget: relsynth ships no ε, and the privacy owner must choose one (see [choosing ε](#choosing-epsilon)). Until then, `gov policy check` refuses the policy with a sentence that says what is missing. A filled example is [examples/org-policy.yaml](examples/org-policy.yaml).

```
relsynth gov policy check policy.yaml      # caps in epsilon and rho, contexts, digest
relsynth gov policy sign policy.yaml --key dpo.key --registry registry/
```

Only a trusted key with the privacy owner's role can sign it. Every manifest records the digest of the policy it was judged under.

The policy's `engine` chooses who evaluates the release rules: `builtin` (default, a Python port of [release.rego](../../src/relsynth/governance/rego/release.rego) that needs no extra binary) or `opa` (Open Policy Agent; needed for the organisation's own Rego modules under `rego:`). Four invariants hold whatever the policy says: an external release needs DP, an external release needs an approval by someone other than the operator, a `fail` or `blocked` privacy verdict is never released, and no domain may exceed its lifetime budget.

## 3. A governed run

```
relsynth run spec.yaml --registry registry/ --context internal --release-target dev-sandbox \
    --source-id dwh-prod --key ana.key --assess
```

- `--source-id` selects the privacy domain of the source in the policy.
- `--context` and `--release-target` are checked against the policy before any data is read. A context outside the organisation is refused unless the run is differentially private (`--epsilon`, `--dp` or a `dp` section; see [dp.md](../dp.md)).
- `--assess` runs the privacy attacks of `relsynth privacy` against the policy's contexts; `--privacy-report FILE` uses a report you already have. Without either, the verdict is missing and the run cannot be released.

The run writes its synthetic data as usual and then records a signed manifest: hashes of the outputs and the spec, versions, seeds, the verdicts, the quality score and what the policy still needs before a release. The manifest holds no data, no row counts and no credentials. A run without DP on a domain is noted in the domain's ledger.

## 4. Approve, release, revoke

```
relsynth gov ls --registry registry/
relsynth gov show RUN --registry registry/
relsynth gov approve RUN --context internal --key dpo.key --registry registry/          # or --reject --reason "..."
relsynth gov release RUN --context internal --target dev-sandbox --key ana.key --out package/ --registry registry/
relsynth gov revoke RUN --reason "wrong source" --key dpo.key --registry registry/
```

`release` evaluates the policy. On deny it exits 1 and prints one sentence per reason, for example:

- `context "partner" is outside the organisation: differential privacy is required, and this run has none`
- `domain "customers" would reach rho 0.47, above its lifetime cap 0.435346 (epsilon 6 at delta 1e-09)`
- `context "public" needs 2 approval(s) from ["dpo"]; 1 valid`

On allow it signs a release token and writes the package: the synthetic tables byte for byte as signed, the manifest, the approvals, the token, a checkpoint, and the DP statement for DP runs. The privacy report never goes into a package: it was computed on real data.

Revoking never refunds privacy budget, because the data has already left.

## 5. Verify

```
relsynth gov verify registry/      # chain, checkpoints, signatures, objects, ledger protocol
relsynth gov verify package/       # what a recipient runs
```

Both exit 1 and list the problems, the first being where the registry or package was changed. A registry check catches edited, deleted, inserted, reordered and truncated log entries and any changed report or manifest (Proposition 803, check C803). A recipient should first check that the root key id in `registry.json` is the one you gave them.

## 6. The budget

```
relsynth gov budget --registry registry/
```

shows, per domain, the ρ and ε spent at the domain's δ, the lifetime cap, the period throttle if any, each DP run's state (reserved, touched, settled, refunded) and the number of releases made without DP. Budget is reserved before a DP run reads data and refunded only if it never read any (Proposition 802).

## Choosing epsilon

NIST SP 800-226 (sec. 2.2) gives no rule for ε. Values below 1 give strong protection; deployments use 1 < ε ≤ 20, and the risk above 1 should be measured, for example with `relsynth privacy`. δ should be at most 1/n² for n people and never above 10⁻⁵ (the policy schema refuses more). The lifetime cap never resets; a period cap only spreads it over time (Proposition 801).

## API

```python
import relsynth.governance as gov
owner = gov.Signer.load("dpo.key")
reg = gov.open_registry("registry/")
run = gov.record_run(reg, spec, tables, context="internal", target="dev-sandbox", operator=analyst,
                     source_id="dwh-prod", privacy_report=report)
gov.approve(reg, gov.current_policy(reg), run.run_id, "internal", owner)
gov.release(reg, run.run_id, "internal", "dev-sandbox", analyst, out="package/")
assert gov.verify_registry(reg) == []
```

## Differential privacy runs

`relsynth run spec.yaml --epsilon 1 ...` with the same governance options reserves the budget in the source's domain before reading, settles it afterwards, and records the charges and the DP statement in the manifest. [dp.md](../dp.md#4-governed-runs-and-the-budget) has the details.

## Not yet built

OpenLineage events and catalog pushes (design sec. 12), Sigstore keyless signing, and a database registry backend (T09).
