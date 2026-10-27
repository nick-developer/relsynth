# ADR 0002: Open Policy Agent (Rego) for release policy

- Status: proposed (T08 design phase), 2026-10-27
- Deciders: maintainers; the user approves through the PR
- Context owner: T08 governance core. Affects T06 (privacy verdicts), T07 (DP budget), T09 (identity and access), T17 (evidence).

## Context

relsynth must refuse to publish a synthetic dataset that fails the organisation's rules, and say why in words a privacy officer can read. The rules read documents that relsynth already produces: the run manifest, the T06 privacy report (`relsynth.privacy-report/1`), the T05 quality report, the privacy budget ledger, and the organisation policy (`relsynth.org-policy/1`). Typical rules are "external releases need DP", "membership AUC upper bound below 0.60", "domain budget not exceeded", "two approvals from the DPO role for public releases", and "this target is allowed for this context".

The roadmap analysis says not to invent a policy language and names two candidates: Open Policy Agent with Rego, and Cedar.

## Options

| | OPA / Rego | Cedar |
|---|---|---|
| Model | General policy over any JSON input; rules produce any JSON (sets of reasons, objects) | Authorisation: principal, action, resource, context; result is allow or deny with the ids of the deciding policies |
| Numbers | 64-bit integers and floating point | `Long` integers and a `decimal` extension with at most 4 fractional digits ([Cedar data types](https://docs.cedarpolicy.com/policies/syntax-datatypes.html)) |
| Readable reasons | Rules return the reason strings directly (`deny contains msg`) | Only through policy ids and annotations |
| Maturity and governance | CNCF graduated project since 2021. In August 2025 Apple hired the Styra founders; OPA stays under CNCF governance ([Cloud Native Now, 2025](https://cloudnativenow.com/features/apple-buys-styra-brains-opa-remains-open/)) | Joined the CNCF sandbox in January 2026 ([InfoQ, 2026](https://infoq.com/news/2026/01/cedar-joins-cncf-sandbox/)) |
| Assurance | Large test suite, wide production use (Kubernetes admission, Envoy, Terraform) | Formally specified in Lean, with the Rust implementation tested against the spec by differential random testing ([Disselkoen et al., 2024, "How we built Cedar"](https://arxiv.org/abs/2407.01688)); policies can be analysed with an SMT solver |
| Running it from Python | A static `opa` binary (`opa eval`), an OPA server as a sidecar, or a policy compiled to WebAssembly. No maintained in-process Python binding (`regopy` exists but is small) | Rust core; community Python bindings (`cedarpy`) run in-process |
| Air-gapped on-premises | Single static binary, no network needed | Python wheel, no network needed |

## Decision

Use **OPA with Rego** for release policy.

1. The rules compare small real numbers. δ is typically 10⁻⁹ and limits like 0.05 come from confidence intervals. Cedar's decimal has four fractional digits, so every number would have to be rescaled into integers by hand. That is a source of silent errors in exactly the place where errors matter most.
2. The acceptance criterion asks for a readable reason. Rego returns the reason text as data. With Cedar we would have to rebuild it from policy annotations.
3. The inputs are documents (manifests and reports), not principal-action-resource triples. This is OPA's home ground.
4. OPA is the more mature project (CNCF graduated). Its maintainer change in 2025 did not change its governance.

relsynth runs OPA as a pinned static binary that it calls with `opa eval`, or as a sidecar in the T09 service. The binary's version goes into every manifest (`policy.engine`). The built-in rules ship as `relsynth/governance/policies/release.rego`, a draft of which is in `docs/governance/examples/policies/`. Organisations add their own modules in `package relsynth.org`. These modules can add deny reasons but cannot remove any.

## Defence in depth

The policy engine is not the only gate. Four invariants are also checked in Python, before and after the policy is evaluated:

1. An external context requires DP.
2. An external context requires an approval.
3. A `fail` or `blocked` verdict can never be released.
4. The domain's lifetime budget can never be exceeded.

If OPA is missing, the policy fails to load or evaluation errors, the release is refused. A broken or permissive policy can therefore only make relsynth stricter than these invariants, never looser.

## Consequences

- An extra binary (about 50 MB) in the on-premises install. The T01 supply-chain job pins it by digest.
- Policy unit tests run with `opa test` in CI (build phase).
- Cedar remains a good fit for *who may do what*, such as which users may approve or re-identify. T09 owns identity and access and may choose Cedar for that layer. The release rules here would not change.
- If a maintained in-process Rego engine for Python appears (for example Python bindings of Microsoft's `regorus`), it can replace the subprocess without changing any policy.

## Not decided here

Guessing: the subprocess call costs about 20–50 ms per decision. That is negligible next to a run, but the build phase should measure it.
