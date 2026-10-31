# Compliance evidence and assurance

This folder is for the privacy, legal and audit teams who decide whether a synthetic dataset made with relsynth may be released. It does not decide for them and is not legal advice; it gives them the argument, the evidence and the templates.

| Document | Use it to |
|---|---|
| [assurance-case.md](assurance-case.md) | Follow the argument that a release is fit for its recipient: claims, the evidence for each, assumptions and known defeaters |
| [regulation-mapping.md](regulation-mapping.md) | See what relsynth contributes to each provision of the GDPR (EDPB Guidelines 02/2026, *EDPS v SRB*), the AI Act (Art. 10, Art. 4a), HIPAA expert determination, NIST SP 800-226 and SP 800-188, ISO/IEC 27559 and 20889, and what stays with the organisation |
| [dpia-template.md](dpia-template.md) | Document one release: a DPIA following GDPR Art. 35(7), pre-filled with what relsynth does |
| [release-guidance.md](release-guidance.md) | Decide how to make and approve a release for an internal, partner or public recipient |
| [claims.yaml](claims.yaml) | Look up the evidence behind a sentence of the documentation |
| [evidence/](evidence/README.md) | The aggregate measurement files that claims cite |

## The claim register

Every sentence of the documents listed under `scope` in [claims.yaml](claims.yaml) that reads as a claim (it says *never*, *every*, *exact*, *guarantee*, *refused* and similar) has an entry. Each entry quotes the sentence and lists its evidence: a proposition of the [proofs](../proofs/relsynth-proofs.tex) by label, a numbered check (C902), a pytest test, a measured report in this repository, or the code that enforces it. A `status` says whether the evidence supports the claim fully, partly (with a note), or shows a limitation.

`python tools/claims_check.py` checks the register, and CI runs it on every pull request. It fails when:

- a quoted sentence is no longer in its document;
- a cited proposition, check, test or file does not exist;
- a claim about relsynth rests only on an external source or on code, without a proposition, check, test or report;
- a claim-like sentence of a document in scope has no entry, and is not listed under `not_claims` with a reason;
- a `Prop. N` in any Markdown document names a number the proofs do not define;
- these documents cite a claim id, a proposition label or a check that does not exist.

When your change makes it fail:

| Message | Fix |
|---|---|
| `quote not found` | You changed a registered sentence. Update `quote`, and check that the evidence still supports the new wording |
| `unregistered claim` | You wrote a new claim. Add an entry with its evidence (`python tools/claims_check.py --suggest docs/x.md` prints stubs), or, if it is not a claim about relsynth, add it to `not_claims` with a reason |
| `no test ...`, `no proposition ...` | You renamed or removed evidence. Point the entry at the new name |
| `needs a prop, check, test or report` | Write the test that shows the claim, or weaken the claim |

`python tools/claims_check.py --json claims.json` writes the register with every proposition label resolved to its printed number, for an auditor's tooling.

Documents in scope today: docs/privacy.md, docs/dp.md, docs/security.md and docs/governance/guide.md. The README and the remaining guides are added one by one; until then their `Prop. N` references are still checked.
