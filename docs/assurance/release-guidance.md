# Release guidance: internal, partner and public

Who a synthetic dataset goes to decides how it must be made. The CJEU (*EDPS v SRB*, C-413/23 P, paras. 75–77 and 86) and the EDPB (Guidelines 02/2026, paras. 11–15 and 29) assess identifiability from the perspective of those who will have access, so relsynth judges every release for a named **context**. This page says, for each of the three built-in contexts, what the organisation's policy enforces, what to configure and what to check before approving. It is guidance for the people who approve releases, not legal advice.

## 1. The three contexts at a glance

| | internal | partner | public |
|---|---|---|---|
| Recipient | Staff of the organisation, for development, testing or analysis | A named organisation under contract | Anyone (open data, publication, a model or dataset shared openly) |
| NIST SP 800-188 sec. 3.4 sharing model | Restricted access inside the organisation | Restricted access under a data use agreement | Publishing synthetic data |
| Attacker assumed to know | The quasi-identifiers | Every attribute except the one guessed, plus population data | Same as partner |
| Differential privacy | Optional | **Required** (enforced before any data is read, DP-02) | **Required** |
| Default limits (`risk`, `mia_auc`, `record_auc`) | 0.10, 0.65, 0.85 | 0.05, 0.60, 0.75 | 0.02, 0.55, 0.65 |
| The most exposed person | Shown; the pooled top-10 AUC gates | Gates the verdict, with 128 test shadows | Gates the verdict, with 128 test shadows |
| Fewer than 1,000 people in the unit | Warning | Fails without DP | Fails without DP |
| Approval in the example policy | Only for a *review*, one DPO or privacy officer | Every release, one DPO | Every release, two DPOs |
| Spec | May be drafted by `introspect` | Must be written from public facts (`spec_is_public: true`) | Same as partner |

The limits are relsynth's suggestions; the organisation sets its own in the signed policy (docs/governance/examples/org-policy.yaml). An organisation can define further contexts, for example a supervisory authority; an external context always requires DP (GOV-04).

## 2. The Five Safes for each context

NIST SP 800-188 sec. 3.5 recommends the Five Safes as a frame for a release decision.

| Safe | internal | partner | public |
|---|---|---|---|
| Safe projects | The purpose is recorded in the DPIA | Purpose in the DPIA and the contract | Purpose in the DPIA; publication is irreversible (SP 800-188 sec. 1: a published release "will remain in the public domain indefinitely") |
| Safe people | Staff with a need; access to the target controlled | Named recipients; contract forbids re-identification and onward transfer (a contract is not a legal prohibition, EDPB para. 34, so it does not lower the attacker model) | No control: assume anyone, including the relevant entities of EDPB para. 30 |
| Safe settings | `allowed_targets` in the policy (for example a dev sandbox) | `allowed_targets`, for example a partner SFTP | An open data portal |
| Safe data | Measured risk below the internal limits | DP at the organisation's ε, and measured risk below the partner limits | DP at a smaller ε than partner, and measured risk below the public limits |
| Safe outputs | The privacy report and model stay inside | Package holds only the synthetic tables, manifest, approvals, token and DP statement (GOV-07) | Same as partner |

## 3. Before approving: a checklist

For every context:

1. `relsynth gov verify registry/` passes.
2. The run was assessed (`--assess` or `--privacy-report`) against this context, and the verdict is *pass*, or *review* with a named sign-off.
3. The privacy unit is the right table, and every table that holds personal data is owned by it or listed as public with a reason (`relsynth privacy` lists the owned tables; DP mode refuses to guess).
4. Tables with `mode: copy` hold no personal data.
5. The quality report is fit for the purpose. A dataset that is wrong in a way that matters to decisions about people is a risk in itself.
6. The DPIA is filled in or updated ([template](dpia-template.md)).

For partner and public, also:

7. The run is a DP run, and the DP statement's unit, ε and δ are what the DPIA states.
8. The spec was written from public facts (`spec_is_public: true`), and the facts declared under `public` really are public.
9. `relsynth gov budget` shows the domain still below its lifetime cap after this release, and the number of earlier non-DP releases of the domain (they are not covered by the DP accounting).
10. Free text, binary and `pii` columns are placeholders in DP mode; check the recipient does not need them.
11. For public: no smaller audience would do. Publication cannot be recalled.

## 4. When the answer is no

| Situation | What to do |
|---|---|
| Verdict *fail* in internal | Drop or generalise the attributes the report names; raise `min_token_rows` for text; or use DP for the internal release too |
| *review* that nobody can sign | Run more shadows (`n_test_gated`) to narrow the intervals, or treat it as a fail |
| Small population (under 1,000 people) | Internal only, or DP with a small ε |
| The budget is exhausted | No further DP release of that domain; the lifetime cap never resets (GOV-10) |
| A released dataset turns out to be risky | `relsynth gov revoke`; for partner, ask the recipient to delete under the contract; for public, revocation cannot recall copies |

## 5. What relsynth cannot decide

- Whether a given ε is acceptable for this recipient and these people. NIST SP 800-226 sec. 2.2 gives no rule; the organisation decides (A4 of the [assurance case](assurance-case.md)).
- Whether the result meets the legal test of anonymity in a given jurisdiction (EDPB para. 22: "insignificant in reality"). **Lawyer.**
- Whether onward transfers by the recipient are lawful. **Lawyer.**
- Whether a HIPAA expert determination can rely on the report; that is the expert's call (45 CFR 164.514(b)(1)).
