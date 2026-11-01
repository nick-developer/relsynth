# Agents: drafting and explaining, never deciding

relsynth's agents save people time on the reading and checking that comes before a decision. They answer the review questions of an introspected draft with evidence, explain reports in plain language, turn doctor findings into proposals, and draft data contracts and policies. People make every decision: which answers to accept, what to release, which epsilon to spend, which policy exception to grant.

This guide covers what each agent does, the safeguards, the command line, the service, and how accurate the agents are on the four real databases (roadmap T16; propositions 1600–1604 in [docs/proofs](proofs/)).

## Contents

- [What the agents do](#what-the-agents-do)
- [Backends: rules or a local model](#backends-rules-or-a-local-model)
- [What an agent may see](#what-an-agent-may-see)
- [Safeguards](#safeguards)
- [Command line](#command-line)
- [In the service](#in-the-service)
- [Records](#records)
- [Evaluation](#evaluation)
- [Limits](#limits)

## What the agents do

| Agent | Input | Output | Who decides |
|---|---|---|---|
| review | an introspected draft and its source | one answer per `review:` item: a verdict, the evidence measured on every row, an explanation, and the change it implies | the spec owner accepts or rejects each answer |
| explain | a quality, privacy or scorecard report | a plain-language summary | nobody: it is information |
| doctor | a diagnostics bundle | the doctor's fixes as proposals; fixes to privacy settings as advice to the privacy officer | the spec owner; the privacy officer for privacy settings |
| draft contract | a spec | an ODCS v3.2.0 contract with status `draft` | the data owner reviews it and sets it active |
| draft policy | a spec | an organisation policy with the privacy unit and source filled in and every budget empty | the privacy owner sets the budgets and signs it |

**Review answers.** `relsynth introspect` lists every inference it made from a sample under `review:` (keys, links, lookups, rules, column types) and asks `ASK` questions the data could not answer. The review agent checks each inference on every row of the table and answers:

| Verdict | Meaning |
|---|---|
| `confirm` | the inference holds on every row |
| `reject` | it does not; the answer proposes the change that removes it (a link or a rule) |
| `answer` | for an `ASK` question about a foreign key: the parent table, or none |
| `unsure` | the evidence does not decide; a person looks |
| `explain` | a privacy decision (personal data, copying a table as-is, leaving a table out) or a note: the agent explains it and does not answer it |

Answers that need a person's attention (every `reject`, `answer`, `unsure` and privacy item, and any answer below 0.8 confidence or with a flag) are marked.

## Backends: rules or a local model

| Backend | What it is | Data leaves the process |
|---|---|---|
| `rules` (default) | deterministic judgement from the evidence; reproducible | no |
| `openai` | any OpenAI-compatible chat server: Ollama, vLLM, llama.cpp or LM Studio on the same machine (default `http://localhost:11434/v1`), or an organisation's gateway | to that server, aggregates at most |

The model is asked only where the rules are unsure (mostly `ASK` questions that need the meaning of a name, such as `SupportRepId` referring to employees) and to word explanations. A server that is not on the local machine is refused unless `allow_remote` is set, must use https, and then gets the `aggregates` scope at most.

```bash
ollama pull qwen2.5:7b-instruct
relsynth agent review draft.yaml --backend openai --model qwen2.5:7b-instruct --md answers.md
```

GPU use for the model is the model server's concern; relsynth sends only small JSON requests.

## What an agent may see

An agent never sees source rows. The data scope of a task decides how much of the aggregates it gets:

| Scope | The agent gets |
|---|---|
| `schema` | names, types and the spec; no data is read |
| `aggregates` | counts, shares and distinct counts of whole columns |
| `values` | as `aggregates`, plus single values that occur at least `k` times in their column (default 10) |

The scope is the lowest of the scope of the person who asked (by role) and the ceiling set for the agent:

| Role | Scope |
|---|---|
| consumer, tenant admin | schema |
| engineer, agent account | aggregates |
| data owner, privacy officer | values |

The service's default ceiling is `aggregates` (`RELSYNTH_SVC_AGENT_SCOPE_CEILING`).

## Safeguards

The safeguards assume the worst case: a model that follows every instruction an attacker planted in a column name, a data value, a spec description or a report field (OWASP Top 10 for LLM Applications 2025: LLM01 prompt injection, LLM02 sensitive information disclosure, LLM05 improper output handling, LLM06 excessive agency; NIST AI 600-1, information security and information integrity risks).

1. **Untrusted text is marked and checked.** Names, values and review text reach a model only inside `<data>` elements, JSON-escaped, after removing control, zero-width and bidirectional characters (spotlighting by datamarking, Hines et al. 2024). Text that reads like an instruction is flagged for the reviewer and recorded.
2. **Closed answers.** A verdict must be one of the verdicts allowed for its item; a parent must be one of the tables the question offered; cited evidence must exist. Privacy items get an explanation only.
3. **Typed changes.** An answer carries only the change its verdict implies: add or remove one relationship, or remove one rule. The guard builds the change from the item, never from model text, and rebuilds it again when a person accepts it. No change can touch privacy settings, personal data declarations, copy modes, DP, epsilon, policies, approvals or releases.
4. **Evidence first.** A confirmation that the evidence on every row contradicts (duplicates in a key, values without a parent, rule violations) is turned into a rejection.
5. **Clean, grounded text.** Explanations are plain text without links, images, HTML or addresses, and every number in a model's text must occur in the evidence; otherwise the guard's own wording is used. A report explanation that states a release decision is replaced.
6. **No authority.** Agents return proposals. Only a person's decision applies them. In the service an agent account can read and propose and nothing else; in the governance registry an agent key cannot sign an approval.

The prompt-injection suite (`tests/unit/agents/test_injection.py`) places 20 payloads in names, values, review lines, descriptions and report fields and runs them against a backend that obeys every instruction; the checks C1600–C1604 try random model outputs and random databases.

## Command line

```bash
# answer the review items (nothing is changed)
relsynth agent review draft.yaml --out answers.json --md answers.md

# a person's decisions: accept every confirmation, one answer, reject another
relsynth agent apply draft.yaml answers.json --accept all-confirmed,3f2a9c0e1b7d --reject 9e1c2b3a4d5f \
    --by alice --out spec.yaml --log decisions.json

relsynth agent explain quality.json
relsynth agent doctor bundle.json --out proposals.json
relsynth agent draft contract spec.yaml --owner alice --out contract.yaml
relsynth agent draft policy spec.yaml --organisation "Acme" --out policy.draft.yaml

# accuracy against ground truth: a draft made with --ignore-declared, and the catalog draft
relsynth introspect sqlite:shop.db --ignore-declared --out draft.yaml
relsynth introspect sqlite:shop.db --out truth.yaml
relsynth agent eval draft.yaml truth.yaml --min-accuracy 0.95
```

In Python:

```python
from relsynth import agents
res = agents.review("draft.yaml")                                  # rules backend, aggregates scope
spec, log = agents.apply_decisions(res.spec, res.answers, {a["item"]: "accept" for a in res.answers
                                                           if a["verdict"] == "confirm"}, decided_by="user:alice")
print(agents.explain_report(quality_report_dict)["text"])
```

## In the service

| Endpoint | Who | What |
|---|---|---|
| `POST /api/v1/t/{tenant}/specs/{id}/agent/review` | engineer, data owner, privacy officer, agent account | queue the review agent on a spec; its answers become proposals |
| `GET /api/v1/t/{tenant}/specs/{id}/proposals` | engineer, data owner, privacy officer, agent account | the proposals and their state |
| `POST /api/v1/t/{tenant}/specs/{id}/proposals/decisions` | engineer, data owner (people only) | accept or reject proposals; accepted changes make a new spec version written by that person |
| `POST /api/v1/t/{tenant}/runs/{id}/agent/explain` | engineer, data owner, privacy officer, agent account | a plain-language explanation of the report the caller may read |

An **agent account** is a service account created with the role `agent` (`POST /service-accounts` with `"roles": ["agent"]`). It holds that role only, whatever else it is given, and may read projects, specs, runs, reports and proposals and ask for reviews and explanations. Approvals, runs, specs, connections, service accounts and the audit log are refused to it, and each refusal is audited. OIDC tokens cannot claim the agent role.

Settings: `RELSYNTH_SVC_AGENT_BACKEND` (`rules` or `openai`), `AGENT_BASE_URL`, `AGENT_MODEL`, `AGENT_API_KEY_ENV`, `AGENT_ALLOW_REMOTE`, `AGENT_SCOPE_CEILING`, `AGENT_NAME`.

## Records

Every agent action is recorded with the agent's own identity, the person it acted for, the backend and model, the scope, and digests of its inputs and output:

- in the service's hash-chained audit log: `agent.review.queued`, `agent.review.proposed`, `agent.proposal.accepted`, `agent.proposal.rejected`, `agent.explain`, and refusals;
- in a governance registry (`--registry --key` on the command line, with a key trusted with the role `agent`): a signed in-toto statement of type `agent-action/1`, covered by the registry's hash chain;
- in a run manifest, under `agents`: the action's digest and who accepted its proposals (`relsynth.agents.record.manifest_entry`, passed to `governance.record_run(agents=...)`).

## Evaluation

`python -m bench.agents` drafts each database with its catalog hidden (benchmark mode), lets the review agent answer, and compares every answer about a key or a link with ground truth the agent does not see: the declared catalog of Chinook, AdventureWorks DW and Contoso, and the hand-reviewed spec of bi/COREP. The CI job `agents` runs it with the rules backend as a gate (accuracy at least 0.95) and with a small local model under Ollama as a measurement.

Rules backend, aggregates scope (2026-11-01):

| Database | Items with ground truth | Correct | Wrong | Left to people | Agent time |
|---|---:|---:|---:|---:|---:|
| Chinook | 22 | 21 | 0 | 1 | 0.4 s |
| AdventureWorks DW | 73 | 71 | 1 | 1 | 40 s |
| Contoso (SQLite) | 15 | 15 | 0 | 0 | 39 s |
| bi/COREP | 16 | 15 | 1 | 0 | 0.8 s |
| **All** | **126** | **122** | **2** | **2** | |

Accuracy on answered items is 122/124 = 0.984 (Wilson 95% interval 0.943–0.996), with 98.4% of items answered. Both wrong answers are links that hold on every row but that the ground truth leaves out: AdventureWorks does not declare `FactAdditionalInternationalProductDescription.ProductKey → DimProduct`, and the bi/COREP reviewer left `DIM_Module.ApplicableFromDateKey → DIM_Date` out of the model. The first was flagged for attention, the second was not. In the simulated approval step a reviewer who knows the truth applies no wrong change. A reviewer who accepts only the answers not flagged for attention would accept the second one, which adds no change to the spec because the draft already holds the link.

Rule items have no independent ground truth, because the agent measures them on every row. Of 92 lookups, derive, check and aggregate rules, constants, dates and key orders across the four databases, 90 held on every row. Two check rules found on a 20,000-row sample failed on a few rows of the whole table and were proposed for removal: `FactResellerSales: OrderDateKey.DayNumberOfWeek == ShipDateKey.DayNumberOfWeek` and `Customer: days(StartDT) < days(EndDT)` in Contoso.

Time: with the stated assumption of 2 minutes for a person to check one item by hand and 30 seconds to read an answer with its evidence, the review of the 4 databases drops from about 7.3 hours (219 items) to about 2.1 hours. This is an estimate from those assumed times, not a measurement of people.

## Limits

- The rules backend answers an `ASK` question only when the evidence and the names decide it: it found the self-reference `Employee.ReportsTo` but left `Customer.SupportRepId` (which data fits four tables equally) to people. A model is needed for meanings, and its accuracy is measured in CI, not promised here.
- Links that introspection did not ask about cannot be answered: on the toy test database `customer.support_rep` is missing from the draft, and the evaluation reports such links separately.
- The evidence tools read whole tables through the source reader. On very large tables this is slow; a pushdown version is future work.
- The agents do not judge whether a column holds personal data; that stays a privacy decision.
