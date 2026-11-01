# ADR 0016: Agent-assisted operations

- Status: proposed (T16), 2026-11-01
- Numbered after the roadmap thread to avoid clashes with ADRs of parallel threads.

## Context

T16 adds agents that save people time on reading and checking (review questions of an introspected draft, report explanations, doctor fixes, contract and policy drafts) without making privacy or release decisions. relsynth runs mostly on premises, often air-gapped, on confidential sources. A language model must therefore be optional, local by default, and treated as untrusted: anything it reads may carry instructions planted in a column name, a value or a report (OWASP LLM01).

## Decisions

1. **Rules first, model optional.** The default backend is deterministic and judges from evidence the agent measures itself on every row. A model is consulted only where the rules are unsure (mostly `ASK` questions that need the meaning of a name) and to word explanations. The release gate in CI uses the rules backend; the model is measured, not gated.
2. **One protocol: OpenAI-compatible chat completions with a JSON schema.** Ollama, vLLM, llama.cpp, LM Studio and most gateways speak it, so no vendor SDK is a dependency. Remote endpoints are refused unless allowed explicitly, must use https, and get the `aggregates` scope at most.
3. **The model is an adversary.** Every model output passes a guard that projects it onto closed sets (verdicts per item kind, candidate parents, cited evidence ids) and builds the change from the item, never from model text. Numbers in model text must occur in the evidence. Privacy items get explanations only. These are proved for every output (Props. 1600–1604), not tested on a sample of prompts only.
4. **Least data.** Agents never see rows. Data scope is the minimum of the caller's role scope and a configured ceiling (default `aggregates`); single values only in the `values` scope and only when they occur at least k = 10 times.
5. **Proposals, never actions.** Agents return proposals. Applying needs a person's identity, in the library (`apply_decisions --by`) and in the service (a decisions endpoint people only may call). In the service the agent is its own identity (`agent` role on a service account) with read and propose permissions only; in the governance registry an agent key cannot approve.
6. **Evaluation against ground truth the agent does not see**: the declared catalogs of three golden databases with the draft made in benchmark mode, and the curated bi/COREP spec, with Wilson intervals; time saved is reported as an estimate from stated minutes per item, not a measurement of people.

## Consequences

Answers that need meaning (which of several tables a foreign key refers to) stay with people unless a model is configured; the rules backend left 2 of 126 items to people on the four databases. The evidence tools read whole tables through the source reader, which is slow on very large tables; a pushdown version is future work. The service and governance changes are proposed to their owners as RFC-T16-01 (service) and RFC-T16-02 (governance).
