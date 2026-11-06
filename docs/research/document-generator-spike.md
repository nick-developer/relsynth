# Spike: a direct document generator (T13)

- Date: 2026-11-01, time-boxed to one day
- Decision: **no-go** for a direct document generator in 2.x; **go** for a targeted follow-up on dependence across
  levels inside the shredding baseline (below)

## Question

Should relsynth draw documents directly as trees (a model over the inferred schema, top-down) instead of shredding
them to tables, synthesising the tables with the relational engine and nesting them again (ADR 0010)?

## Candidates

1. **A probabilistic tree grammar over the inferred schema.** Each node of the schema becomes a production: for an
   object, which optional fields appear (given the discriminator and the parent's values); for an array, its length;
   for a scalar, a value given the context. Close to probabilistic context-free grammars and to the "schema-guided"
   JSON generators. Conditioning every node on its context makes the model a Bayesian network over paths whose
   size grows with the schema.
2. **A sequence model over serialised documents** (an autoregressive transformer or a local LLM over JSON tokens,
   as in GReaT, Borisov et al., ICLR 2023, for tables). Flexible, and dependence across levels comes for free.
   It needs a GPU, a lot of documents per shape, constrained decoding to stay valid, and DP-SGD to avoid memorising
   rare documents; it has no calibrated posterior.
3. **Baseline (adopted).** Shred, synthesise with the engine, nest, conform.

## What the baseline already gives

Measured in this thread (T13 report, `benchmarks/documents/run_documents.py`): the inferred schema admits every
source document; `nest(shred(D)) = D` on all ten sample collections; every synthetic document validates; references
resolve 100%; on `sample_supplies` and `sample_analytics` all seven shape metrics are within their calibrated limits.
Keys, references, counts per parent, rules, DP mode, governance and the scorecard come from the engine. A direct
generator would have to rebuild each of these.

## Where the baseline is weak

Dependence between levels. Cramér's V between pairs of fields on 5 000 `sample_supplies` sales, real versus one
synthetic draw (seed 3):

| Pair | Levels | Real | Synthetic |
|---|---|---|---|
| purchaseMethod × couponUsed | document × document | 0.013 | 0.026 |
| items.name × items.quantity band | element × element (same table) | 0.428 | 0.428 |
| purchaseMethod × number of items | document × array length | 0.031 | 0.032 |
| storeLocation × items.name | document × element | 0.010 | 0.016 |
| items.name × items.tags | element × its sub-array | **0.570** | **0.262** |

Dependence within one table (an item's name and quantity) and between a document and its array lengths (the
length column is modelled with the document) are kept. The sample data has almost no dependence between the
document and its elements, so those rows say little. The strong dependence of an item's tags on the item's name is
halved: the tags are a child table whose values depend on the parent row only through the engine's latent segments,
and the segments do not resolve twelve product names.

## Recommendation

- **No-go** on a direct generator now: it would duplicate the engine's guarantees, and the acceptance criteria are
  met without it.
- **Go** on a follow-up (proposed as T13b): *parent-conditioned element columns*. For a child table, add the
  parent's low-cardinality categorical columns (and its length columns) as conditioning columns of the child's
  copula, drawn from the synthetic parent at sampling time. This is a change to the relational engine (an option
  of facts, owner T03/engine), so it goes through an RFC. The measurement above is its acceptance test
  (items.name × tags within its calibrated limit), together with a new shape metric for dependence across levels.
- Revisit a direct generator (candidate 2) when neural generator plug-ins (T11) land, as an optional generator plug-in behind
  the same conform and metrics.
