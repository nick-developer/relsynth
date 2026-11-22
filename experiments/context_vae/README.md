# Context VAE experiments

Research code, not part of the package. It tests the design in the review
"Transformer and VAE generators for relational and sequential data" before any engine change:
a row VAE (column tokens, PreNorm transformer) plus a family prior over row latents (none, GRU
or causal transformer) that fills the rows of one family (the lines of an invoice, the
transactions of an account) jointly. As in the engine, the family's size, parent context and
dates are given.

## Running

```
python -m experiments.context_vae.run --dataset berka --seed 1 --out out/berka-1.json
python -m experiments.context_vae.summarize out/*.json
```

On GitHub Actions, the `experiments` workflow runs one job per dataset and seed. Start it by
hand once it is on `main`, or push a commit whose message contains `[experiments]`. The
summary table is in the run's summary.

## Variants

| Variant | Row encoder | Family prior |
|---|---|---|
| `transformer:none` | column-token transformer | none: rows independent given the parent (the tvae-like baseline) |
| `transformer:gru` | column-token transformer | GRU over the family |
| `transformer:transformer` | column-token transformer | causal transformer over the family |
| `mlp:transformer` | MLP | causal transformer |
| `…:transformer+cat`, `…:gru+cat` | as above | the prior also draws each row's categorical levels: p(c_i given history) then p(z_i given history and c_i); the decoder gives only numerics |
| `…:transformer+res`, `…:gru+res` | as above | latent route kept: p(z_i given history), then levels with logits = row decoder's logits for z_i plus a history correction (zero at start) |
| `shuffle` | none | real rows permuted across families: exact marginals, no within-family dependence |

`+cat` (run 3) follows the second run, where every latent-only prior matched numeric and sequence
structure but lost categorical sharing on Berka and Online Retail: levels decoded independently
from noisy latents break ties that siblings share. Autoregressing discrete codes directly is what
HALO (Theodorou et al. 2023) and REaLTabFormer (Solatorio and Dupriez 2023) do; the latent then
only carries the numerics and their link to the row's levels. The factorisation is exact (chain
rule), so it does not change what the model can represent, only where the categorical
uncertainty lives. The acceptance rule below is unchanged and still compares with
`transformer:none`.

`+res` (run 4) follows run 3, where `+cat` fixed Online Retail but fell to shuffle level on Chinook
(about 330 training invoices), while the latent-only prior kept Chinook and AdventureWorks. It
keeps the latent route and adds, per categorical column, a correction to the decoder's logits from
the family history (residual logits, a product of experts in probability). The correction starts
at zero, so training starts from the latent-only model and fitting the extra term can only lower
the teacher-forced negative log-likelihood from that starting point.

## Metrics

- Within family: the probability that two siblings share a level (per categorical) and the
  within-family share of variance (per numeric). The error is the absolute difference from the
  real training families.
- Sequences: lag-1 autocorrelation of each numeric along the history, and the TVD between
  real and synthetic transition matrices of each categorical.
- C2ST-Agg: accuracy of xgboost separating real from synthetic family summaries (size, mean
  and std, distinct-level share, modal share, parent context). 0.5 means indistinguishable.
- Privacy: DCR share of families (within size buckets; the engine-set size and context are
  excluded) and of rows. This is the share of synthetic items closer to a training item than
  to a held-out item, with equal-size reference sets; 0.5 means no closer to training data.
- Diagnostics: active latent units, and prior NLL on training against held-out families
  (overfitting). For unordered families, the NLL gap between two orders (order sensitivity).

## Acceptance rule (fixed before the runs)

A variant moves into the engine only if all three hold:

1. It beats `transformer:none` on the within-family errors on at least 3 of 4 datasets, over 3 seeds.
2. Its family and row DCR shares are no worse than `transformer:none` plus 0.02.
3. Its CPU time fits the CI budget.

Otherwise the negative result is recorded and no generator is added.
