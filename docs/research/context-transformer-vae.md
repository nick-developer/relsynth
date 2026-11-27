# Context transformer VAE: design record

This record covers the family generator `ctvae` (generator contract 1.1): the question it answers, the
literature it rests on, the ablation that chose it, and its limits. The user guide is [../neural.md](../neural.md),
and the propositions are 1860-1863 in `docs/proofs/sec_v74_ctvae.tex`. The full review, with comments and
decisions, is the design document shared in the project ("Transformer and VAE generators for relational and
sequential data").

## Question

Row-wise generators (copula, cart, tvae, ddpm) draw each row of a fact from its own context: its segment and its
parent attributes. The rows that share an anchor row (a family: the lines of an invoice, the transactions of an
account) are then independent given their segments. Siblings in real data are not independent in that way: they
share categories, their amounts share a level, and sequences are autocorrelated. The task was to model families
jointly, with transformers adapted to the dataset's context and VAEs where useful, using literature-backed methods.
Every step was to be tested before any engine change.

## Literature (concrete findings that shaped the design)

- Relational generators that model children given parents: REaLTabFormer (Solatorio and Dupriez 2023) uses a
  sequence-to-sequence transformer for child rows given the parent. ClavaDDPM (Pang et al. 2024) uses parent
  clusters. RelDiff and GRDM (2025) use graph-conditioned diffusion; GRDM reports a Berka 3-hop score of 93.13
  against 80.98 for the best earlier baseline.
- Latent tabular models: TabSyn (Zhang et al. 2024) trains a VAE with column tokens, then a generative model in its
  latent space. It cut density error by 86% and correlation error by 67% against the strongest baselines.
  FT-Transformer (Gorishniy et al. 2021) gives the column tokenizer and PreNorm blocks.
- VAE stability: free bits (Kingma et al. 2016; Li et al. 2019) and KL warm-up against posterior collapse; an
  active-units monitor (Burda et al. 2016).
- Sequences and visits: HALO (Theodorou et al. 2023) generates hierarchical patient records autoregressively and
  reports same-visit R² 0.362 against 0.262 for a flat GPT. TabBERT and TabGPT (Padhi et al. 2021) model
  transaction sequences. TimeVAE and Diffusion-TS cover regular series, and neural temporal point processes
  (Transformer Hawkes; Shchur et al. 2021 review) cover event timing.
- Privacy: membership inference on TabDDPM reached 94.4% TPR at 1% FPR on 12k rows, and the MIDST challenge
  (2025) targets diffusion models on tabular and relational data. Entity-level DP-SGD is the standard defence.

## Design

The rows of a family are drawn by a row VAE plus a family prior over the row latents (two-stage training, as in
TabSyn):

1. **Row VAE.** Column tokens (numeric value times a learned vector, categorical embeddings), a CLS token, PreNorm
   transformer blocks, then a Gaussian posterior. An MLP decoder gives the numeric scores and categorical logits.
   It is trained with free bits 0.05 per unit, beta 0.5 and a KL warm-up over the first fifth of the steps.
2. **Family prior.** It is causal over the family; the default is a transformer, with GRU as an option. Its inputs
   are the previous row's latent and levels, the row's own segment, coarse segment and parent attributes, log of
   the family size, and the time gap for ordered families. A 5-component mixture density network gives
   p(z_i | history). Unordered siblings are trained in random orders; ordered families follow the time column.
3. **Residual categorical head.** The logits of a level are the decoder's logits for the row latent plus a
   history correction initialised at zero (Prop. 1861).
4. **Marginals.** The columns are encoded and decoded through relsynth's Bayesian marginals, with calibration, as
   tvae does (Props. 1800-1805).

The engine keeps keys, counts and dates. A family generator receives the anchor row of every row and the time
column as order key (contract 1.1), and the engine draws a fact's dates before its attributes (Prop. 1862).

## Ablation (experiments/context_vae, GitHub Actions, CPU)

All runs used 4 datasets, 3 seeds and 3,000 steps per stage. The acceptance rule was fixed before the runs: a
variant must beat the context-only model `transformer:none` on every within-family error on at least 3 of 4
datasets (mean over seeds), keep family and row DCR within none + 0.02, and fit the CI CPU budget. The datasets
were Chinook (invoice lines), Berka (account transactions, ordered), Online Retail II (invoice lines) and
AdventureWorks (a customer's orders, ordered).

| Run | Change | Outcome |
|---|---|---|
| 1 | Chinook and Berka only | Prior halves numeric within-family error. C2ST near 0.99 for every model, traced to continuous scores compared with tied real values. |
| 2 | Scores snapped to the real support; 4 datasets | Numeric and lag-1 errors fall on all 4; categorical sharing worse on Berka and Online Retail: 2 of 4, rule not met. |
| 3 | Categorical head that replaces the decoder (`+cat`) | Fixes Online Retail (0.025 against 0.077) but falls to shuffle level on Chinook (330 training invoices): 1 of 4. |
| 4 | Residual head over the latent route (`+res`) | **3 of 4** (Chinook, Online Retail, AdventureWorks); Berka misses on categorical share only (0.028 against 0.017). DCR and CPU pass. |
| 5 | Same variants after the harness review (codecs fitted on training families only, matching training windows, multi-row numeric share) | **3 of 4** again for `transformer+res` and `gru+res` (same three datasets); Berka misses on categorical share only (0.033 against 0.018). DCR and CPU pass (slowest 410 s per seed). |

Run 4, within-family errors (mean over 3 seeds; lower is better):

| Dataset, error | none | transformer+res | gru+res |
|---|---|---|---|
| Chinook, categorical share | 0.071 | 0.028 | 0.021 |
| Chinook, numeric within | 0.282 | 0.251 | 0.173 |
| Berka, categorical share | 0.017 | 0.028 | 0.024 |
| Berka, numeric within | 0.281 | 0.091 | 0.118 |
| Berka, lag-1 | 0.520 | 0.040 | 0.053 |
| Online Retail, categorical share | 0.056 | 0.046 | 0.047 |
| Online Retail, numeric within | 0.266 | 0.112 | 0.108 |
| AdventureWorks, categorical share | 0.028 | 0.020 | 0.018 |
| AdventureWorks, numeric within | 0.096 | 0.041 | 0.069 |
| AdventureWorks, lag-1 | 0.331 | 0.138 | 0.162 |

Run 5, within-family errors under the corrected harness (mean over 3 seeds; lower is better; GitHub Actions run
45923141764):

| Dataset, error | none | transformer+res | gru+res |
|---|---|---|---|
| Chinook, categorical share | 0.067 | 0.042 | 0.028 |
| Chinook, numeric within | 0.283 | 0.259 | 0.167 |
| Berka, categorical share | 0.018 | 0.033 | 0.027 |
| Berka, numeric within | 0.280 | 0.155 | 0.160 |
| Berka, lag-1 | 0.518 | 0.023 | 0.029 |
| Online Retail, categorical share | 0.052 | 0.045 | 0.032 |
| Online Retail, numeric within | 0.270 | 0.122 | 0.126 |
| AdventureWorks, categorical share | 0.020 | 0.019 | 0.010 |
| AdventureWorks, numeric within | 0.094 | 0.042 | 0.061 |
| AdventureWorks, lag-1 | 0.327 | 0.155 | 0.153 |

In run 4, family DCR stayed between 0.48 and 0.56 on average (the highest single seed was 0.62), and row DCR between 0.50 and
0.53. The slowest variant took 578 s per seed on a hosted runner. In run 5 the variants' family DCR averaged 0.46 to 0.55 and row DCR 0.50 to 0.54, all within none + 0.02.

## Limits

- **Online Retail is a narrow pass.** CPU training is not bit-for-bit reproducible across runs: the baseline's
  categorical error moved between 0.056 and 0.077 with the same code and seeds.
- **AdventureWorks categorical sharing (run 5).** `transformer+res` beat the baseline by 0.001 only (0.019 against
  0.020); `gru+res` has the clearer margin (0.010).
- **Berka categorical sharing.** When an account's context already fixes the transaction type, the context-only
  model is hard to beat, and the prior adds a little noise.
- **Row-level detail.** Row-level C2ST is 0.64 to 0.79 on three datasets for every variant, so the shared row
  decoder loses some joint detail inside a row. This is a target for later work, such as a larger decoder or
  diffusion in latent space as in TabSyn.
- **Exchangeability.** Unordered families are exchangeable only up to the order gap the prior reports (Prop. 1860).
- **Privacy.** DP-SGD is entity-level for one table and conditional on the preprocessing; DP mode remains the
  end-to-end route.
