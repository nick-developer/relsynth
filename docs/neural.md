# Neural generators (`tvae`, `ddpm`)

relsynth 2.3 adds two neural [generator plug-ins](generators.md) for the attribute columns of a table. They are
optional: they need PyTorch, which the core package does not install.

```bash
pip install "relsynth[neural]"          # CPU or GPU build of PyTorch, whichever pip finds
# or, for the CPU build only:
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install relsynth
```

The relational skeleton (rows, keys, links, latent segments, counts, rules) stays relsynth's, whatever the
generator (Prop. 700). A neural generator only models the joint law of a table's own columns given the row's
segment and its parent attributes. The default generator stays `copula`; nothing changes unless a spec names
`tvae` or `ddpm`.

The design record, with the literature survey and the reasons for these two models, is
[research/neural-and-simulation-generators.md](research/neural-and-simulation-generators.md). The propositions are
1800-1805 (`docs/proofs/sec_v70_neural.tex`).

## Choosing one

```yaml
synthesis:
  generator: copula                 # the default for every table
  compute:
    preset: gpu-auto                # optional: use a GPU when there is one (compute.md)

tables:
  customer:
    generator: tvae                 # fast; a good first neural model
  invoice_line:
    role: fact
    generator:
      name: ddpm                    # best fidelity on mixed columns, slower
      options: {max_steps: 30000, sample_steps: 200}
```

| Generator | Model | When to use it | Cost (Adult, 32k rows, 4 CPU cores) |
|---|---|---|---|
| `tvae` | Conditional tabular VAE (Xu et al. 2019; Sohn et al. 2015). | Many columns with non-linear dependence that the copula's single correlation matrix misses; quick runs. | 142 s |
| `ddpm` | Gaussian-multinomial diffusion (TabDDPM, Kotelnikov et al. 2023), conditioned on the row's segment like ClavaDDPM (Pang et al. 2024). | Mixed columns with complex, multi-modal dependence, with a GPU and a training budget; on a CPU `tvae` did better on Adult (design record, section 6). | 735 s; best with a GPU preset and longer training |

Use the quality report (`relsynth evaluate`) and the privacy attacks on your own data to choose: the bench numbers
in the design record are a guide, not a guarantee.

## How they model a table

**Through relsynth's marginals (Prop. 1800).** The networks never see raw values. Each continuous column (numbers,
dates, datetimes) becomes a normal score through the table's Bayesian marginal, the copula's: the randomised
probability integral transform puts an observation uniformly inside its interval of the marginal and maps it to
the normal scale, so ties, point masses and nulls are handled as in the copula. Each discrete column (categories,
flags, integers coded as levels) becomes a one-hot block, with a null level when the column has nulls. When a
column has more than `max_levels` levels, the rare ones share one `other` level, which decodes as a draw from the
marginal restricted to those levels. A synthetic score decodes through the marginal's quantile function, so

- decoding never picks a donor row: continuous values are interpolated between the marginal's quantile knots, so
  an observed number comes out only where it is a knot or a point mass of the marginal (Prop. 1800(c));
- each dataset's marginals are a fresh posterior draw (`synthesis.posterior`), so datasets are posterior predictive
  draws of the marginals given one fitted network (the same split as `cart`, Prop. 706).

**Conditioning (Prop. 1801).** The row's segment and coarse segment enter as learned embeddings, and the parent
attributes of a fact row as encoded inputs (capability `parent`). During training the segment is replaced by a
null token for a share `p_uncond` of the rows (and the coarse segment for half that share), as in classifier-free
guidance training. A segment the fit never saw, such as a sparse fact block passed with its coarse label, is then
sampled through the null token with its coarse segment: the engine's back-off, in neural form.

**Marginal calibration (Prop. 1805).** A network rarely gets each column's marginal law exactly right on its own.
After training, the generator draws `calib_rows` rows with the contexts of the training rows and fits, per
continuous score, a strictly increasing map that makes the aggregate law of the score N(0, 1), and, per discrete
column, logit offsets that bring the aggregate level frequencies close to those of the marginal (two rounds of an
iterative prior-shift correction; approximate, since each context's softmax renormalises). Strictly increasing maps leave the network's rank dependence untouched, so calibration repairs the
marginals without changing what the network learned about dependence. The offsets follow each dataset's posterior
draw of the marginals. `calibrate: false` turns it off.

**Early stopping.** Without DP, a share `valid_share` of the rows is held out and the epoch with the lowest held-out
loss is kept (patience `patience` epochs). This guards against the memorisation that diffusion models show on
small tables (Fang et al. 2025).

## Options

Shared by both:

| Option | Default | Meaning |
|---|---|---|
| `epochs` | tvae 300, ddpm 2000 | Most passes over the rows. |
| `max_steps` | tvae none, ddpm 20000 | Most optimiser steps (whichever limit comes first). |
| `batch_size` | tvae 500, ddpm 1024 | Rows per step. `synthesis.compute.overrides.batch_size` sets it when the option is absent. |
| `lr` | 1e-3 | AdamW learning rate. |
| `weight_decay` | 1e-5 | AdamW weight decay. |
| `max_rows` | 200000 | Rows fitted on (a seeded uniform sample beyond that). |
| `max_levels` | 64 | Levels of a discrete column with a slot of their own; the rest share `other`. |
| `valid_share` | 0.1 | Rows held out for early stopping (0 turns it off; always off with `dp`). |
| `patience` | tvae 30, ddpm 200 | Epochs without a better held-out loss before stopping. |
| `cond_dim` | 16 | Width of the segment embeddings. |
| `p_uncond` | 0.1 | Share of training rows given the null segment token. |
| `sample_batch` | 65536 | Rows generated per batch. |
| `calibrate` | true | Marginal calibration after training (Prop. 1805). |
| `calib_rows` | 20000 | Rows drawn for the calibration. |
| `dp` | none | DP-SGD, see below: `{epsilon, delta, clip, population}`. |

`tvae` only: `hidden` (`[128, 128]`), `latent` (64), `loss_factor` (2.0, TVAE's weight of the reconstruction term).

`ddpm` only: `hidden` (`[256, 512, 512, 256]`), `emb_dim` (128), `timesteps` (1000, cosine schedule),
`sample_steps` (200; the reverse chain runs on that many respaced steps, exact for the coarser chain), `aux_weight`
(1.0; weight of the cross-entropy of the predicted clean category, the hybrid loss of Austin et al. 2021).

## Privacy

The neural generators give no end-to-end formal guarantee by themselves (`privacy_cost().dp` is false). Measure
every release with the privacy attacks (`relsynth.privacy`, run by default as the quick check).

**For a release outside the organisation, use DP mode** (`relsynth run --dp`, [dp.md](dp.md)). Its stage 1 releases
noisy statistics per privacy unit and builds a seed database; stage 2 is the ordinary engine fitted to the seed
database. A neural generator in stage 2 never sees real data, so it is post-processing and keeps the guarantee
(Prop. 905). Benchmarks of DP synthesis on tabular data find marginal-based methods, which DP mode uses, ahead of
DP-trained deep generators at the same budget (Tao et al. 2021).

**DP-SGD (Prop. 1802).** The `dp` option trains the network by DP-SGD (Abadi et al. 2016): rows are Poisson-sampled
at rate `batch_size / rows`, each row's gradient is clipped to L2 norm `clip`, the clipped gradients are summed and
Gaussian noise is added. The noise multiplier is chosen so that the Rényi accountant (Mironov, Talwar and Zhang
2019, converted by Canonne, Kamath and Steinke 2020) gives `epsilon` at `delta` after all steps (default delta:
1/(10 rows), at most 1e-5). The accountant agrees with Opacus to 1e-11 (Prop. 1803).
The row count that sets the rate, delta and the gradient scale is treated as public, as in Abadi et al. (2016) and
Opacus. To remove that assumption, set `population` to a public figure (for example a published number of
customers): every mechanism parameter is then computed from it instead of the actual count.

```yaml
generator:
  name: tvae
  options: {dp: {epsilon: 8, clip: 1.0}, epochs: 50}
```

What this covers, and what it does not:

- It bounds what the **network weights** reveal about any one **row of this table**, for a fixed category set and
  fixed null presence. It is row-level, not per person: a person with many rows, or with rows in several tables,
  is covered only by DP mode.
- The levels kept and whether a column has nulls set the shapes of the weight tensors, so a row that adds a new
  level (below `max_levels`) or a column's first null changes the shapes, which the stored weights show. The bound
  does not cover such changes.
- The rare-column selection (`marginals.min_dependence_rows`) is skipped under DP-SGD: which columns stay out of
  the network depends on the rows and would change the input width, so every column is learned.
- The encoding marginals, the category levels kept and the latent segments are fitted without noise. The
  guarantee is therefore not end to end, and the generator reports it with `dp: false` and the spent epsilon and
  delta in `privacy_cost()` and the model summary.
- As in DP mode, the noise and the sampling come from the operating system's CSPRNG (`os.urandom`) and are never seeded,
  so a DP-SGD fit is not bit-reproducible. Early stopping is off, since choosing an epoch on held-out rows would
  spend budget that is not accounted.
- The noise is floating-point Gaussian noise; the attacks on floating-point noise (Mironov 2012) apply in principle.

Use DP-SGD as defence in depth for releases inside the organisation, and DP mode for anything that leaves it.

## Families: `ctvae`

`tvae` and `ddpm` draw each row given its own context, so the rows that share an anchor row are independent given
their segments: the genres of one invoice, or the balances of one account over time, do not go together the way
they do in the data. `ctvae` (context transformer VAE) draws the rows of a **family** jointly. A
family is the rows of a fact that share an anchor row. When the fact has a time column, the family is ordered by it.

```yaml
tables:
  trans:
    role: fact
    fact: {anchor: account, time_column: date}
    generator:
      name: ctvae
      options: {prior: transformer}      # transformer (default), gru (cheaper) or none
```

How it works (Props. 1860-1863, `docs/proofs/sec_v74_ctvae.tex`):

- A row VAE maps each row to a latent: column tokens and a PreNorm transformer over them (FT-Transformer, TabSyn's
  VAE), with free bits and a KL warm-up.
- A family prior draws the latents of a family one row after the other. A causal transformer reads the earlier rows
  (latents and levels), each row's own segment and parent attributes, the family size and the time gaps. A mixture
  density network gives the next latent.
- A residual categorical head draws each row's levels from the decoder's logits for its latent, plus a correction
  from the family history. The correction starts at zero, so the model starts as the latent-only one (Prop. 1861).
- The columns go through relsynth's Bayesian marginals and the calibration, as in `tvae` (Props. 1800-1805).
- The engine keeps keys, counts and dates. For a family generator it draws a fact's dates before its attributes,
  which leaves the law of the dates unchanged because the date law does not read the attributes (Prop. 1862).

The model was chosen by an ablation before any engine code: four real datasets (Chinook, Berka, Online Retail II,
AdventureWorks), three seeds each, and an acceptance rule fixed before the runs. With `transformer+res`
under the corrected harness (run 5), within-family numeric errors fell by 8% to 55% across the four datasets and
lag-1 errors on the two ordered ones by 53% to 96%, and categorical sharing was kept or improved on three of the four,
as in run 4. Every variant met the rule's family
and row DCR limits on these runs; DCR is a measurement, not a privacy guarantee. The design record is [research/context-transformer-vae.md](research/context-transformer-vae.md);
the harness is `experiments/context_vae/`.

Options (defaults are the ones of the ablation): `prior`, `categorical_head` (`residual` or `decoder`), `latent`,
`d_column`, `layers`, `heads`, `d_prior`, `prior_layers`, `mixtures`, `window` (rows of history the prior reads),
`beta`, `free_bits`, `vae_steps`, `prior_steps`, `batch_size`, `batch_families`, `lr`, `weight_decay`,
`max_rows`, `max_levels`, `cond_dim`, `sample_chunk`, `calibrate`, `calib_families` and `dp`.

What to know:

- `ctvae` cannot fit from in-database statistics and a sample, because a row sample would cut families. Pushdown
  facts need another generator, and the spec check says so.
- On a table that is not a fact with an anchor, every row is a family of one row, and the spec check warns.
- Rows that the engine rejects and redraws are drawn again given the kept rows of their family, which the prior reads
  as history.
- With `dp`, DP-SGD runs on families: families are Poisson-sampled, and each family's gradient over at most
  `window` of its rows is clipped as one unit. The bound covers one anchor row's whole history in this table
  (Prop. 1863), but is conditional on the preprocessing, as for `tvae`. The family count sets the sampling rate
  and is treated as public unless `dp.population` gives a public number of families. The family calibration reads the training
  families' sizes and contexts, which the accountant does not cover, so with `dp` it is off unless the spec sets
  `calibrate: true`; it then joins the preprocessing outside the bound.
- On a CPU, a fit with the default steps takes a few minutes on 100k rows. Sampling is one step per row of the
  longest family in a chunk, so very long families are the slow case. `prior: gru` is about twice as fast on long
  sequences.

## Reproducibility and devices

On the CPU a seed gives the same model and data (Prop. 701): every random number comes from torch generators seeded
from the engine's generator, inside a fork of torch's global random state, which is left as it was. On a GPU
results are reproducible up to floating point. The device comes from the compute preset ([compute.md](compute.md));
both generators declare `gpu: optional` and train in fp32 (`precision: fp32` or `tf32`).

## Saving models

`serialize()` stores the network weights as numpy arrays with the rest of the fitted state, and `load(blob,
trusted=True)` rebuilds the network. As for every generator, model files are pickles: load only files you wrote.
The file holds no source rows, so a loaded model cannot redraw its marginals from the posterior: every dataset
decodes through the marginals saved with it, and `uncertainty()` reports `posterior_frozen: true`.

## References

- L. Xu, M. Skoularidou, A. Cuesta-Infante, K. Veeramachaneni (2019). Modeling tabular data using conditional GAN.
  *NeurIPS* (CTGAN and TVAE).
- K. Sohn, H. Lee, X. Yan (2015). Learning structured output representation using deep conditional generative
  models. *NeurIPS*.
- J. Ho, A. Jain, P. Abbeel (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- E. Hoogeboom, D. Nielsen, P. Jaini, P. Forré, M. Welling (2021). Argmax flows and multinomial diffusion. *NeurIPS*.
- A. Nichol, P. Dhariwal (2021). Improved denoising diffusion probabilistic models. *ICML*.
- J. Austin, D. D. Johnson, J. Ho, D. Tarlow, R. van den Berg (2021). Structured denoising diffusion models in
  discrete state-spaces. *NeurIPS*.
- A. Kotelnikov, D. Baranchuk, I. Rubachev, A. Babenko (2023). TabDDPM: Modelling tabular data with diffusion
  models. *ICML*.
- W. Pang, M. Shafieinejad, L. Liu, S. Hazlewood, X. He (2024). ClavaDDPM: Multi-relational data synthesis with
  cluster-guided diffusion models. *NeurIPS*.
- J. Ho, T. Salimans (2022). Classifier-free diffusion guidance. arXiv:2207.12598.
- Z. Fang et al. (2025). Understanding and mitigating memorization in diffusion models for tabular data. *ICML*.
- M. Saerens, P. Latinne, C. Decaestecker (2002). Adjusting the outputs of a classifier to new a priori
  probabilities: a simple procedure. *Neural Computation* 14(1).
- M. Abadi et al. (2016). Deep learning with differential privacy. *ACM CCS*.
- I. Mironov (2017). Rényi differential privacy. *IEEE CSF*. I. Mironov, K. Talwar, L. Zhang (2019). Rényi
  differential privacy of the sampled Gaussian mechanism. arXiv:1908.10530.
- C. Canonne, G. Kamath, T. Steinke (2020). The discrete Gaussian for differential privacy. *NeurIPS*.
- Y. Tao, R. McKenna, M. Hay, A. Machanavajjhala, G. Miklau (2021). Benchmarking differentially private synthetic
  data generation algorithms. arXiv:2112.09238.
- L. Rüschendorf (2009). On the distributional transform, Sklar's theorem, and the empirical copula process.
  *Journal of Statistical Planning and Inference* 139(11).
- I. Mironov (2012). On significance of the least significant bits for differential privacy. *ACM CCS*.
