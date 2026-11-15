# Neural and simulation generators: survey and design record

Status: design adopted for 2.3 (generators `tvae`, `ddpm`); simulation mechanics proposed as follow-up work.
Propositions 1800-1804 (`docs/proofs/sec_v70_neural.tex`). User guide: [docs/neural.md](../neural.md).

## 1. Summary

- relsynth keeps its split: the **engine** owns the relational skeleton (rows, keys, links, latent segments,
  counts, rules, scaling); a **generator** owns the joint law of one table's attribute columns given the row's
  segment and parent attributes (contract 1.0, Props. 700-705). Neural models enter as generators, so keys, links
  and rules stay exact whatever the network learns.
- From the tabular generative-model literature we adopt two models, both behind the optional extra
  `relsynth[neural]` (PyTorch only; no new dependency beyond the `text-lm` extra's):
  - **`tvae`**, the conditional tabular VAE (Xu et al. 2019): stable, fast on the CPU, the strongest of the
    first-generation deep tabular models in its authors' own benchmark, and small enough for DP-SGD.
  - **`ddpm`**, Gaussian-multinomial diffusion (TabDDPM, Kotelnikov et al. 2023), conditioned on the row's latent
    segment the way ClavaDDPM (Pang et al. 2024) conditions on parent clusters. Diffusion is the family that leads
    the recent fidelity benchmarks on mixed-type tables.
- Both model the **dependence** only: every column goes through relsynth's own Bayesian marginal (Prop. 1800), so
  marginals stay Bayesian, nulls and point masses are handled as in the copula, continuous values are never copied
  from a row, and posterior predictive datasets keep working.
- Privacy: the formal route for releases outside the organisation stays **DP mode** (stage 1 noisy statistics,
  stage 2 any generator as post-processing, Prop. 905). DP-SGD is available inside both generators with an exact
  Rényi accountant (checked against Opacus), but it covers the network weights only and is reported as such.
- Defaults do not change. `copula` remains the default generator; neural generators are chosen per table in the spec.
- **Simulation mechanics** (scenario interventions, self-exciting and multi-state event processes, agent-style
  lifecycles) belong to the engine's time and count models, not to the attribute contract. Section 7 proposes them
  as separate work.

## 2. The baseline these models must beat

| Piece | What it does today | Where neural models can help |
|---|---|---|
| `copula` (default) | Bayesian marginals per segment, rank-likelihood Gaussian copula per segment (entities) or pooled (facts). | One correlation matrix per segment: monotone, elliptical dependence. Interactions (a category changing the slope between two numbers), multi-modal joint shapes and high-order categorical associations are only captured through the segments. On Adult the copula fails the bench's `kway3_tvd_mean`, `kway2_tvd_max`, `association_max_diff` and `beta_recall_gap` checks (section 6). |
| `cart` | Sequential CART, leaf donors with Bayesian-bootstrap weights. | Strong on interactions, but releases observed categories from small leaves and is weak on tiny dimensions (T03 report). |
| `independent` | Marginals given the segment only. | Reference only. |
| DP mode | Per-unit DP seed database (MST/AIM marginals, correlations), then the ordinary engine. | Stage 2 can be any generator, neural included, at no extra privacy cost. |
| Bench (G7) | 31 datasets in four tiers, quality report 2.0, privacy attacks (singling out, linkage, inference, DOMIAS), quick check. | Gives the yardstick used in section 6. |

## 3. Survey of model families

### 3.1 GANs
CTGAN (Xu et al. 2019) introduced mode-specific normalisation and training-by-sampling for imbalanced categories;
CTAB-GAN and CTAB-GAN+ (Zhao et al. 2021, 2024) extend it. GAN training is adversarial and unstable, needs many
heuristics, and Xu et al.'s own benchmark found TVAE ahead of CTGAN on several datasets. Under DP, GANs trained
with DP-SGD (DP-CTGAN, PATE-GAN) were dominated by marginal-based methods in the independent benchmark of Tao et
al. (2021). **Not adopted**: TVAE covers the same ground with a likelihood objective, which also gives a held-out
loss for early stopping.

### 3.2 Variational autoencoders
TVAE (Xu et al. 2019) is a VAE with a Gaussian likelihood per continuous column and a categorical one per discrete
column. It trains by maximum likelihood (the ELBO), is stable, cheap and has few parameters, which matters for
DP-SGD, whose noise grows with the number of steps and whose utility falls with model size. **Adopted as `tvae`**,
as a conditional VAE (Sohn et al. 2015) on the row's context.

### 3.3 Diffusion models
TabDDPM (Kotelnikov et al. 2023) applies Gaussian diffusion (Ho et al. 2020) to numbers and multinomial diffusion
(Hoogeboom et al. 2021) to categories with one MLP denoiser, and outperformed GAN and VAE baselines on most of its
15 datasets for both fidelity and machine-learning efficiency. TabSyn (Zhang et al. 2024) runs diffusion in the
latent space of a VAE; TabDiff (Shi et al. 2025) learns per-column noise schedules. They report further gains, but
add a second model (TabSyn) or learned schedules (TabDiff) on top of the same backbone. ClavaDDPM (Pang et al. 2024)
extends TabDDPM to multi-table data by clustering parent-child pairs and conditioning each child table's diffusion
model on its parent's cluster: relsynth's latent segments are exactly such clusters (fitted jointly with the
relational structure), so a TabDDPM conditioned on the row's segment is the ClavaDDPM pattern on relsynth's
skeleton. **Adopted as `ddpm`**. TabSyn and TabDiff are possible later versions behind the same plug-in contract.

Privacy caveat: diffusion models memorise training rows, more so on small tables and with long training (Fang et
al. 2025). The design answers with early stopping on a held-out loss, decoding through marginals (no copied
numbers), the default quick privacy check, and optional DP-SGD.

### 3.4 Autoregressive transformers and language models
GReaT (Borisov et al. 2023) fine-tunes a language model on rows written as text; REaLTabFormer (Solatorio and
Dupriez 2023) adds a sequence-to-sequence model for parent-child tables. They need a pre-trained language model,
GPU time, careful sampling to stay inside each column's domain, and have shown verbatim copying of training rows.
**Deferred**: relsynth's text stack (T14) already runs local language models with DP fine-tuning; a row-as-text
generator can reuse it once its leakage audit applies to tabular rows.

### 3.5 Flows and neural copulas
Copula flows (Kamthe et al. 2021) keep marginals separate and model the copula with a normalising flow, the same
split relsynth uses for its Gaussian copula. Our Prop. 1800 design is that split with a VAE or diffusion model in
place of the flow, which handles categorical columns natively; a flow generator is a later option.

### 3.6 Differential privacy for deep generators
DP-SGD (Abadi et al. 2016) clips per-row gradients and adds Gaussian noise; Rényi DP (Mironov 2017) with the exact
analysis of the sampled Gaussian mechanism (Mironov, Talwar and Zhang 2019) and the conversion of Canonne, Kamath and
Steinke (2020) gives the tightest standard accounting. For tabular data, Tao et al. (2021) and McKenna et al. (2022,
AIM) found marginal-based mechanisms clearly ahead of DP-SGD-trained deep generators at equal budgets, which is why
relsynth's DP mode measures marginals. We therefore:
1. keep DP mode as the formal route (any generator in stage 2 is post-processing of the DP seed database);
2. implement DP-SGD inside the neural generators for in-organisation defence in depth, reported with its scope
   (row-level, weights only, preprocessing not covered; Prop. 1802).

Wiring DP-SGD into the governance ledger as an alternative stage 1 would require DP preprocessing (public column
domains or stage-1 noisy marginals as the encoding) and per-unit contribution bounding across tables. This is
listed as an open decision (section 8).

### 3.7 Simulation
Discrete-event simulation (Law 2015) and agent-based modelling (Macal and North 2010) generate data by running a
mechanism forward in time; temporal point processes, from Poisson and renewal processes to self-exciting Hawkes
processes (Hawkes 1971; simulated by Ogata's thinning, 1981), model when events happen; multi-state (semi-)Markov
models describe entities moving between states (Andersen and Keiding 2002). relsynth already fits a hierarchical
gamma-renewal model of each entity's event history with seasonal intensities (T12). Simulation answers questions
the attribute contract cannot: "what if churn doubles", "what does a launch week look like", "produce three years
of history consistent with these rules". Section 7 maps these to the engine.

## 4. Decision matrix

| Family | Fidelity evidence on mixed tables | Memorisation / privacy | CPU cost | Dependency | Fits the contract | Decision |
|---|---|---|---|---|---|---|
| CTGAN / CTAB-GAN+ | moderate; below TVAE on several sets (Xu et al.) | moderate | medium | torch | yes | not adopted |
| **TVAE** | good; best of the 2019 baselines | lower (smooth decoder; small model) | low | torch | yes | **`tvae`** |
| **TabDDPM** | best of its benchmark | higher (Fang et al. 2025): needs guards | high (training and sampling) | torch | yes | **`ddpm`** |
| TabSyn / TabDiff | higher still in their papers | as TabDDPM | higher | torch | yes | later |
| ClavaDDPM | multi-table SOTA in its paper | as TabDDPM | high | torch | relational part = relsynth segments | pattern used in `ddpm` |
| GReaT / REaLTabFormer | good, needs an LLM | high (verbatim copies) | very high | transformers | yes | deferred (reuse T14) |
| Copula flows | good on continuous data | moderate | medium | torch | yes | later |
| DP-SGD generators | below marginal methods under DP (Tao et al.) | formal for the weights | +per-row gradients | torch | yes | option inside `tvae`/`ddpm` |

## 5. Design

**Feature space (Prop. 1800).** Continuous columns become normal scores through the table's posterior-mean
marginal by the randomised probability integral transform; discrete columns become one-hot blocks with a null
level and an `other` level for the levels beyond `max_levels` (decoded from the marginal restricted to the rare
levels). Scores decode through a posterior draw of the marginal, so continuous values are interpolated between
quantile knots and never copied from a row, and each dataset has its own marginal draw.

**Conditioning (Prop. 1801).** Learned embeddings of the row's segment and coarse segment, plus the encoded parent
attributes. A null token replaces the segment for a share `p_uncond` of training rows (classifier-free guidance
training, Ho and Salimans 2022), so unseen segments back off to their coarse segment, as the engine's copula does.

**`tvae`.** Encoder and decoder MLPs (128, 128), latent 64, Gaussian likelihood per score with learned scales,
categorical likelihood per discrete column, reconstruction weight 2 (TVAE's `loss_factor`). Sampling draws from the
decoder's likelihoods.

**`ddpm`.** MLP denoiser (256, 512, 512, 256) with sinusoidal step embedding and the conditioning added to the
input embedding; cosine schedule, 1000 steps; Gaussian noise prediction plus multinomial KL terms and the
cross-entropy of the predicted clean category (the hybrid loss of Austin et al. 2021; without it every pair of
categorical columns of Adult came out independent, because the KL terms carry a signal of order beta_t); sampling on
`sample_steps` respaced steps with exact posteriors for the coarser chain.

**Training.** AdamW; without DP, early stopping on a 10% held-out loss; with DP, DP-SGD with Poisson sampling,
per-row clipping via `torch.func.vmap`, calibrated noise and no early stopping (Prop. 1802).

**Engineering.** torch is imported only when a neural generator fits (`relsynth generators list` works without
it, and `relsynth check` reports the install command). Seeds give identical CPU results and leave torch's global
state untouched (Prop. 1804). Model files store weights as numpy arrays. Devices and batch size come from
`compute.yaml`. No file or network I/O during fit or sample.

## 6. Bench results

Adult (UCI, 32,561 rows, 15 columns; tier `pr`), seed 1, default options, 4 CPU cores. Lower is better except
`area:*` and where noted; the bench limits are in the second column.

RESULTS_TABLE

Reading: RESULTS_READING

The CI job `neural` reruns Adult with both generators on every pull request and uploads the records.

## 7. Simulation mechanics: proposal

Simulation belongs where relsynth decides *which rows exist and when*: counts, link degrees and event times. Three
packages, each driven by the spec, each fitted from data with the user able to override every parameter:

1. **Scenarios (interventions).** A `scenarios:` block that edits fitted parameters before sampling: scale a table
   or a segment's share, shift a column's marginal (a quantile map), multiply an event intensity over a date range,
   change a link degree law. Formally a do-intervention on the fitted generative model (Pearl 2009), recorded in the
   model log. Cheap, high value for test data and stress tests.
2. **Event processes beyond renewal.** Add a self-exciting (Hawkes) option to the T12 sequence model for bursty
   histories (Bayesian fit of the exponential kernel; Ogata thinning), and multi-state models for status columns
   that move through states (orders: placed, shipped, returned) with transition intensities per segment.
3. **Agent-style lifecycles.** Entities simulated forward through a state machine whose transitions come from the
   multi-state model, emitting fact rows as events. This is the discrete-event core; it reuses 2.

Recommendation: run 1 and 2 as one follow-up thread (spec, engine, bench on AdventureWorks/Contoso/Electricity,
props in the 1820s), and 3 after them.

## 8. Open decisions

1. **DP-SGD in the ledger.** Keep DP-SGD as reported defence in depth (built), or make a DP-SGD generator an
   alternative DP stage 1 with ledger charges (needs public domains or DP marginals for the encoding and per-unit
   contribution bounding). Recommended: keep as built; DP mode already gives neural generators a formal guarantee.
2. **Simulation follow-up.** Start the scenario and event-process thread (section 7). Recommended: yes.
3. **Next neural models.** TabSyn or a copula flow behind the same contract once the bench shows where `ddpm` falls
   short. Recommended: decide after a weekly-tier run.

## References

- M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang (2016). Deep learning with
  differential privacy. *ACM CCS*.
- P. K. Andersen, N. Keiding (2002). Multi-state models for event history analysis. *Statistical Methods in Medical
  Research* 11(2).
- V. Borisov, K. Seßler, T. Leemann, M. Pawelczyk, G. Kasneci (2023). Language models are realistic tabular data
  generators. *ICLR*.
- C. Canonne, G. Kamath, T. Steinke (2020). The discrete Gaussian for differential privacy. *NeurIPS*.
- Z. Fang et al. (2025). Understanding and mitigating memorization in diffusion models for tabular data. *ICML*
  (PMLR 267).
- A. G. Hawkes (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1).
- J. Ho, A. Jain, P. Abbeel (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- J. Ho, T. Salimans (2022). Classifier-free diffusion guidance. arXiv:2207.12598.
- E. Hoogeboom, D. Nielsen, P. Jaini, P. Forré, M. Welling (2021). Argmax flows and multinomial diffusion. *NeurIPS*.
- S. Kamthe, S. Assefa, M. Deisenroth (2021). Copula flows for synthetic data generation. arXiv:2101.00598.
- J. Austin, D. D. Johnson, J. Ho, D. Tarlow, R. van den Berg (2021). Structured denoising diffusion models in
  discrete state-spaces. *NeurIPS*.
- A. Kotelnikov, D. Baranchuk, I. Rubachev, A. Babenko (2023). TabDDPM: Modelling tabular data with diffusion
  models. *ICML*.
- A. M. Law (2015). *Simulation Modeling and Analysis*, 5th ed. McGraw-Hill.
- C. M. Macal, M. J. North (2010). Tutorial on agent-based modelling and simulation. *Journal of Simulation* 4(3).
- R. McKenna, B. Mullins, D. Sheldon, G. Miklau (2022). AIM: An adaptive and iterative mechanism for differentially
  private synthetic data. *PVLDB* 15(11).
- I. Mironov (2017). Rényi differential privacy. *IEEE CSF*.
- I. Mironov, K. Talwar, L. Zhang (2019). Rényi differential privacy of the sampled Gaussian mechanism.
  arXiv:1908.10530.
- A. Nichol, P. Dhariwal (2021). Improved denoising diffusion probabilistic models. *ICML*.
- Y. Ogata (1981). On Lewis' simulation method for point processes. *IEEE Trans. Information Theory* 27(1).
- W. Pang, M. Shafieinejad, L. Liu, S. Hazlewood, X. He (2024). ClavaDDPM: Multi-relational data synthesis with
  cluster-guided diffusion models. *NeurIPS*.
- J. Pearl (2009). *Causality*, 2nd ed. Cambridge University Press.
- L. Rüschendorf (2009). On the distributional transform, Sklar's theorem, and the empirical copula process.
  *Journal of Statistical Planning and Inference* 139(11).
- J. Shi, M. Xu, H. Hua, H. Zhang, S. Ermon, J. Leskovec (2025). TabDiff: a mixed-type diffusion model for tabular
  data generation. *ICLR*.
- K. Sohn, H. Lee, X. Yan (2015). Learning structured output representation using deep conditional generative
  models. *NeurIPS*.
- A. V. Solatorio, O. Dupriez (2023). REaLTabFormer: Generating realistic relational and tabular data using
  transformers. arXiv:2302.02041.
- Y. Tao, R. McKenna, M. Hay, A. Machanavajjhala, G. Miklau (2021). Benchmarking differentially private synthetic
  data generation algorithms. arXiv:2112.09238.
- L. Xu, M. Skoularidou, A. Cuesta-Infante, K. Veeramachaneni (2019). Modeling tabular data using conditional GAN.
  *NeurIPS*.
- H. Zhang, J. Zhang, B. Srinivasan, Z. Shen, X. Qin, C. Faloutsos, H. Rangwala, G. Karypis (2024). Mixed-type
  tabular data synthesis with score-based diffusion in latent space. *ICLR*.
- Z. Zhao, A. Kunar, R. Birke, L. Y. Chen (2021). CTAB-GAN: Effective table data synthesizing. *ACML*.
