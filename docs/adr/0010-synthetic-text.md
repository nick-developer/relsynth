# ADR 0010: Synthetic text

- Status: proposed (roadmap thread T14, release 5.0)
- Date: 2026-10-31

## Context

relsynth 1.0 synthesises free text with a word-level Dirichlet back-off n-gram model conditioned on one column or the row's latent segment (Props. 12, 13). Its memorisation guards count **rows**: a token or an 8-gram must occur in at least three rows. The roadmap (T14) asks for text that agrees with the row, a choice of stronger generators (a local language model with DP fine-tuning, Private Evolution), a PII layer with measured recall, a leakage audit, a licence register for models, and a fully offline mode. The organisation's decisions bind the design: DP for every external release with ε set by the organisation, no bundled weights, GPUs only through compute presets.

## Options

1. **Local LLM, DP fine-tuning** (Yue et al. ACL 2023): fine-tune a pretrained causal LM with DP-SGD (Abadi et al. 2016), often on LoRA adapters. Best fluency; needs PyTorch, a pretrained model and, above a few hundred million parameters, a GPU. User-level DP needs per-person clipping (McMahan et al. 2018).
2. **Private Evolution** (Lin et al. 2024; Xie et al., Aug-PE, ICML 2024): only inference of a public generator plus DP nearest-neighbour votes. No training on private data; quality bounded by the public generator; cost in generator calls.
3. **DP n-gram release** (Kim et al. NeurIPS 2021): noisy, thresholded counts of n-grams per person and class. Runs anywhere, composes exactly with the zCDP accountant of DP mode; limited fluency.
4. **Keep only the 1.0 model** with stronger guards. Not DP; the guards cannot bound what one person reveals.

## Decision

Ship all three as methods of a text column (`text.method: ngram | lm | pe`, `text.dp`), with the 1.0 model as the default and byte-identical. Every DP mechanism is a Gaussian mechanism on a full batch or a histogram with user-level sensitivity, charged to the DP-mode `Accountant` in zCDP, so the charges compose with the ledger without a new accountant: the DP n-gram release (level-wise partition selection, Prop. 1501), DP-GD without subsampling (Prop. 1502; full batches are near-optimal for DP training, De et al. 2022, Sander et al. 2023), and the PE vote histograms (Prop. 1503). The PII layer (regex built in; Presidio, spaCy, GLiNER optional) redacts before training and fills typed surrogates after generation. The audit measures canary exposure against the DP bound (Props. 1500, 1504), copies (Prop. 1505) and extraction.

## Consequences

- No weights in the repository or the wheel; models load from local folders (`local_files_only`). The register `src/relsynth/data/text_models.yaml` records each model's licence; Apache-2.0 and MIT models are preferred.
- PyTorch, transformers and peft are optional (`relsynth[text-lm]`); without them, the `ngram` methods and the audit run on CPU only.
- Subsampled DP-SGD (privacy amplification by sampling) needs an RDP or PRV accountant and a ledger entry that is not zCDP; it is left to RFC-T14-01, together with text columns inside DP mode.
- A `dp` column inside an ordinary run protects that column only; the run log states the scope.
