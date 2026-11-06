# Synthetic text

relsynth synthesises free-text columns (ticket bodies, notes, comments) so that each text fits its row: a ticket in the `billing` category talks about billing. You choose a method per column, set a privacy guarantee if the text leaves the organisation, and measure what the output still reveals with a leakage audit. Proofs are Props. 1500–1507 in [docs/proofs](proofs/) (section v5.0); the design record is [ADR 0010](adr/0010-synthetic-text.md); measured numbers are in the T14 report.

## Choosing a method

| Method | Needs | Privacy | Quality | When |
|---|---|---|---|---|
| `ngram` (default) | nothing | memorisation guards per **row** (not DP) | keeps vocabulary, phrases and the link to the row; no meaning | internal test data; the 1.0 behaviour, byte for byte |
| `ngram` + `dp` | nothing (air-gapped, CPU) | user-level DP | fewer rare words; shorter phrases at small ε | any release that needs DP and runs without models |
| `lm` | PyTorch, transformers and a local model folder | not DP | the best fluency; may memorise | internal data where a model is allowed |
| `lm` + `dp` | as `lm`; a GPU for models above ~300M parameters | user-level DP (DP-GD) | fluent at moderate ε with a pretrained model | external releases with a model |
| `pe` (+ `dp`, always) | a public generator: a local pretrained model, or a public corpus | user-level DP; the model never trains on private text | as good as the public generator lets it | no training allowed, or a strong public model |

The 1.0 guards count **rows**, not people: a person who writes the same rare sentence in three tickets passes the "three rows" guard. The audit shows this (one person with four copies of a canary reaches a mean exposure of about 9 of 12 bits on the S6 tickets). Use `dp` when people may repeat themselves.

## Spec

```yaml
tables:
  ticket:
    columns:
      body:
        type: text
        text:
          condition_on: [category, priority]      # one column or several; default: the row's latent segment
          method: ngram                           # ngram | lm | pe
          pii: {recognizers: [regex]}             # redact before training (below)
          dp:
            epsilon: 2.0                          # no default: the organisation sets it
            delta: 1.0e-6
            unit_column: customer_id              # the person each text belongs to
            max_texts: 4                          # texts kept per person
          lm:                                     # method lm
            model: /models/SmolLM2-135M           # a local folder; never downloaded
            lora: {r: 8, alpha: 16}               # adapters (needs peft); omit to train every weight
            steps: 200
            lr: 0.001
          pe:                                     # method pe
            api: {kind: lm, model: /models/SmolLM2-135M}
            # or {kind: ngram, corpus: public.csv, column: text, class_column: category}
            iterations: 4
            pool: 64
```

Every key is in the [spec reference](spec-reference.md#textdp). A column without `method`, `dp`, `pii`, `lm`, `pe` and with a single `condition_on` runs the 1.0 model unchanged.

**Scope of `dp` on a column.** It protects the people of `unit_column` in *this column*. The other columns of an ordinary run are not differentially private, and the run log says so (`decisions.<table>.<column> text.dp.scope`). In DP mode (`relsynth run --dp`) text columns are still left empty; joining the column's budget to the DP-mode accountant is proposed in RFC-T14-01.

## Methods

**DP n-gram** (Prop. 1501). Each person keeps at most `max_texts` texts of at most `max_tokens` tokens. Level by level (words, then pairs, then triples), relsynth counts how many *people* use each n-gram in each class, adds discrete Gaussian noise (OpenDP) and keeps the n-grams above a threshold (DP n-gram extraction, Kim et al. NeurIPS 2021, with the partition-selection threshold of Prop. 903). A longer n-gram is only counted when its parts were released. The released counts replace the counts of the 1.0 back-off model, so generation is post-processing. No neural network and no weights.

**Local language model** (Prop. 1502). Each row becomes the prompt `category: billing | priority: high` and the model learns the text after it. With `dp`, training is DP-GD at the person level: every step uses every person, clips each person's gradient to `clip`, adds Gaussian noise and divides by a noisy count of people. Full batches make the zCDP cost exact (ρ = T / 2σ²) and are where DP training works best (De et al. 2022; Sander et al. 2023). The built-in `stub` model (a tiny byte-level network, randomly initialised) is for tests and for air-gapped smoke runs; useful DP text needs a pretrained model.

**Private Evolution** (Prop. 1503; Lin et al. ICLR 2024, Xie et al. ICML 2024). A public generator proposes texts for each class; the private texts vote, through a noisy histogram, for the proposals nearest to them in an embedding that learns nothing from the data (hashed n-grams and a fixed random projection); the winners are varied and the loop repeats. The private texts never train anything.

## Personal data (PII layer)

Before a text reaches any model, recognizers find personal data and replace it by typed placeholders: `<PERSON>`, `<EMAIL>`, `<PHONE>`, `<IBAN>`, `<CARD>`, `<IP>`, `<URL>`, `<DATE>`, `<ID>`, `<ADDRESS>`, `<LOCATION>`, `<ORG>`. Generated texts get fresh surrogates of the right format from reserved ranges (example.com, 555-01xx, documentation IP addresses, Luhn-valid test cards, mod-97-valid test IBANs) (Prop. 1506).

| Recognizer | Install | Finds |
|---|---|---|
| `regex` | built in | e-mails, URLs, IP addresses, IBANs (checksum), payment cards (Luhn), phones, dates, long identifiers, street addresses, names after cues ("Regards,", "Mr", "my name is", "patient") |
| `presidio` | `relsynth[text-pii]` and a local spaCy model | Microsoft Presidio's recognizers |
| `spacy` | `relsynth[text-pii]` and a local pipeline | names, places, organisations, dates |
| `gliner` | `relsynth[text-pii]` and a local GLiNER folder | zero-shot entities, including names without cues |

Recognizers are combined by union. **Measure recall before you trust it**:

```bash
relsynth text pii-eval                                         # the generated labelled set, regex only
relsynth text pii-eval --recognizer regex --recognizer gliner=/models/gliner_multi_pii-v1
relsynth text pii-eval --labelled my-labelled.jsonl            # your own sample: {"text", "spans": [[s, e, TYPE]]}
```

On the generated set (1,000 texts, 1,597 entities) the built-in regex catches every planted e-mail, URL, IP, IBAN, card, phone, date and identifier (these formats are matched by construction, so this is an upper bound) and about 79% of names (95% interval 75–82%), with precision 1.00. Names without a cue need GLiNER or spaCy; the CI job `text-lm` measures them on the same set. The labelled set is generated, not collected: label a sample of your own texts for the recall that matters.

## Leakage audit

```bash
relsynth text audit spec.yaml --table ticket --column body [--canaries 64 --repeats 4]
```

It refits the column's configuration with added canary people (Carlini et al. 2019): each canary person holds `repeats` copies of "my reference words are w1 w2 w3", with the words drawn from common words of the corpus. It reports:

- **exposure** of each canary (log₂ of the candidates minus log₂ of the canary's rank under the model). With `dp`, the share of canaries with exposure ≥ t is tested against the bound e^ε 2^-t + δ (Prop. 1500) and the mean against (ε + 1)/ln 2 + δ log₂|R| (Prop. 1504);
- **exact and near copies** of training texts among generated ones (MinHash over word 3-grams, exact Jaccard check; Prop. 1505);
- an **extraction attack** (Carlini et al. 2021): many samples ranked by a membership score, the top ones matched against training texts held by fewer than 3 people.

## Utility metrics

```bash
relsynth text metrics real.csv synthetic.csv --column body --condition category --condition priority
```

MAUVE (Pillutla et al. 2021) on the data-independent embedding (not comparable with MAUVE on GPT-2 features), consistency with the row (a classifier trained on real texts predicts each conditioning value from a synthetic text; the ratio to held-out real texts is 1 for a perfect generator, Prop. 1507), train-on-synthetic-test-on-real accuracy, and the copy rates.

## Models and licences

relsynth never ships or downloads weights. Download a model under its own licence into a folder and give the folder in the spec. `relsynth text models` lists the register (`src/relsynth/data/text_models.yaml`) with each model's licence and status; prefer `preferred` (Apache-2.0 or MIT). `restricted` models (Llama, Gemma, Qwen2.5-3B) carry use or user-count terms and need legal review. Offline: set `HF_HUB_OFFLINE=1`; relsynth loads with `local_files_only`.

GPUs: `lm.device` (or the T03 compute preset of the run) picks the device; the default is a GPU when one is present.
