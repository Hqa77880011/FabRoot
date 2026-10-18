# FabRoot

Implementation of **FabRoot: Event-Grounded Multimodal Evidence Reasoning for Semiconductor Root-Cause Diagnosis**.

FabRoot validates wafer/lot/step/tool/chamber identity before fusing asynchronous inspection images, metrology, tool alarms, and maintenance text. Observable reliability and separate support/opposition heads produce an eight-cause posterior. Retrieved evidence constrains typed reports, and training-only inspection-outcome likelihoods choose the next check by expected information gain per cost.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[vision]"
```

Use Python 3.11+ and activate the environment. Select an appropriate CPU or CUDA PyTorch build. Image preparation uses a locally stored CLIP ViT-B/16 model and image processor in Hugging Face format. Obtain the [CLIP ViT-B/16 checkpoint](https://huggingface.co/openai/clip-vit-base-patch16) separately and pass its local directory; the preparation command uses `local_files_only=True`. The [Transformers CLIP interface](https://huggingface.co/docs/transformers/v4.48.2/model_doc/clip) supplies patch and region features. No model download occurs during training or inference.

## Event data and preparation

The paper's FabRCA corpus is a private, de-identified semiconductor event collection. This repository consumes an authorized JSON export; it includes no FabRCA records, labels, reports, images, or checkpoint weights.

The source JSON is a list of events. Each event has:

| Field | Meaning |
| --- | --- |
| `id` | Unique immutable event identifier |
| `key` | Object containing `wafer`, `lot`, `step`, `tool`, `chamber` |
| `anchor` | First post-process out-of-control inspection trigger, ISO timestamp with explicit timezone |
| `split` | `train`, `val`, or `test`, with whole lots in one chronological period |
| `label` | Integer primary adjudicated cause, 0–7 in the order below |
| `records` | Native source records carrying their own complete `key`, `id`, `time`, and `type` |
| `inspections` | Training-only outcome records with integer `action` and integer `outcome` indices |
| `exposing_actions` | Independently adjudicated action indices that expose the primary cause, used only for CWAS evaluation |

Causes are recipe drift, chamber contamination, sensor calibration error, wafer alignment error, gas-delivery deviation, temperature-control deviation, wafer-handling damage, and inspection-system artifact.

Every source record has `type` (`image`, `metrology`, `alarm`, `text`), source/equipment `entity`, and optional fixed expert `ontology_tags`. These tags are input ontology relations, not post-mortem cause labels. Optional quality fields are `qc`, `duplicate`, `entity_match`, and `valid_fraction`. Training evidence records additionally have `support` and `oppose` lists of cause indices; an empty list explicitly means no annotated link. Evaluation evidence mAP uses the same independent annotations without supplying them to the model.

Source-specific fields are:

- Images: relative `path`, numeric `severity`, `coordinate` as a two-element normalized wafer coordinate, and `after_step: true`. The highest-severity four regions are retained. Their CLIP patch/region features are cached locally.
- Metrology: exactly 64 `values` and boolean `valid` entries, optionally 64 `field_names`; `qc` must pass. The closest valid record within ±15 minutes is selected. Invalid fields never become evidence propositions.
- Alarms: `code`, optional `text`, and a stable `retransmission_key` for deduplication. Only alarms in ±10 minutes are selected and ordered by time.
- Maintenance text: de-identified `text` entered from -60 to +120 minutes. Email/phone patterns and the structured author field are removed during preparation; person-name de-identification belongs in the export process.

Image/alarm/text records may carry structured `facts`, each with normalized `subject`, `relation`, and `value`. These describe the actual observation available for report verification. Metrology facts are generated only from valid measured fields. Missing sources have no invented normal token.

```bash
fabroot prepare --source fabrca_events.json --clip-path /path/to/local/clip-vit-base-patch16 --output data/fabrca
```

Preparation rejects mismatched event keys before time-window selection, enforces unique retained evidence IDs, collapses alarm retransmissions, and checks lot exclusivity and chronological train/validation/test ordering. `data/fabrca/dataset.json` contains selected records, training-only text/alarm vocabulary, equipment vocabulary, and fieldwise metrology statistics. `--vocabulary-size` caps the encountered training-word vocabulary at 20000 entries; other words map to the unknown token. `excluded.json` records rejected record IDs and reasons. Image arrays remain under the ignored data directory. A no-image export does not require `--clip-path` or the vision extra.

## Train and compare

```bash
fabroot train --data data/fabrca/dataset.json --output outputs/full --epochs 50 --seed 11
fabroot train --data data/fabrca/dataset.json --output outputs/early --variant early_concat
fabroot train --data data/fabrca/dataset.json --output outputs/image --source-only image
fabroot train --data data/fabrca/dataset.json --output outputs/metrology --source-only metrology
fabroot train --data data/fabrca/dataset.json --output outputs/no_alignment --variant no_alignment
fabroot train --data data/fabrca/dataset.json --output outputs/no_reliability --variant no_reliability
fabroot train --data data/fabrca/dataset.json --output outputs/no_opposition --variant no_opposition
fabroot train --data data/fabrca/dataset.json --output outputs/no_evidence --variant no_evidence_loss
fabroot train --data data/fabrca/dataset.json --output outputs/no_grounding --variant no_grounding
```

Use `--device cuda` when available. The default shared width is 128; text and alarm encoders each have four Transformer layers, while two typed graph-attention layers connect evidence and eight trainable cause queries. Source-specific sinusoidal relative-time projections, source type, equipment context, and four-bit availability embeddings align tokens. Reliability has floor 0.05 and uses observed quality and absolute delay. Signed cause scores use opposition coefficient 0.6.

The loss combines cause cross-entropy and a three-candidate listwise loss, 0.2 same-event contrastive alignment against other lots/chambers, 0.5 annotated support/opposition loss, and 0.1 structured-claim validity loss. Claim training uses observed facts and explicitly contradicted-value negatives. AdamW uses 2e-4 and weight decay 0.01; gradient norm is clipped at one. Defaults are batch eight, maximum 50 epochs, and patience seven on validation NDCG@3. Batch size is reduced from the manuscript's 32 for direct use with cached patch tokens; it is configurable. Report factuality is audited separately rather than used as an automatic model-selection tie-breaker.

The image backbone is frozen during feature caching; its learned projection trains with the other encoders. This differs from a jointly fine-tuned visual backbone. `early_concat` mean-pools and concatenates the same four source representations. Single-source models are trained using only their selected source. No result is labeled MulT, CLIP-FiLM, or Evidence-RAG without those distinct systems being implemented.

## Diagnose, inspect, and stress

Edit `actions.json` before evaluation. It contains the paper's eight inspection names, positive normalized costs, and discrete outcome names. The supplied equal costs illustrate the schema; use the actual registered technician/equipment cost table. Training inspection outcomes estimate `P(outcome | cause, action)` with unit Dirichlet pseudocounts. Missing outcome evidence remains uninformative rather than receiving a fabricated effect size.

```bash
fabroot evaluate --data data/fabrca/dataset.json --model outputs/full/model.pt --actions actions.json --output outputs/full_eval
fabroot evaluate --data data/fabrca/dataset.json --model outputs/full/model.pt --actions actions.json --output outputs/shift30 --time-shift-minutes 30
fabroot evaluate --data data/fabrca/dataset.json --model outputs/full/model.pt --actions actions.json --output outputs/missing2 --missing-count 2
fabroot evaluate --data data/fabrca/dataset.json --model outputs/full/model.pt --actions actions.json --output outputs/no_text --drop text
```

Positive temperature is fitted only on validation cause labels. Every event receives a pre-release top-three ranking. Report release requires at least two present sources, top-one probability at least 0.55, and signed support/opposition margin at least 0.08. Abstention requests manual escalation and contributes zero CWAS. No control command is issued.

Time-shift stress adds 15/30/60 minutes to non-image relative times after fixed record selection; records are not reselected. `--missing-count 1|2|3` averages all combinations of that many removed sources, retaining separate per-condition event records. For a study starting from fully observed events, supply the registered fully observed test subset. A foreign-lot sentence inserted into the source export is rejected by default. `prepare --allow-foreign` is an explicit identity-check ablation; `no_alignment` instead removes time/equipment conditioning after preparation, so these two interventions test different mechanisms.

`reports.json` contains rankings, posterior, signed margin, retrieved support/counter-evidence IDs, typed propositions, next inspection, information-per-cost, abstention, and latency. Retrieval takes up to twelve positive and four negative evidence records, first preserving one positively scoring record from each available source where possible. Ranking and evidence metrics are computed before abstention on all test events.

## Grounded report and independent audit

The default report path uses a finite typed decoder over retrieved structured facts. This implementation choice makes the identifier vocabulary and proposition verification explicit rather than requiring a six-layer free-text language decoder. It emits candidate cause, support, counter-evidence, uncertainty, and next-action objects. The verifier checks exact normalized subject/relation/value membership against the cited observation and learned claim confidence. A failed statement becomes `Evidence is insufficient for this statement.`

An external typed decoder can provide JSON proposals through `evaluate --proposals proposals.json`. This file maps event IDs to lists with `type` (`support` or `counter_evidence`), `evidence` (one retrieved evidence ID), and `fact`. Unretrieved IDs remain blocked. `--no-verifier` admits retrieved-ID proposals without factual/claim-confidence checks, providing a direct verifier-removal comparison when the same external proposals are used.

The model's verifier is not used to certify GP or USR. Have independent engineers label released diagnostic proposition IDs, blinded to model identity, then run:

```bash
fabroot audit --reports outputs/full_eval/reports.json --annotations independent_audit.json --output outputs/full_eval/audit.json
```

The annotation file is a JSON list with `id` and `status`: `supported`, `unsupported`, or `unadjudicated`. Unknown or duplicate IDs are rejected. Missing judgments count as unadjudicated. Candidate-cause, support, and counter-evidence diagnostic objects share the denominator; insufficiency/abstention messages and operational uncertainty/action objects are excluded. Audit output reports GP, USR, and unadjudicated rate, whose sum is one. Without an independent audit these metrics remain `null`.

`metrics.json` also reports all-event NDCG@3, eight-class Macro-F1, evidence mAP against annotated signed links, release coverage, released-only NDCG@3, all-event release-weighted NDCG@3, and CWAS when every event has an adjudicated `exposing_actions` list. Evidence AP is averaged over available nonempty support/opposition reference lists; missing annotations are not scored as correct links. `confidence.png` shows posterior confidence. No experiment values are prefilled.

## Technician workflow records

The human workflow endpoint requires observed participant trials. A CSV contains `participant`, `interface`, `diagnostic_minutes`, `normalized_cost`, and binary `correct_action`. Analyze actual counterbalanced trials with:

```bash
fabroot workflow --source technician_trials.csv --reference baseline --output outputs/workflow.json --seed 11
```

This reports interface means and participant-paired differences, 10,000 participant bootstrap intervals, Wilcoxon p-values, and Holm-adjusted p-values across the compared endpoints. It does not synthesize technician performance from classifier accuracy. Repeat the model runs with seeds 11/23/37/51/71 and use lot-level paired event resampling for model comparisons.

## Fixed reproduction setting

Precomputed 768-dimensional image features plus aligned text/metrology/wafer facts; lot/time-disjoint CPU training50 epochs seed11 and train-only action likelihoods.

The checked-in entry point fixes data seeds, methods, budgets and output paths. Inspect its commands without starting any computation:

```bash
python reproduce.py --plan --source data/source
```

Remove `--plan` to execute the registered pipeline in a fresh output directory. The entry point calls the existing implementation; it does not insert paper results. Installation uses the package instructions above.
