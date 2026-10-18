# FabRoot

Implementation of **FabRoot: Event-Grounded Multimodal Evidence Reasoning for Semiconductor Root-Cause Diagnosis**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

The finite generator renders wafer PNGs, sensor readings, alarms and process text with chronological disjoint lots. These are generated process observations, separate from the private FabRCA collection.

```bash
python finite_data.py --output data/generated --seed 11
fabroot prepare --image-encoder finite_texture --source data/generated/events.json --output data/prepared
```

`finite_texture` extracts RGB patches without pretrained weights. Authorized external event JSON uses `prepare --source`; the optional CLIP route requires the vision extra and a local `--clip-path`.

## Training and inference

```bash
fabroot train --data data/prepared/dataset.json --output outputs/full --epochs 50 --seed 11
```

## Evaluation

```bash
fabroot evaluate --data data/prepared/dataset.json --model outputs/full/model.pt --actions actions.json --output outputs/full_eval
```

## Comparisons and ablations

```bash
fabroot train --data data/prepared/dataset.json --output outputs/early --variant early_concat --seed 11
fabroot train --data data/prepared/dataset.json --output outputs/image --source-only image --seed 11
fabroot train --data data/prepared/dataset.json --output outputs/metrology --source-only metrology --seed 11
fabroot train --data data/prepared/dataset.json --output outputs/no_alignment --variant no_alignment --seed 11
fabroot train --data data/prepared/dataset.json --output outputs/no_reliability --variant no_reliability --seed 11
fabroot train --data data/prepared/dataset.json --output outputs/no_opposition --variant no_opposition --seed 11
fabroot train --data data/prepared/dataset.json --output outputs/no_evidence --variant no_evidence_loss --seed 11
fabroot train --data data/prepared/dataset.json --output outputs/no_grounding --variant no_grounding --seed 11
```

Evaluation supports `--time-shift-minutes`, `--missing-count`, `--drop` and `--no-verifier` with the same fitted model.

## Outputs

Training retains `model.pt` and `history.json`. Evaluation stores `metrics.json` and typed grounded `reports.json` under `outputs/full_eval`. Independent report audit/workflow commands require actual annotations/participant records. The complete finite full/early-concat pipeline is available via:

```bash
python reproduce.py --plan
python reproduce.py --output outputs/finite-setting
```
