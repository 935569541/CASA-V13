# CASA V13

Behaviour-aware Content-Adaptive Sparse Attention for short-video click-through-rate prediction.

## Overview

CASA V13 is a PyTorch research prototype for selecting useful events from a long user history before target attention. For each candidate video, a lightweight gate scores all valid history events using:

- item identity and content features;
- candidate-history interactions;
- a candidate-aware recency signal;
- action strength, watch completion and temporal freshness.

During training, a soft Gumbel relaxation allows gradients to reach the selector. During inference, deterministic hard Top-k selection gathers only the selected history events before fine attention.

## Main result

The locked five-seed KuaiRand-Pure holdout experiment used histories of up to 500 events and selected 100 events for fine attention.

| Model | Mean AUC | Mean LogLoss | CPU latency (ms/example) | Fine-attention events |
|---|---:|---:|---:|---:|
| CASA V13 | 0.675300 | 0.645593 | 0.39933 | 100 |
| Recent Top-100 | 0.674719 | 0.645639 | 0.35070 | 100 |
| Full attention | 0.674563 | 0.645754 | 0.35990 | up to 500 |
| Random Top-100 | 0.674488 | 0.645260 | 0.35369 | 100 |

CASA V13 had the highest observed mean AUC and reduced the number of events entering fine attention by 80%. The paired confidence interval against full attention included zero, so statistical superiority was not established. The current CPU implementation was also about 11% slower than full attention because gate scoring, Top-k and gathering added overhead.

## Architecture

1. Encode the candidate and up to 500 historical events.
2. Add content and behaviour features to each history representation.
3. Score every history event with a candidate-aware gate.
4. Select the highest-scoring Top-k events.
5. Apply exact target attention to the selected events.
6. Combine candidate and user-interest vectors to predict a click.

The full-attention model can be used as a teacher. The student objective supports label loss, temperature-scaled logit distillation and cosine representation alignment.

## Project files

- `casa/model.py`: CASA model, selector and attention paths.
- `casa/data.py`: CSV and streaming dataset loaders.
- `preprocess_kuairand.py`: temporal KuaiRand preprocessing.
- `train.py`: training, evaluation and distillation.
- `evaluate_checkpoint.py`: checkpoint evaluation.
- `run_baselines.py`: baseline experiments.
- `run_v13_validation_suite.py`: fixed V13 validation suite.
- `run_v13_fair_final_suite.py`: locked matched-seed evaluation.
- `tests/`: automated data, preprocessing and model tests.

## Installation

Python 3.10 to 3.12 is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the virtual environment before running the commands below.

## Quick smoke test

This command uses generated data and does not require KuaiRand:

```bash
python train.py --epochs 1 --num-samples 512 --sequence-length 40 --top-k 8 --output runs/smoke
```

Run the automated tests:

```bash
python -m unittest discover -s tests -v
```

## Training with processed data

```bash
python train.py \
  --train-csv data/processed/train.csv \
  --validation-csv data/processed/validation.csv \
  --num-items 7583 \
  --sequence-length 500 \
  --top-k 100 \
  --embedding-dim 32 \
  --epochs 8 \
  --batch-size 128 \
  --learning-rate 0.0005 \
  --selection-mode learned \
  --use-content-features \
  --use-behavior-features \
  --num-tags 43 \
  --num-author-buckets 2048 \
  --num-duration-buckets 6 \
  --content-embedding-dim 8 \
  --streaming-data \
  --early-stopping-patience 2 \
  --output runs/v13
```

Teacher distillation can be enabled with `--teacher-checkpoint`, `--initialize-from-teacher`, `--distillation-weight 0.5`, `--representation-weight 0.1` and `--distillation-temperature 2.0`.

## Data and scope

The raw KuaiRand-Pure dataset is not included in this repository. Obtain it from its official source and follow its licence and usage terms.

This repository demonstrates an offline research prototype. It does not represent TikTok data, a complete industrial recommendation system or a production latency claim.

## Reproducibility

The final study used matched seeds `7`, `42`, `123`, `2026` and `3407`. Eleven automated tests cover core data shapes, feature requirements, preprocessing and selector behaviour.

