# Week 4 Fine-Tuning Plan

## Objective

Week 4 focuses on LoRA fine-tuning of `facebook/nllb-200-distilled-600M` for Pashto-to-English translation using the high-quality 10k Pashto-English subset.

## Current Status

Completed before Week 4:

1. Baseline NLLB implementation.
2. Cleaned Pashto-English dataset.
3. High-quality 10k subset.
4. Gold test candidates.
5. Clean-test baseline score files.
6. Manual error analysis files.

## Why LoRA

LoRA is selected because it needs less GPU memory, trains small adapter weights, and is practical for Google Colab GPU.

## Dataset

Training file:

`data/train_high_quality_10k.csv`

Evaluation file:

`data/gold_test_candidates_100.csv`

## Planned Experiment

| Experiment | Model | Dataset | Status |
|---|---|---|---|
| E0 | NLLB 600M baseline | No fine-tuning | Completed earlier |
| E1 | NLLB 600M + LoRA | High-quality 10k pairs | Week 4 target |

## Evaluation Metrics

The model will be evaluated using BLEU, chrF, prediction comparison, and manual analysis.

## Important Note

Fine-tuning results should be added only after actual GPU training is completed.
