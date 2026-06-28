# Remaining Research Work

This file tracks the remaining research work for the Pashto Neural Machine Translation project.

## Remaining Tasks

1. Manually score selected baseline and fine-tuned translations.
2. Train LoRA again using semantically filtered data.
3. Compare baseline, original LoRA, and semantic-filtered LoRA checkpoints.
4. Compare direct Pashto-to-Hindi translation with pivot Pashto-English-Hindi translation.
5. Explore IndicTrans2 for the English-to-Hindi stage.

## Purpose

The purpose of these remaining tasks is to move the project from initial fine-tuning results toward a stronger research-quality evaluation. Automatic metrics such as BLEU and chrF are useful, but manual scoring and error analysis are necessary to understand whether the translations are actually better in meaning, fluency, completeness, and named entity correctness.

## Expected Outputs

- Manual evaluation CSV.
- Semantic-filtered LoRA checkpoint evaluation.
- Baseline vs original LoRA vs semantic-filtered LoRA comparison table.
- Direct Hindi vs pivot Hindi output file.
- IndicTrans2 exploration notes.
- Updated graphs and README analysis.
