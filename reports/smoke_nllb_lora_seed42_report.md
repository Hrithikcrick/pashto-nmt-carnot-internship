# NLLB LoRA Smoke-Test Report

## Purpose

This experiment verifies that the saved LoRA adapter can be reloaded and used for Pashto-to-English generation.

It is not a final translation-quality experiment.

## Training check

- Training rows: 100
- Validation rows: 20
- Training duration: approximately 34 seconds
- Epoch fraction: 0.1
- Device: CPU

## Inference check

- Test samples: 20
- BLEU: 14.566518
- chrF++: 27.944296
- TER: 80.660377
- Empty outputs: 0
- Inference duration: 137.0533 seconds

## Interpretation

These scores must not be used as final research results because the adapter was trained on only 100 rows for 0.1 epoch.

A successful run confirms the training, adapter-saving, adapter-loading, generation, and metric-computation pipeline.