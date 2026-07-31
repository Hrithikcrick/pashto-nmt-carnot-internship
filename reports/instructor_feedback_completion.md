# Instructor Feedback Completion Report

## Requested improvements

### 1. Improve figures and diagrams

Completed:

- Informative end-to-end research pipeline diagram
- BLEU and chrF++ model comparison figure
- TER model comparison figure
- COMET model comparison figure where scores are available

### 2. Add qualitative translation examples

Completed:

- Baseline NLLB outputs
- Original LoRA outputs
- Semantic-filtered LoRA outputs
- Automatically selected improvement candidates
- Automatically selected regression candidates

These examples still require careful bilingual interpretation.

### 3. Compare with additional models

Completed:

- M2M100 418M zero-shot comparison
- mBART-50 zero-shot comparison

IndicTrans2 is documented as applicable only to the English-to-Hindi pivot stage.

## Additional research improvements completed

- BLEU, chrF++, TER, and COMET evaluation
- Paired bootstrap significance analysis
- Automatic translation diagnostics
- Fixed source-grouped leak-free splits
- Dataset checksums and environment records
- Reproducible experiment manifests
- Multi-seed experiment configurations
- Successful LoRA smoke training and inference checks

## Generated files

- Combined metrics: `outputs\tables\instructor_all_model_metrics.csv`
- Qualitative examples: `outputs\tables\instructor_qualitative_examples.csv`
- `paper\figures\instructor_extended_pipeline.png`
- `paper\figures\instructor_model_quality_comparison.png`
- `paper\figures\instructor_model_ter_comparison.png`
- `paper\figures\instructor_model_comet_comparison.png`
- `paper\generated\instructor_model_comparison_table.tex`
- `paper\generated\instructor_qualitative_examples.tex`