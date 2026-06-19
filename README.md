# Pashto to English and Hindi Neural Machine Translation

This repository contains my Carnot internship research work on low-resource Pashto Neural Machine Translation.

## Project Overview

This project focuses on building and improving a Neural Machine Translation system for:

1. Pashto to English translation
2. Direct Pashto to Hindi translation
3. Pashto to English to Hindi pivot-based translation

The baseline system uses the pretrained facebook/nllb-200-distilled-600M model.

## Current Work Completed

- Implemented baseline Pashto to English translation.
- Implemented direct Pashto to Hindi translation.
- Implemented Pashto to English to Hindi pivot translation.
- Used WMT20 Pashto-English data.
- Built dataset cleaning pipeline.
- Removed duplicate, noisy, URL, HTML, short, long, and mismatched sentence pairs.
- Created train-validation-test split.
- Evaluated baseline NLLB model using BLEU, chrF, and inference time.
- Started initial error analysis.

## Dataset Summary

| Stage | Rows |
|---|---:|
| Raw rows | 93,498 |
| After duplicate removal | 93,418 |
| Final clean rows | 90,978 |
| Removed noisy rows | 2,440 |

## Baseline Result

| Model | Direction | Samples | BLEU | chrF |
|---|---|---:|---:|---:|
| NLLB 600M | Pashto to English | 100 | 17.97 | 36.49 |

## Average Inference Time on CPU

| Direction | Time |
|---|---:|
| Pashto to English | 4.46 sec |
| Pashto to Hindi Direct | 5.11 sec |
| Pashto to English to Hindi Pivot | 4.76 sec |

## Future Work

- Create manually verified Pashto-English-Hindi test set.
- Add semantic similarity filtering using LaBSE, LASER, or multilingual sentence embeddings.
- Fine-tune NLLB on 10k and 50k cleaned Pashto-English pairs.
- Compare baseline and fine-tuned models.
- Add IndicTrans2 for better English-to-Hindi translation.
- Perform manual evaluation and error analysis.
- Build final Streamlit demo.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- SentencePiece
- Pandas
- Scikit-learn
- SacreBLEU
- Streamlit
- PEFT / LoRA

## Author

Hrithik Sharma


## Week 2 Analysis Dashboard

The Week 2 research analysis includes dataset cleaning summary, train-validation-test split, baseline BLEU/chrF score, inference time analysis, and research progress dashboard.

### Analysis Figures

![Dataset Cleaning Summary](outputs/figures/dataset_cleaning_summary.png)

![Train Validation Test Split](outputs/figures/train_val_test_split.png)

![Baseline Metrics](outputs/figures/baseline_metrics.png)

![Inference Time](outputs/figures/inference_time.png)

![Research Progress Dashboard](outputs/figures/research_progress_dashboard.png)

### Research Tables

- `outputs/tables/dataset_cleaning_summary.csv`
- `outputs/tables/train_val_test_split.csv`
- `outputs/tables/baseline_metrics.csv`
- `outputs/tables/inference_time.csv`
- `outputs/tables/model_comparison_plan.csv`
- `outputs/tables/manual_evaluation_template.csv`


## Week 3 Dataset Quality Integration

Week 3 work integrates cleaned dataset files, high-quality training subset, gold test candidates, clean-test baseline scores, and manual error analysis into the research repository.

### Week 3 Graphs

![Week 3 Dataset Files Summary](outputs/figures/week3_dataset_files_summary.png)

![Week 3 Status Summary](outputs/figures/week3_status_summary.png)

### Week 3 Report

- `docs/week3_dataset_quality_report.md`

### Week 3 Tables

- `outputs/tables/week3_dataset_files_summary.csv`
- `outputs/tables/week3_status_summary.csv`
- `outputs/tables/week3_high_quality_subset_summary.csv`
- `outputs/tables/week3_clean_test_baseline_scores.csv`
- `outputs/tables/week3_manual_error_analysis_clean_test.csv`


## Week 3 Final Dataset Quality and Evaluation

Week 3 completed dataset-quality preparation before fine-tuning. The work includes cleaned dataset integration, high-quality 10k subset preparation, gold test candidates, clean-test baseline score integration, and manual error analysis.

### Week 3 Final Graphs

![Week 3 Final Dataset Summary](outputs/figures/week3_final_dataset_summary.png)

![Week 3 Final Status](outputs/figures/week3_final_status.png)

### Week 3 Final Report

- `docs/week3_final_report.md`

### Week 3 Final Tables

- `outputs/tables/week3_final_dataset_summary.csv`
- `outputs/tables/week3_final_status.csv`
- `outputs/tables/week3_final_clean_test_scores.csv`
- `outputs/tables/week3_final_error_type_counts.csv`

## Week 4 LoRA Fine-Tuning and Evaluation

Week 4 focuses on LoRA-based fine-tuning of the `facebook/nllb-200-distilled-600M` model for Pashto-to-English Neural Machine Translation.

The fine-tuning pipeline uses Hugging Face Transformers, Datasets, and PEFT/LoRA. Initial local training trials were completed successfully, and the models were evaluated using BLEU and chrF scores. Early evaluations showed consistent chrF values around 38-39.

### Week 4 Work Completed

- Prepared LoRA fine-tuning plan.
- Created Hugging Face NLLB fine-tuning script.
- Created evaluation script for BLEU and chrF.
- Created automatic Week 4 report generation script.
- Fixed local training compatibility issues with the updated Transformers library.
- Completed initial LoRA training trials on local machine.
- Generated baseline vs fine-tuned comparison outputs.

### Week 4 Important Files

- `docs/week4_finetuning_plan.md`
- `docs/week4_colab_steps.md`
- `docs/week4_finetuning_report.md`
- `src/finetune_nllb_lora.py`
- `src/evaluate_week4_model.py`
- `src/generate_week4_report.py`

### Week 4 Results

- `outputs/tables/week4_model_comparison.csv`
- `outputs/figures/week4_bleu_comparison.png`
- `outputs/figures/week4_chrf_comparison.png`

### Week 4 Note

The full model checkpoints are not pushed to GitHub because they are large. Only scripts, reports, graphs, and evaluation outputs are included in the repository.


## Week 5 Future Work

In Week 5, the project will move toward stronger evaluation and improvement of the Pashto Neural Machine Translation system.

### Planned Week 5 Tasks

- Continue larger LoRA fine-tuning experiments using more high-quality Pashto-English sentence pairs.
- Compare different fine-tuned checkpoints using BLEU and chrF.
- Add semantic similarity filtering using LaBSE, LASER, or multilingual sentence-transformers.
- Improve dataset quality by filtering noisy or weak sentence pairs.
- Perform manual evaluation of translation outputs.
- Analyze errors such as missing words, named entity mistakes, tense/gender errors, hallucinations, and poor fluency.
- Compare direct Pashto-to-Hindi translation with Pashto-to-English-to-Hindi pivot translation.
- Explore IndicTrans2 for better English-to-Hindi translation.
- Prepare final comparison report with baseline vs fine-tuned results.
- Move toward final demo preparation.

### Week 5 Note

These are planned future tasks and should not be marked as completed until the experiments are actually run.

