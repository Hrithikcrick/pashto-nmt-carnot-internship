# Week 2 Research Analysis

## Summary

This week, the Pashto Neural Machine Translation project was moved from a simple baseline demo to a research-level dataset and evaluation foundation.

The current system uses the pretrained `facebook/nllb-200-distilled-600M` model for:

1. Pashto to English translation.
2. Direct Pashto to Hindi translation.
3. Pashto to English to Hindi pivot-based translation.

## Dataset Cleaning Result

The WMT20 Pashto-English dataset was cleaned using duplicate removal, URL removal, HTML cleaning, sentence length filtering, script validation, and length ratio filtering.

| Stage | Rows |
|---|---:|
| Raw rows | 93,498 |
| After duplicate removal | 93,418 |
| Final clean rows | 90,978 |
| Removed noisy rows | 2,440 |

![Dataset Cleaning Summary](../outputs/figures/dataset_cleaning_summary.png)

## Train-Validation-Test Split

| Split | Rows |
|---|---:|
| Training | 72,782 |
| Validation | 9,098 |
| Test | 9,098 |

![Train Validation Test Split](../outputs/figures/train_val_test_split.png)

## Baseline Evaluation

The baseline NLLB 600M model was evaluated on 100 Pashto-English test samples.

| Model | Direction | Samples | BLEU | chrF |
|---|---|---:|---:|---:|
| NLLB 600M | Pashto-English | 100 | 17.97 | 36.49 |

![Baseline Metrics](../outputs/figures/baseline_metrics.png)

## Inference Time Analysis

| Translation Direction | Average Time |
|---|---:|
| Pashto-English | 4.46 sec |
| Pashto-Hindi Direct | 5.11 sec |
| Pashto-English-Hindi Pivot | 4.76 sec |

![Inference Time](../outputs/figures/inference_time.png)

## Research Progress

![Research Progress Dashboard](../outputs/figures/research_progress_dashboard.png)

## Key Observations

1. The pretrained NLLB baseline can generate meaningful Pashto-English translations, but the quality is still moderate.
2. The BLEU score of 17.97 and chrF score of 36.49 show that fine-tuning is required.
3. The WMT20 dataset contains noisy and mismatched sentence pairs, so semantic filtering is important before fine-tuning.
4. Direct Pashto-to-Hindi translation is weaker in some cases compared to pivot-based translation.
5. CPU inference is slow, so GPU or optimized inference will be useful for future experiments.

## Next Research Steps

1. Create manually verified Pashto-English-Hindi gold test set.
2. Apply semantic similarity filtering using LaBSE or multilingual sentence embeddings.
3. Fine-tune NLLB on 10k clean pairs.
4. Fine-tune NLLB on 10k semantic-filtered pairs.
5. Fine-tune on 50k filtered pairs if resources allow.
6. Compare baseline and fine-tuned models using BLEU, chrF, inference time, and human scoring.
7. Add IndicTrans2 for English-to-Hindi translation.
8. Compare direct Hindi, pivot Hindi, and fine-tuned NLLB + IndicTrans2 pipeline.
