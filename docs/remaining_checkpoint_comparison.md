# Remaining Work: Checkpoint Comparison

This report compares the baseline NLLB model, original LoRA checkpoints, and the semantic-filtered LoRA checkpoint.

## Comparison Table

| Model | Samples | BLEU | chrF |
|---|---:|---:|---:|
| Baseline NLLB | 100 | 19.63 | 37.91 |
| LoRA 20 | 100 | 19.63 | 37.91 |
| LoRA 100 | 100 | 19.65 | 37.97 |
| LoRA 500 | 100 | 20.09 | 38.27 |
| LoRA 1000 | 100 | 19.79 | 38.33 |
| LoRA 3000 | 100 | 19.7 | 38.49 |
| Original LoRA 10k | 100 | 19.77 | 38.76 |
| Semantic-filtered LoRA 8000 | 100 | 20.38 | 39.02 |

## Graphs

![chrF Checkpoint Comparison](../outputs/figures/remaining_chrf_checkpoint_comparison.png)

![BLEU Checkpoint Comparison](../outputs/figures/remaining_bleu_checkpoint_comparison.png)

## Interpretation

The semantic-filtered LoRA checkpoint is compared against the baseline and previous LoRA checkpoints to check whether cleaner sentence-pair selection improves translation quality. chrF is especially important here because it captures character-level similarity, which is useful for low-resource translation evaluation.
