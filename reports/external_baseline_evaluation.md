# External Multilingual Baseline Evaluation

M2M100 and mBART-50 were evaluated zero-shot on the same Pashto-English instances used for NLLB evaluation.

| model       | model_checkpoint                         | evaluation_type   |   samples |    BLEU |   chrF++ |     TER |    COMET |   empty_outputs | device   |   num_beams |   elapsed_seconds |   seconds_per_sentence |
|:------------|:-----------------------------------------|:------------------|----------:|--------:|---------:|--------:|---------:|----------------:|:---------|------------:|------------------:|-----------------------:|
| M2M100 418M | facebook/m2m100_418M                     | zero-shot         |       100 | 14.6243 |  30.5912 | 85.4247 | 0.616897 |               0 | cpu      |           4 |           338.1   |                3.381   |
| mBART-50    | facebook/mbart-large-50-many-to-many-mmt | zero-shot         |       100 | 22.6482 |  38.0761 | 75.7722 | 0.693222 |               0 | cpu      |           4 |           500.018 |                5.00018 |

## Interpretation

These models are external zero-shot baselines. They were not fine-tuned on the Pashto-English training corpus.

The comparison contextualizes the NLLB and LoRA results across independent multilingual translation architectures.