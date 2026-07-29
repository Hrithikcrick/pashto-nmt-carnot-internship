# Automatic Evaluation of Pashto-to-English NMT

No human evaluation was conducted in this experiment.

The systems were evaluated automatically using BLEU, chrF++, TER and reference-based COMET.

Automatic diagnostic indicators were used to detect number mismatches, unusually short or long outputs, repetition and empty translations.

## Main automatic results

| model         |   samples |    BLEU |   chrF++ |     TER |    COMET |   COMETKiwi_QE |
|:--------------|----------:|--------:|---------:|--------:|---------:|---------------:|
| Baseline NLLB |       100 | 19.6256 |  35.4826 | 75.6757 | 0.681784 |            nan |
| Original LoRA |       100 | 19.7687 |  36.3065 | 76.834  | 0.677456 |            nan |
| Semantic LoRA |       100 | 20.3755 |  36.6075 | 76.4479 | 0.681866 |            nan |

## Bootstrap significance

| metric   | baseline      | comparison_model   |   mean_difference_model_minus_baseline |   bootstrap_95_ci_low |   bootstrap_95_ci_high |   p_value | significant_at_0.05   |
|:---------|:--------------|:-------------------|---------------------------------------:|----------------------:|-----------------------:|----------:|:----------------------|
| BLEU     | Baseline NLLB | Original LoRA      |                               1.13015  |             -0.502166 |               2.75801  |    0.1804 | False                 |
| BLEU     | Baseline NLLB | Semantic LoRA      |                               1.88501  |              0.147212 |               3.70886  |    0.0328 | True                  |
| chrF++   | Baseline NLLB | Original LoRA      |                               0.810252 |             -0.604213 |               2.08631  |    0.2476 | False                 |
| chrF++   | Baseline NLLB | Semantic LoRA      |                               1.15115  |             -0.297291 |               2.46566  |    0.1148 | False                 |
| TER      | Baseline NLLB | Original LoRA      |                               4.07015  |            -12.9946   |               2.8993   |    0.3408 | False                 |
| TER      | Baseline NLLB | Semantic LoRA      |                               2.77279  |            -13.154    |               4.54203  |    0.6112 | False                 |
| COMET    | Baseline NLLB | Original LoRA      |                              -0.004329 |             -0.014665 |               0.006355 |    0.4196 | False                 |
| COMET    | Baseline NLLB | Semantic LoRA      |                               8.2e-05  |             -0.010247 |               0.010115 |    0.9744 | False                 |

## Pairwise sentence-level wins

| metric   | system_a      | system_b      |   system_a_wins |   system_b_wins |   ties |   mean_difference_b_minus_a |   difference_95_ci_low |   difference_95_ci_high |   p_value | significant_at_0.05   |
|:---------|:--------------|:--------------|----------------:|----------------:|-------:|----------------------------:|-----------------------:|------------------------:|----------:|:----------------------|
| BLEU     | Baseline NLLB | Original LoRA |              19 |              36 |     45 |                    1.13015  |              -0.502166 |                2.75801  |    0.1804 | False                 |
| BLEU     | Baseline NLLB | Semantic LoRA |              24 |              31 |     45 |                    1.88501  |               0.147212 |                3.70886  |    0.0328 | True                  |
| BLEU     | Original LoRA | Semantic LoRA |              18 |              13 |     69 |                    0.754865 |              -0.307774 |                2.12805  |    0.19   | False                 |
| chrF++   | Baseline NLLB | Original LoRA |              29 |              48 |     23 |                    0.810252 |              -0.604213 |                2.08631  |    0.2476 | False                 |
| chrF++   | Baseline NLLB | Semantic LoRA |              30 |              43 |     27 |                    1.15115  |              -0.297291 |                2.46566  |    0.1148 | False                 |
| chrF++   | Original LoRA | Semantic LoRA |              26 |              22 |     52 |                    0.340903 |              -0.381553 |                1.10568  |    0.3696 | False                 |
| TER      | Baseline NLLB | Original LoRA |              28 |              21 |     51 |                    4.07015  |              -2.8993   |               12.9946   |    0.3408 | False                 |
| TER      | Baseline NLLB | Semantic LoRA |              28 |              18 |     54 |                    2.77279  |              -4.54203  |               13.154    |    0.6112 | False                 |
| TER      | Original LoRA | Semantic LoRA |              17 |              11 |     72 |                   -1.29736  |              -8.33595  |                5.14015  |    0.7536 | False                 |
| COMET    | Baseline NLLB | Original LoRA |              38 |              43 |     23 |                   -0.004329 |              -0.014665 |                0.006355 |    0.4196 | False                 |
| COMET    | Baseline NLLB | Semantic LoRA |              38 |              35 |     27 |                    8.2e-05  |              -0.010247 |                0.010115 |    0.9744 | False                 |
| COMET    | Original LoRA | Semantic LoRA |              29 |              24 |     52 |                    0.004411 |              -0.001836 |                0.010735 |    0.1648 | False                 |

## Automatic diagnostic indicators

| model         |   samples |   mean_length_ratio |   number_mismatch_rate_percent |   length_outlier_rate_percent |   repetition_alert_rate_percent |   empty_output_rate_percent |   exact_reference_match_percent |
|:--------------|----------:|--------------------:|-------------------------------:|------------------------------:|--------------------------------:|----------------------------:|--------------------------------:|
| Baseline NLLB |       100 |            0.983748 |                             15 |                             4 |                               0 |                           0 |                               1 |
| Original LoRA |       100 |            1.0532   |                             15 |                             4 |                               0 |                           0 |                               2 |
| Semantic LoRA |       100 |            1.03655  |                             15 |                             5 |                               0 |                           0 |                               3 |

## Interpretation rule

For BLEU, chrF++ and COMET, a higher score is better. For TER, a lower score is better.

## Limitation

Automatic metrics estimate translation quality but do not replace evaluation by qualified human annotators. The absence of human evaluation should therefore be stated explicitly in the paper.