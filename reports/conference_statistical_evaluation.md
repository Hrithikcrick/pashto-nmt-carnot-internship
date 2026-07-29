# Statistical Evaluation of Pashto-to-English NMT

Input predictions: `outputs\tables\conference_predictions_combined.csv`

The same reference sentences were used for all systems.

## Corpus-level results

| Model         |   Samples |    BLEU |   BLEU 95% CI Low |   BLEU 95% CI High |   chrF++ |   chrF 95% CI Low |   chrF 95% CI High |
|:--------------|----------:|--------:|------------------:|-------------------:|---------:|------------------:|-------------------:|
| Baseline NLLB |       100 | 19.6256 |           14.3092 |            24.8558 |  35.4826 |           30.0558 |            41.0169 |
| Original LoRA |       100 | 19.7687 |           14.6335 |            24.9171 |  36.3065 |           30.8922 |            41.8657 |
| Semantic LoRA |       100 | 20.3755 |           15.0521 |            25.6919 |  36.6075 |           30.9847 |            42.2642 |

Confidence intervals were calculated using paired bootstrap resampling.

## Paired significance tests

| System A      | System B      |   BLEU Difference B-A |   BLEU p-value |   chrF Difference B-A |   chrF p-value | Significant at 0.05   |
|:--------------|:--------------|----------------------:|---------------:|----------------------:|---------------:|:----------------------|
| Baseline NLLB | Original LoRA |                0.1432 |          0.923 |                0.8238 |          0.166 | False                 |
| Baseline NLLB | Semantic LoRA |                0.7499 |          0.401 |                1.1249 |          0.066 | False                 |

## Interpretation

- A p-value below 0.05 is commonly treated as evidence that the difference is statistically significant.
- Statistical significance does not automatically mean that the improvement is practically large.
- Automatic metrics should be accompanied by blinded human evaluation of adequacy, fluency, missing content, named entities, and hallucination.