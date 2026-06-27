
# Week 5 Research Work: Semantic Filtering and Manual Evaluation

Week 5 extends the Pashto Neural Machine Translation project beyond initial LoRA fine-tuning. The main focus is to improve dataset quality using semantic similarity filtering and prepare deeper evaluation through manual comparison of baseline and fine-tuned outputs.

### Week 5 Objective

The goal is to identify stronger Pashto-English sentence pairs for future fine-tuning and prepare a more reliable evaluation process. Instead of depending only on BLEU and chrF, this stage adds semantic filtering, graph-based analysis, and manual evaluation planning.

### Work Completed

- Added semantic similarity filtering using multilingual sentence embeddings.
- Generated semantic similarity scores for Pashto-English sentence pairs.
- Created semantically filtered training data.
- Generated semantic similarity distribution graph.
- Generated filtering overview graph.
- Generated similarity score statistics graph.
- Created manual evaluation template for baseline vs fine-tuned outputs.
- Added work-status graph for the next research stage.
- Prepared analysis plan for direct Pashto-to-Hindi vs pivot Pashto-English-Hindi translation.

### Week 5 Semantic Filtering Results

| Metric | Value |
|---|---:|
| Total sentence pairs scored | 10,000 |
| Filtered sentence pairs retained | 8,000 |
| Mean semantic similarity | 0.4001 |
| Median semantic similarity | 0.3967 |
| Minimum semantic similarity | -0.2038 |
| Maximum semantic similarity | 1.0000 |

### Week 5 Graph Analysis

#### Semantic Similarity Distribution

![Week 5 Semantic Similarity Distribution](outputs/figures/week5_semantic_similarity_distribution.png)

This graph shows how strongly Pashto-English sentence pairs are semantically aligned. Higher similarity values indicate better aligned sentence pairs, while low values indicate potentially noisy or weakly matched pairs.

#### Semantic Filtering Overview

![Week 5 Filtering Overview](outputs/figures/week5_filtering_overview.png)

This graph compares the number of sentence pairs scored and the number of sentence pairs retained after filtering. It helps show how the dataset was reduced from a larger pool into a cleaner subset.

#### Semantic Similarity Score Statistics

![Week 5 Similarity Score Statistics](outputs/figures/week5_similarity_score_stats.png)

This graph summarizes the mean, median, minimum, and maximum semantic similarity scores. It provides a quick view of the quality range of the sentence-pair alignment.

#### Week 5 Work Status

![Week 5 Work Status](outputs/figures/week5_work_status.png)

This graph tracks completed and pending research tasks. Semantic filtering and evaluation-template preparation are completed, while human scoring, direct-vs-pivot Hindi evaluation, and IndicTrans2 integration remain future work.

### Week 5 Important Files

- `src/week5_semantic_filter.py`
- `src/week5_create_manual_eval_template.py`
- `src/week5_generate_report.py`
- `src/week5_work_status.py`
- `docs/week5_research_work.md`
- `outputs/tables/week5_semantic_filtering_summary.csv`
- `outputs/tables/week5_semantic_similarity_scores.csv`
- `outputs/tables/week5_top_bottom_similarity_examples.csv`
- `outputs/tables/week5_manual_evaluation_template.csv`
- `outputs/tables/week5_work_status.csv`
- `outputs/figures/week5_semantic_similarity_distribution.png`
- `outputs/figures/week5_filtering_overview.png`
- `outputs/figures/week5_similarity_score_stats.png`
- `outputs/figures/week5_work_status.png`

### Week 5 Interpretation

Semantic filtering improves the research pipeline by ranking sentence pairs according to cross-lingual similarity. This helps identify better training pairs for future LoRA fine-tuning. The filtered dataset can be used to test whether a smaller but cleaner training set performs better than a larger but noisier dataset.

Manual evaluation is also important because automatic metrics such as BLEU and chrF do not fully capture meaning preservation, missing words, named entity correctness, hallucination, or fluency. The manual evaluation template will help compare baseline and fine-tuned outputs in a more research-oriented way.

### Remaining Work

- Manually score selected baseline and fine-tuned translations.
- Train LoRA again using semantically filtered data.
- Compare baseline, original LoRA, and semantic-filtered LoRA checkpoints.
- Compare direct Pashto-to-Hindi translation with pivot Pashto-English-Hindi translation.
- Explore IndicTrans2 for the English-to-Hindi stage.
