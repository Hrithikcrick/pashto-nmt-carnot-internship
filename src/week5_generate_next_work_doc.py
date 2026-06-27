import os
from pathlib import Path
import pandas as pd

os.makedirs("docs", exist_ok=True)

report = """# Next Research Work: Semantic Filtering, Manual Evaluation, and Hindi Pivot Improvement

## Objective

The next stage of this Pashto Neural Machine Translation project focuses on improving translation quality beyond the initial LoRA fine-tuning trials. The main goal is to strengthen the research by adding semantic similarity filtering, manual evaluation, deeper error analysis, and direct-vs-pivot Hindi translation comparison.

## 1. Larger LoRA Fine-Tuning Experiments

The current LoRA trials show stable chrF improvement over the baseline. The next step is to continue larger fine-tuning experiments using progressively larger subsets:

- 1k high-quality sentence pairs
- 3k high-quality sentence pairs
- 10k high-quality sentence pairs
- semantically filtered high-quality sentence pairs

Each checkpoint will be evaluated using BLEU and chrF on the same gold test candidate set.

## 2. Semantic Similarity Filtering

Even after rule-based cleaning, some Pashto-English sentence pairs may still be weakly aligned. Semantic filtering will be used to score each pair using multilingual sentence embeddings.

The planned semantic filtering pipeline is:

1. Encode Pashto sentences using a multilingual sentence embedding model.
2. Encode English sentences using the same embedding model.
3. Compute cosine similarity between source and target embeddings.
4. Sort sentence pairs by similarity.
5. Keep the top high-quality pairs for additional fine-tuning.

The first implementation uses:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Future alternatives may include LaBSE or LASER.

## 3. Manual Evaluation

Automatic metrics such as BLEU and chrF are useful, but they cannot fully measure semantic correctness. Manual evaluation will compare baseline and fine-tuned outputs using:

- meaning preservation
- fluency
- completeness
- named entity correctness
- missing word errors
- hallucination
- tense/gender errors
- literal or unnatural translation

A manual evaluation template has been created for comparing baseline and fine-tuned predictions.

## 4. Baseline vs Fine-Tuned Error Analysis

The next stage will analyze which types of errors are reduced after LoRA fine-tuning. This will help identify whether the fine-tuned model improves only surface similarity or also improves real translation quality.

Important comparisons:

- baseline better cases
- fine-tuned better cases
- both wrong cases
- named entity mistakes
- missing information
- hallucinated content

## 5. Hindi Translation Extension

The project will also compare:

1. Direct Pashto-to-Hindi translation
2. Pivot Pashto-to-English-to-Hindi translation

The pivot approach is expected to be stronger because Pashto-to-English can be improved and evaluated first. Later, IndicTrans2 can be used for the English-to-Hindi stage.

## Expected Deliverables

- Semantic similarity scored dataset
- Semantically filtered training dataset
- Semantic similarity distribution graph
- Manual evaluation template
- Baseline vs fine-tuned comparison table
- Updated research report
- Direct vs pivot Hindi analysis plan

## Important Note

These tasks represent the next research stage. They should be marked as completed only after the scripts are run and the generated outputs are reviewed.
"""

Path("docs/week5_next_research_work.md").write_text(report, encoding="utf-8")

print("Created docs/week5_next_research_work.md")
