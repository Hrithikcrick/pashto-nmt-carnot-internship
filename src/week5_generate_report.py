import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("docs", exist_ok=True)

summary_file = "outputs/tables/week5_semantic_filtering_summary.csv"

report = "# Week 5 Research Work: Semantic Filtering and Manual Evaluation\n\n"

report += "## Objective\n\n"
report += "This research stage focuses on improving the Pashto Neural Machine Translation pipeline beyond initial LoRA fine-tuning. The main tasks are semantic similarity filtering, manual evaluation, baseline versus fine-tuned error analysis, and Hindi pivot-translation planning.\n\n"

report += "## Work Added\n\n"
report += "1. Semantic filtering script using multilingual sentence embeddings.\n"
report += "2. Filtered Pashto-English training dataset generation.\n"
report += "3. Semantic similarity score table.\n"
report += "4. Semantic similarity distribution graph.\n"
report += "5. Manual evaluation template for baseline versus fine-tuned outputs.\n"
report += "6. Research plan for direct Hindi and pivot Hindi comparison.\n\n"

if os.path.exists(summary_file):
    df = pd.read_csv(summary_file)
    report += "## Semantic Filtering Summary\n\n"
    report += "| Total Rows Scored | Filtered Rows Saved | Mean Similarity | Median Similarity | Min Similarity | Max Similarity |\n"
    report += "|---:|---:|---:|---:|---:|---:|\n"
    r = df.iloc[0]
    report += f"| {int(r['total_rows_scored'])} | {int(r['filtered_rows_saved'])} | {r['mean_similarity']} | {r['median_similarity']} | {r['min_similarity']} | {r['max_similarity']} |\n\n"
    report += "![Semantic Similarity Distribution](../outputs/figures/week5_semantic_similarity_distribution.png)\n\n"

report += "## Manual Evaluation Plan\n\n"
report += "Manual evaluation will compare baseline and fine-tuned translations using meaning preservation, fluency, completeness, named entity correctness, hallucination detection, and missing-word analysis.\n\n"

report += "## Hindi Translation Extension\n\n"
report += "The next translation extension will compare direct Pashto-to-Hindi translation with pivot-based Pashto-to-English-to-Hindi translation. IndicTrans2 can be explored for the English-to-Hindi stage to improve Hindi fluency and naturalness.\n\n"

report += "## Expected Impact\n\n"
report += "Semantic filtering is expected to remove weakly aligned sentence pairs and improve the quality of future LoRA fine-tuning. Manual evaluation will provide deeper insight into whether fine-tuned outputs are semantically better than baseline outputs.\n"

Path("docs/week5_research_work.md").write_text(report, encoding="utf-8")

print("Week 5 research report generated: docs/week5_research_work.md")
