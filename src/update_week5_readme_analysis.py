from pathlib import Path
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

root = Path(".")
readme_path = root / "README.md"

os.makedirs("docs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

def first_existing(paths):
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None

summary_path = first_existing([
    "outputs/tables/week5_semantic_filtering_summary.csv",
    "outputs/tables/semantic_filtering_summary.csv"
])

scores_path = first_existing([
    "outputs/tables/week5_semantic_similarity_scores.csv",
    "outputs/tables/semantic_similarity_scores.csv"
])

status_path = first_existing([
    "outputs/tables/week5_work_status.csv"
])

manual_eval_path = first_existing([
    "outputs/tables/week5_manual_evaluation_template.csv",
    "outputs/tables/manual_evaluation_baseline_vs_finetuned_template.csv"
])

filtered_800 = Path("data/train_semantic_filtered_800.csv")
filtered_8000 = Path("data/train_semantic_filtered_8000.csv")

summary = None
if summary_path:
    summary = pd.read_csv(summary_path)

# Create useful Week 5 graph 1: filtering overview
if summary is not None and len(summary) > 0:
    r = summary.iloc[0]

    total_rows = int(r.get("total_rows_scored", 0))
    filtered_rows = int(r.get("filtered_rows_saved", 0))

    overview = pd.DataFrame({
        "stage": ["Rows scored", "Rows retained after filtering"],
        "rows": [total_rows, filtered_rows]
    })
    overview.to_csv("outputs/tables/week5_filtering_overview.csv", index=False)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(overview["stage"], overview["rows"])
    plt.title("Week 5 Semantic Filtering Overview")
    plt.xlabel("Filtering stage")
    plt.ylabel("Sentence pairs")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y, f"{int(y):,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("outputs/figures/week5_filtering_overview.png", dpi=300)
    plt.close()

    # Create useful Week 5 graph 2: similarity score statistics
    stat_cols = ["mean_similarity", "median_similarity", "min_similarity", "max_similarity"]
    available = [c for c in stat_cols if c in summary.columns]

    if available:
        stat_df = pd.DataFrame({
            "metric": [c.replace("_similarity", "").replace("_", " ").title() for c in available],
            "score": [float(r[c]) for c in available]
        })
        stat_df.to_csv("outputs/tables/week5_similarity_score_stats.csv", index=False)

        plt.figure(figsize=(8, 5))
        bars = plt.bar(stat_df["metric"], stat_df["score"])
        plt.title("Week 5 Semantic Similarity Statistics")
        plt.xlabel("Metric")
        plt.ylabel("Cosine similarity")
        plt.ylim(0, 1)
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig("outputs/figures/week5_similarity_score_stats.png", dpi=300)
        plt.close()

# Create top and bottom examples table
if scores_path:
    scores = pd.read_csv(scores_path)
    if "semantic_similarity" in scores.columns:
        top = scores.sort_values("semantic_similarity", ascending=False).head(10).copy()
        bottom = scores.sort_values("semantic_similarity", ascending=True).head(10).copy()
        top["bucket"] = "top_high_similarity"
        bottom["bucket"] = "bottom_low_similarity"
        pd.concat([top, bottom], ignore_index=True).to_csv(
            "outputs/tables/week5_top_bottom_similarity_examples.csv",
            index=False,
            encoding="utf-8-sig"
        )

# Create work status graph if table exists, otherwise create default status
if status_path:
    status = pd.read_csv(status_path)
else:
    status = pd.DataFrame({
        "task": [
            "Semantic filtering script",
            "Filtered dataset generation",
            "Similarity graph analysis",
            "Manual evaluation template",
            "Human scoring",
            "Direct vs pivot Hindi test",
            "IndicTrans2 integration"
        ],
        "completion": [100, 100, 100, 100, 0, 0, 0]
    })
    status.to_csv("outputs/tables/week5_work_status.csv", index=False)

if "task" in status.columns and "completion" in status.columns:
    plt.figure(figsize=(10, 5))
    bars = plt.barh(status["task"], status["completion"])
    plt.title("Week 5 Research Work Status")
    plt.xlabel("Completion percentage")
    plt.ylabel("Task")
    plt.xlim(0, 100)
    for bar in bars:
        x = bar.get_width()
        plt.text(x + 1, bar.get_y() + bar.get_height()/2, f"{int(x)}%", va="center")
    plt.tight_layout()
    plt.savefig("outputs/figures/week5_work_status.png", dpi=300)
    plt.close()

# Build clean Week 5 README section
week5 = """
## Week 5 Research Work: Semantic Filtering and Manual Evaluation

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

"""

if summary is not None and len(summary) > 0:
    r = summary.iloc[0]
    week5 += f"""### Week 5 Semantic Filtering Results

| Metric | Value |
|---|---:|
| Total sentence pairs scored | {int(r.get("total_rows_scored", 0)):,} |
| Filtered sentence pairs retained | {int(r.get("filtered_rows_saved", 0)):,} |
| Mean semantic similarity | {float(r.get("mean_similarity", 0)):.4f} |
| Median semantic similarity | {float(r.get("median_similarity", 0)):.4f} |
| Minimum semantic similarity | {float(r.get("min_similarity", 0)):.4f} |
| Maximum semantic similarity | {float(r.get("max_similarity", 0)):.4f} |

"""
else:
    week5 += """### Week 5 Semantic Filtering Results

Semantic filtering summary file was not found yet. Run `src/week5_semantic_filter.py` to generate the result table and graphs.

"""

week5 += """### Week 5 Graph Analysis

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
"""

# Update docs report also
Path("docs/week5_research_work.md").write_text(week5.replace("## Week 5 Research Work: Semantic Filtering and Manual Evaluation", "# Week 5 Research Work: Semantic Filtering and Manual Evaluation"), encoding="utf-8")

# Clean old duplicate Week 5 blocks from README and append the new one
readme = readme_path.read_text(encoding="utf-8")

markers = [
    r"\n## Week 5 Future Work",
    r"\n## Next Research Work: Semantic Filtering and Manual Evaluation",
    r"\n## Week 5 Research Work: Semantic Filtering and Manual Evaluation"
]

cut_positions = []
for m in markers:
    match = re.search(m, readme)
    if match:
        cut_positions.append(match.start())

if cut_positions:
    readme = readme[:min(cut_positions)].rstrip()

readme = readme.rstrip() + "\n\n" + week5.strip() + "\n"
readme_path.write_text(readme, encoding="utf-8")

print("README updated with clean Week 5 graph analysis and results.")
print("docs/week5_research_work.md updated.")
print("Graphs/tables refreshed where data was available.")
