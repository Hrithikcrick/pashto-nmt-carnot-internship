import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("docs", exist_ok=True)

def read_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, encoding="utf-8-sig")

clean = read_csv("data/cleaned_90k.csv")
train10k = read_csv("data/train_high_quality_10k.csv")
gold = read_csv("data/gold_test_candidates_100.csv")

summary = pd.DataFrame({
    "file": [
        "cleaned_90k.csv",
        "train_high_quality_10k.csv",
        "gold_test_candidates_100.csv"
    ],
    "purpose": [
        "Final cleaned Pashto-English dataset",
        "High-quality subset prepared for fine-tuning",
        "Clean test/gold candidate set for evaluation"
    ],
    "rows": [
        len(clean),
        len(train10k),
        len(gold)
    ],
    "columns": [
        len(clean.columns),
        len(train10k.columns),
        len(gold.columns)
    ]
})

summary.to_csv("outputs/tables/week3_final_dataset_summary.csv", index=False)

plt.figure(figsize=(9, 5))
bars = plt.bar(summary["file"], summary["rows"])
plt.title("Week 3 Final Dataset Summary")
plt.xlabel("Dataset file")
plt.ylabel("Number of rows")
plt.xticks(rotation=20, ha="right")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{int(y):,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/week3_final_dataset_summary.png", dpi=300)
plt.close()

status = pd.DataFrame({
    "task": [
        "Cleaned 90k dataset prepared",
        "High-quality 10k subset prepared",
        "Gold test candidates prepared",
        "Clean-test baseline score integrated",
        "Manual error analysis integrated",
        "Week 3 report generated",
        "LoRA fine-tuning"
    ],
    "completion": [100, 100, 100, 100, 100, 100, 0]
})

status.to_csv("outputs/tables/week3_final_status.csv", index=False)

plt.figure(figsize=(10, 5))
bars = plt.barh(status["task"], status["completion"])
plt.title("Week 3 Completion Status")
plt.xlabel("Completion percentage")
plt.ylabel("Task")
plt.xlim(0, 100)
for bar in bars:
    x = bar.get_width()
    plt.text(x + 1, bar.get_y() + bar.get_height()/2, f"{int(x)}%", va="center")
plt.tight_layout()
plt.savefig("outputs/figures/week3_final_status.png", dpi=300)
plt.close()

if os.path.exists("outputs/tables/week3_clean_test_baseline_scores.csv"):
    scores = read_csv("outputs/tables/week3_clean_test_baseline_scores.csv")
    scores.to_csv("outputs/tables/week3_final_clean_test_scores.csv", index=False)

    numeric_cols = []
    for c in scores.columns:
        if pd.api.types.is_numeric_dtype(scores[c]):
            numeric_cols.append(c)

    if len(numeric_cols) > 0:
        row = scores.iloc[0]
        plot_data = []
        for c in numeric_cols:
            plot_data.append({"metric": c, "value": row[c]})

        metric_df = pd.DataFrame(plot_data)
        metric_df.to_csv("outputs/tables/week3_final_score_plot_data.csv", index=False)

        plt.figure(figsize=(8, 5))
        bars = plt.bar(metric_df["metric"], metric_df["value"])
        plt.title("Week 3 Clean-Test Baseline Evaluation")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=20, ha="right")
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig("outputs/figures/week3_final_clean_test_scores.png", dpi=300)
        plt.close()

if os.path.exists("outputs/tables/week3_manual_error_analysis_clean_test.csv"):
    err = read_csv("outputs/tables/week3_manual_error_analysis_clean_test.csv")

    possible_cols = ["error_type", "Error Type", "error", "Error", "category", "Category"]
    error_col = None

    for c in possible_cols:
        if c in err.columns:
            error_col = c
            break

    if error_col is None:
        error_col = err.columns[-1]

    counts = err[error_col].fillna("not_marked").astype(str).value_counts().reset_index()
    counts.columns = ["error_type", "count"]
    counts.to_csv("outputs/tables/week3_final_error_type_counts.csv", index=False)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(counts["error_type"], counts["count"])
    plt.title("Week 3 Manual Error Analysis")
    plt.xlabel("Error type")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y, str(int(y)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("outputs/figures/week3_final_error_analysis.png", dpi=300)
    plt.close()

report = "# Week 3 Final Dataset Quality and Evaluation Report\n\n"

report += "## Objective\n\n"
report += "The goal of Week 3 was to complete dataset-quality preparation before fine-tuning. This stage does not include LoRA fine-tuning. It focuses on cleaned data integration, high-quality subset preparation, gold test candidate preparation, clean-test baseline score integration, and manual error analysis.\n\n"

report += "## Final Week 3 Dataset Summary\n\n"
report += "| File | Purpose | Rows | Columns |\n"
report += "|---|---|---:|---:|\n"

for _, r in summary.iterrows():
    report += f"| `{r['file']}` | {r['purpose']} | {int(r['rows'])} | {int(r['columns'])} |\n"

report += "\n![Week 3 Final Dataset Summary](../outputs/figures/week3_final_dataset_summary.png)\n\n"

report += "## Week 3 Completion Status\n\n"
report += "![Week 3 Final Status](../outputs/figures/week3_final_status.png)\n\n"

if os.path.exists("outputs/figures/week3_final_clean_test_scores.png"):
    report += "## Clean-Test Baseline Evaluation\n\n"
    report += "The clean-test baseline evaluation file was integrated and visualized. This provides the evaluation foundation before fine-tuning.\n\n"
    report += "![Week 3 Clean-Test Scores](../outputs/figures/week3_final_clean_test_scores.png)\n\n"

if os.path.exists("outputs/figures/week3_final_error_analysis.png"):
    report += "## Manual Error Analysis\n\n"
    report += "Manual error analysis was integrated to understand common translation issues before fine-tuning.\n\n"
    report += "![Week 3 Manual Error Analysis](../outputs/figures/week3_final_error_analysis.png)\n\n"

report += "## Week 3 Conclusion\n\n"
report += "Week 3 completed the dataset-quality foundation required for fine-tuning. The cleaned dataset, high-quality 10k subset, gold test candidates, clean-test baseline scores, and manual error analysis are now organized in the research repository. The next stage is Week 4 LoRA fine-tuning of NLLB using the high-quality 10k subset.\n\n"

report += "## Next Step\n\n"
report += "Proceed to Week 4: LoRA fine-tuning of `facebook/nllb-200-distilled-600M` on `data/train_high_quality_10k.csv` using Google Colab GPU.\n"

Path("docs/week3_final_report.md").write_text(report, encoding="utf-8")

readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")

extra = """
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
"""

if "## Week 3 Final Dataset Quality and Evaluation" not in readme:
    readme_path.write_text(readme + "\n" + extra, encoding="utf-8")

print("Week 3 final report completed.")
print("Generated docs/week3_final_report.md")
print("Generated Week 3 final graphs and tables.")
