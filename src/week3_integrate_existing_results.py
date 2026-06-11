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

files = {
    "Cleaned dataset": "data/cleaned_90k.csv",
    "High-quality training subset": "data/train_high_quality_10k.csv",
    "Gold test candidates": "data/gold_test_candidates_100.csv"
}

rows = []
for name, path in files.items():
    if os.path.exists(path):
        df = read_csv(path)
        rows.append({
            "dataset": name,
            "file": path,
            "rows": len(df),
            "columns": len(df.columns)
        })

summary = pd.DataFrame(rows)
summary.to_csv("outputs/tables/week3_dataset_files_summary.csv", index=False)

plt.figure(figsize=(9, 5))
bars = plt.bar(summary["dataset"], summary["rows"])
plt.title("Week 3 Dataset Files Integrated")
plt.xlabel("Dataset file")
plt.ylabel("Number of rows")
plt.xticks(rotation=20, ha="right")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{int(y):,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/week3_dataset_files_summary.png", dpi=300)
plt.close()

status = pd.DataFrame({
    "task": [
        "Cleaned 90k dataset copied",
        "High-quality 10k subset copied",
        "Gold test candidates copied",
        "Baseline clean-test scores copied",
        "Manual error analysis copied",
        "Week 3 report generated",
        "Fine-tuning completed"
    ],
    "completion_percent": [100, 100, 100, 100, 100, 100, 0]
})
status.to_csv("outputs/tables/week3_status_summary.csv", index=False)

plt.figure(figsize=(10, 5))
bars = plt.barh(status["task"], status["completion_percent"])
plt.title("Week 3 Dataset Quality Work Status")
plt.xlabel("Completion percentage")
plt.ylabel("Task")
plt.xlim(0, 100)
for bar in bars:
    x = bar.get_width()
    plt.text(x + 1, bar.get_y() + bar.get_height()/2, f"{int(x)}%", va="center")
plt.tight_layout()
plt.savefig("outputs/figures/week3_status_summary.png", dpi=300)
plt.close()

if os.path.exists("outputs/tables/week3_clean_test_baseline_scores.csv"):
    score = read_csv("outputs/tables/week3_clean_test_baseline_scores.csv")
    score.to_csv("outputs/tables/week3_clean_test_baseline_scores_copy.csv", index=False)

    numeric_cols = []
    for c in score.columns:
        if pd.api.types.is_numeric_dtype(score[c]):
            numeric_cols.append(c)

    if len(numeric_cols) > 0:
        first = score.iloc[0]
        plot_rows = []
        for c in numeric_cols:
            plot_rows.append({"metric": c, "value": first[c]})

        metric_df = pd.DataFrame(plot_rows)
        metric_df.to_csv("outputs/tables/week3_clean_test_metric_plot_data.csv", index=False)

        plt.figure(figsize=(8, 5))
        bars = plt.bar(metric_df["metric"], metric_df["value"])
        plt.title("Week 3 Clean Test Baseline Scores")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=20, ha="right")
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig("outputs/figures/week3_clean_test_baseline_scores.png", dpi=300)
        plt.close()

if os.path.exists("outputs/tables/week3_manual_error_analysis_clean_test.csv"):
    err = read_csv("outputs/tables/week3_manual_error_analysis_clean_test.csv")

    possible_cols = ["error_type", "Error Type", "error", "Error", "category", "Category"]
    col = None
    for c in possible_cols:
        if c in err.columns:
            col = c
            break

    if col is not None:
        counts = err[col].fillna("not_marked").astype(str).value_counts().reset_index()
        counts.columns = ["error_type", "count"]
        counts.to_csv("outputs/tables/week3_error_type_counts.csv", index=False)

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
        plt.savefig("outputs/figures/week3_error_type_counts.png", dpi=300)
        plt.close()

report = "# Week 3 Dataset Quality Integration Report\n\n"
report += "## Objective\n\n"
report += "The objective of Week 3 is to organize and integrate the cleaned Pashto-English dataset, high-quality subset, clean test set, baseline clean-test scores, and manual error analysis files into the GitHub research repository.\n\n"
report += "This week does not claim fine-tuning results. It prepares the dataset-quality foundation required before fine-tuning.\n\n"

report += "## Dataset Files Integrated\n\n"
report += "| Dataset | File | Rows | Columns |\n"
report += "|---|---|---:|---:|\n"
for _, r in summary.iterrows():
    report += f"| {r['dataset']} | `{r['file']}` | {int(r['rows'])} | {int(r['columns'])} |\n"

report += "\n![Week 3 Dataset Files Summary](../outputs/figures/week3_dataset_files_summary.png)\n\n"

report += "## Week 3 Work Status\n\n"
report += "![Week 3 Status Summary](../outputs/figures/week3_status_summary.png)\n\n"

if os.path.exists("outputs/figures/week3_clean_test_baseline_scores.png"):
    report += "## Clean Test Baseline Scores\n\n"
    report += "The clean-test baseline score file was copied from the previous project workspace and added to the current repository for Week 3 analysis.\n\n"
    report += "![Week 3 Clean Test Baseline Scores](../outputs/figures/week3_clean_test_baseline_scores.png)\n\n"

if os.path.exists("outputs/figures/week3_error_type_counts.png"):
    report += "## Manual Error Analysis\n\n"
    report += "Manual error analysis was integrated and visualized from the clean-test evaluation file.\n\n"
    report += "![Week 3 Error Type Counts](../outputs/figures/week3_error_type_counts.png)\n\n"

report += "## Files Generated\n\n"
report += "- `data/gold_test_candidates_100.csv`\n"
report += "- `outputs/tables/week3_dataset_files_summary.csv`\n"
report += "- `outputs/tables/week3_status_summary.csv`\n"
report += "- `outputs/tables/week3_high_quality_subset_summary.csv`\n"
report += "- `outputs/tables/week3_clean_test_baseline_scores.csv`\n"
report += "- `outputs/tables/week3_manual_error_analysis_clean_test.csv`\n"
report += "- `outputs/figures/week3_dataset_files_summary.png`\n"
report += "- `outputs/figures/week3_status_summary.png`\n"
report += "- `docs/week3_dataset_quality_report.md`\n\n"

report += "## Next Step\n\n"
report += "The next step is to use the high-quality 10k subset for NLLB fine-tuning. After fine-tuning, the model will be compared with the baseline using BLEU, chrF, inference time, and manual evaluation.\n"

Path("docs/week3_dataset_quality_report.md").write_text(report, encoding="utf-8")

readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")

extra = """
## Week 3 Dataset Quality Integration

Week 3 work integrates cleaned dataset files, high-quality training subset, gold test candidates, clean-test baseline scores, and manual error analysis into the research repository.

### Week 3 Graphs

![Week 3 Dataset Files Summary](outputs/figures/week3_dataset_files_summary.png)

![Week 3 Status Summary](outputs/figures/week3_status_summary.png)

### Week 3 Report

- `docs/week3_dataset_quality_report.md`

### Week 3 Tables

- `outputs/tables/week3_dataset_files_summary.csv`
- `outputs/tables/week3_status_summary.csv`
- `outputs/tables/week3_high_quality_subset_summary.csv`
- `outputs/tables/week3_clean_test_baseline_scores.csv`
- `outputs/tables/week3_manual_error_analysis_clean_test.csv`
"""

if "## Week 3 Dataset Quality Integration" not in readme:
    readme_path.write_text(readme + "\n" + extra, encoding="utf-8")

print("Week 3 dataset quality integration completed.")
print("Report created: docs/week3_dataset_quality_report.md")
print("Graphs created in outputs/figures")
print("Tables created in outputs/tables")
