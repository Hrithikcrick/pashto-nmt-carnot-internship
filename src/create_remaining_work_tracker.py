import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("docs", exist_ok=True)

tasks = pd.DataFrame([
    {"remaining_work": "Manually score selected baseline and fine-tuned translations", "status": "pending", "priority": "high"},
    {"remaining_work": "Train LoRA again using semantically filtered data", "status": "pending", "priority": "high"},
    {"remaining_work": "Compare baseline, original LoRA, and semantic-filtered LoRA checkpoints", "status": "pending", "priority": "high"},
    {"remaining_work": "Compare direct Pashto-to-Hindi with pivot Pashto-English-Hindi translation", "status": "pending", "priority": "medium"},
    {"remaining_work": "Explore IndicTrans2 for English-to-Hindi stage", "status": "pending", "priority": "medium"}
])

tasks.to_csv("outputs/tables/remaining_work_status.csv", index=False)

plot_df = pd.DataFrame({
    "task": [
        "Manual scoring",
        "Semantic LoRA training",
        "Checkpoint comparison",
        "Direct vs pivot Hindi",
        "IndicTrans2 exploration"
    ],
    "completion": [0, 0, 0, 0, 0]
})

plot_df.to_csv("outputs/tables/remaining_work_progress.csv", index=False)

plt.figure(figsize=(10, 5))
bars = plt.barh(plot_df["task"], plot_df["completion"])
plt.title("Remaining Research Work Progress")
plt.xlabel("Completion percentage")
plt.ylabel("Task")
plt.xlim(0, 100)
for bar in bars:
    x = bar.get_width()
    plt.text(x + 1, bar.get_y() + bar.get_height()/2, f"{int(x)}%", va="center")
plt.tight_layout()
plt.savefig("outputs/figures/remaining_work_progress.png", dpi=300)
plt.close()

doc = """# Remaining Research Work

This file tracks the remaining research work for the Pashto Neural Machine Translation project.

## Remaining Tasks

1. Manually score selected baseline and fine-tuned translations.
2. Train LoRA again using semantically filtered data.
3. Compare baseline, original LoRA, and semantic-filtered LoRA checkpoints.
4. Compare direct Pashto-to-Hindi translation with pivot Pashto-English-Hindi translation.
5. Explore IndicTrans2 for the English-to-Hindi stage.

## Purpose

The purpose of these remaining tasks is to move the project from initial fine-tuning results toward a stronger research-quality evaluation. Automatic metrics such as BLEU and chrF are useful, but manual scoring and error analysis are necessary to understand whether the translations are actually better in meaning, fluency, completeness, and named entity correctness.

## Expected Outputs

- Manual evaluation CSV.
- Semantic-filtered LoRA checkpoint evaluation.
- Baseline vs original LoRA vs semantic-filtered LoRA comparison table.
- Direct Hindi vs pivot Hindi output file.
- IndicTrans2 exploration notes.
- Updated graphs and README analysis.
"""

Path("docs/remaining_research_work.md").write_text(doc, encoding="utf-8")

print("Remaining work tracker created.")
