from pathlib import Path
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

def done(path):
    return Path(path).exists()

rows = [
    {
        "task": "Semantic filtering script",
        "completion": 100 if done("src/week5_semantic_filter.py") else 0
    },
    {
        "task": "Filtered dataset generation",
        "completion": 100 if done("data/train_semantic_filtered_8000.csv") or done("data/train_semantic_filtered_800.csv") else 0
    },
    {
        "task": "Semantic LoRA training",
        "completion": 100 if done("outputs/tables/remaining_semantic_lora_8000_predictions.csv") or done("outputs/tables/week4_scores_models_nllb_lora_semantic_filtered_8000.csv") else 0
    },
    {
        "task": "Checkpoint comparison",
        "completion": 100 if done("outputs/tables/remaining_checkpoint_comparison.csv") else 0
    },
    {
        "task": "Direct vs pivot Hindi comparison",
        "completion": 100 if done("outputs/tables/remaining_direct_vs_pivot_hindi.csv") or done("outputs/tables/remaining_direct_vs_pivot_hindi_20.csv") else 0
    },
    {
        "task": "IndicTrans2 exploration note",
        "completion": 100 if done("docs/remaining_indictrans2_exploration.md") else 0
    },
    {
        "task": "Manual human scoring",
        "completion": 0
    }
]

df = pd.DataFrame(rows)
df.to_csv("outputs/tables/week5_work_status.csv", index=False)
df.to_csv("outputs/tables/remaining_work_updated_status.csv", index=False)

plt.figure(figsize=(11, 5.5))
bars = plt.barh(df["task"], df["completion"])
plt.title("Remaining Research Work Status")
plt.xlabel("Completion percentage")
plt.ylabel("Task")
plt.xlim(0, 100)

for bar in bars:
    x = bar.get_width()
    plt.text(x + 1, bar.get_y() + bar.get_height()/2, f"{int(x)}%", va="center")

plt.tight_layout()
plt.savefig("outputs/figures/week5_work_status.png", dpi=300)
plt.savefig("outputs/figures/remaining_work_updated_status.png", dpi=300)
plt.close()

readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")

old_para = """This graph tracks completed and pending research tasks. Semantic filtering and evaluation-template preparation are completed, while human scoring, direct-vs-pivot Hindi evaluation, and IndicTrans2 integration remain future work."""

new_para = """This graph tracks the updated remaining research status. Semantic filtering, filtered-data generation, semantic-filtered LoRA evaluation, checkpoint comparison, direct-vs-pivot Hindi comparison, and IndicTrans2 exploration note are completed. Manual human scoring is the main remaining task."""

readme = readme.replace(old_para, new_para)

# Add a clean updated interpretation section if not already present
updated_block = """
### Updated Remaining Research Status

The remaining research work has been updated after completing the semantic-filtered LoRA experiment and direct-vs-pivot Hindi comparison.

Completed items:

- Semantic similarity filtering.
- Semantically filtered dataset generation.
- LoRA training using semantically filtered data.
- Semantic-filtered LoRA evaluation.
- Baseline vs original LoRA vs semantic-filtered LoRA checkpoint comparison.
- Direct Pashto-to-Hindi vs pivot Pashto-English-Hindi comparison.
- IndicTrans2 exploration note for the English-to-Hindi stage.

Pending item:

- Manual human scoring of selected baseline, fine-tuned, and Hindi translation outputs.

![Updated Remaining Research Work Status](outputs/figures/remaining_work_updated_status.png)
"""

if "### Updated Remaining Research Status" not in readme:
    readme = readme.rstrip() + "\n\n" + updated_block + "\n"

readme_path.write_text(readme, encoding="utf-8")

print("Updated work status graph and README.")
print("Saved outputs/figures/week5_work_status.png")
print("Saved outputs/figures/remaining_work_updated_status.png")
print("Saved outputs/tables/remaining_work_updated_status.csv")
