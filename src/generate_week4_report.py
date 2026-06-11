import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("docs", exist_ok=True)

files = list(Path("outputs/tables").glob("week4_scores_*.csv"))

rows = []

for f in files:
    df = pd.read_csv(f)
    rows.append(df.iloc[0].to_dict())

if len(rows) == 0:
    print("No Week 4 score files found. Run evaluation first.")
    raise SystemExit

scores = pd.DataFrame(rows)
scores.to_csv("outputs/tables/week4_model_comparison.csv", index=False)

plt.figure(figsize=(9, 5))
bars = plt.bar(scores["model"], scores["BLEU"])
plt.title("Week 4 BLEU Comparison")
plt.xlabel("Model")
plt.ylabel("BLEU")
plt.xticks(rotation=20, ha="right")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/week4_bleu_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
bars = plt.bar(scores["model"], scores["chrF"])
plt.title("Week 4 chrF Comparison")
plt.xlabel("Model")
plt.ylabel("chrF")
plt.xticks(rotation=20, ha="right")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/week4_chrf_comparison.png", dpi=300)
plt.close()

report = "# Week 4 Fine-Tuning Evaluation Report\n\n"
report += "## Objective\n\n"
report += "Week 4 compares baseline NLLB and LoRA fine-tuned NLLB on the gold test candidate set.\n\n"
report += "## Model Comparison\n\n"
report += "| Model | Samples | BLEU | chrF |\n"
report += "|---|---:|---:|---:|\n"

for _, r in scores.iterrows():
    report += f"| {r['model']} | {int(r['samples'])} | {r['BLEU']} | {r['chrF']} |\n"

report += "\n## Graphs\n\n"
report += "![BLEU Comparison](../outputs/figures/week4_bleu_comparison.png)\n\n"
report += "![chrF Comparison](../outputs/figures/week4_chrf_comparison.png)\n\n"
report += "## Interpretation\n\n"
report += "The comparison shows whether LoRA fine-tuning improves the baseline model. If scores improve, it indicates better adaptation to the Pashto-English dataset. If improvement is small, the result still gives research insight about low-resource data quality and training size limitations.\n"

Path("docs/week4_finetuning_report.md").write_text(report, encoding="utf-8")

print("Week 4 report generated.")
