import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("docs", exist_ok=True)

score_files = list(Path("outputs/tables").glob("week4_scores_*.csv"))

rows = []

for f in score_files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            row = df.iloc[0].to_dict()
            row["score_file"] = str(f)
            rows.append(row)
    except Exception as e:
        print("Skipped:", f, e)

if len(rows) == 0:
    print("No score files found.")
    print("Run evaluation first.")
    raise SystemExit

df = pd.DataFrame(rows)

if "model" not in df.columns:
    print("Score files found but model column missing.")
    print(df.columns)
    raise SystemExit

df = df.drop_duplicates(subset=["model"], keep="last")

def short_name(model):
    model = str(model)
    if "facebook/nllb-200-distilled-600M" in model:
        return "Baseline NLLB"
    if "semantic_filtered_8000" in model:
        return "Semantic-filtered LoRA 8000"
    if "10k" in model or "10K" in model:
        return "Original LoRA 10k"
    if "3000" in model:
        return "LoRA 3000"
    if "1000" in model:
        return "LoRA 1000"
    if "500" in model:
        return "LoRA 500"
    if "100" in model:
        return "LoRA 100"
    if "20" in model:
        return "LoRA 20"
    return model.replace("models\\", "").replace("models/", "")

df["model_short"] = df["model"].apply(short_name)

preferred_order = {
    "Baseline NLLB": 0,
    "LoRA 20": 1,
    "LoRA 100": 2,
    "LoRA 500": 3,
    "LoRA 1000": 4,
    "LoRA 3000": 5,
    "Original LoRA 10k": 6,
    "Semantic-filtered LoRA 8000": 7
}

df["order"] = df["model_short"].map(preferred_order).fillna(99)
df = df.sort_values("order").drop(columns=["order"])

df.to_csv("outputs/tables/remaining_checkpoint_comparison.csv", index=False)

if "chrF" in df.columns:
    plt.figure(figsize=(11, 5))
    bars = plt.bar(df["model_short"], df["chrF"])
    plt.title("Checkpoint Comparison using chrF")
    plt.xlabel("Model checkpoint")
    plt.ylabel("chrF")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("outputs/figures/remaining_chrf_checkpoint_comparison.png", dpi=300)
    plt.close()

if "BLEU" in df.columns:
    plt.figure(figsize=(11, 5))
    bars = plt.bar(df["model_short"], df["BLEU"])
    plt.title("Checkpoint Comparison using BLEU")
    plt.xlabel("Model checkpoint")
    plt.ylabel("BLEU")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("outputs/figures/remaining_bleu_checkpoint_comparison.png", dpi=300)
    plt.close()

report = "# Remaining Work: Checkpoint Comparison\n\n"
report += "This report compares the baseline NLLB model, original LoRA checkpoints, and the semantic-filtered LoRA checkpoint.\n\n"

report += "## Comparison Table\n\n"
report += "| Model | Samples | BLEU | chrF |\n"
report += "|---|---:|---:|---:|\n"

for _, r in df.iterrows():
    samples = int(r["samples"]) if "samples" in df.columns and pd.notna(r.get("samples")) else ""
    bleu = r["BLEU"] if "BLEU" in df.columns else ""
    chrf = r["chrF"] if "chrF" in df.columns else ""
    report += f"| {r['model_short']} | {samples} | {bleu} | {chrf} |\n"

report += "\n## Graphs\n\n"
report += "![chrF Checkpoint Comparison](../outputs/figures/remaining_chrf_checkpoint_comparison.png)\n\n"
report += "![BLEU Checkpoint Comparison](../outputs/figures/remaining_bleu_checkpoint_comparison.png)\n\n"

report += "## Interpretation\n\n"
report += "The semantic-filtered LoRA checkpoint is compared against the baseline and previous LoRA checkpoints to check whether cleaner sentence-pair selection improves translation quality. chrF is especially important here because it captures character-level similarity, which is useful for low-resource translation evaluation.\n"

Path("docs/remaining_checkpoint_comparison.md").write_text(report, encoding="utf-8")

print("Checkpoint comparison generated successfully.")
print("Saved: outputs/tables/remaining_checkpoint_comparison.csv")
print("Saved: outputs/figures/remaining_chrf_checkpoint_comparison.png")
print("Saved: outputs/figures/remaining_bleu_checkpoint_comparison.png")
print("Saved: docs/remaining_checkpoint_comparison.md")
