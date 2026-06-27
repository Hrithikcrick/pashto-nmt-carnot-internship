import os
import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)

baseline_file = "outputs/tables/week4_baseline_predictions.csv"

candidate_files = [
    "outputs/tables/week4_lora_10k_predictions.csv",
    "outputs/tables/week4_lora_3000_predictions.csv",
    "outputs/tables/week4_lora_1000_predictions.csv",
    "outputs/tables/week4_lora_500_predictions.csv",
    "outputs/tables/week4_lora_trial_20_predictions.csv"
]

fine_file = None
for f in candidate_files:
    if os.path.exists(f):
        fine_file = f
        break

if not os.path.exists(baseline_file):
    print("Baseline prediction file missing:", baseline_file)
    print("Create baseline predictions first using src/evaluate_week4_model.py")
    raise SystemExit

if fine_file is None:
    print("No fine-tuned prediction file found.")
    print("Create at least one LoRA prediction file first.")
    raise SystemExit

base = pd.read_csv(baseline_file)
fine = pd.read_csv(fine_file)

def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in low:
            return low[name]
    return None

pcol = find_col(base, ["pashto", "ps", "source", "pbt", "input"])
refcol = find_col(base, ["english_reference", "english", "en", "target", "reference", "eng"])
pred_base = find_col(base, ["prediction"])
pred_fine = find_col(fine, ["prediction"])

n = min(len(base), len(fine), 50)

template = pd.DataFrame({
    "sample_id": list(range(1, n + 1)),
    "pashto_source": base[pcol].astype(str).head(n) if pcol else "",
    "english_reference": base[refcol].astype(str).head(n) if refcol else "",
    "baseline_prediction": base[pred_base].astype(str).head(n) if pred_base else "",
    "finetuned_prediction": fine[pred_fine].astype(str).head(n) if pred_fine else "",
    "better_output_baseline_or_finetuned": "",
    "meaning_preservation_score_1_to_5": "",
    "fluency_score_1_to_5": "",
    "completeness_score_1_to_5": "",
    "named_entity_correct_yes_no": "",
    "main_error_type": "",
    "manual_comment": ""
})

out = "outputs/tables/week5_manual_evaluation_template.csv"
template.to_csv(out, index=False, encoding="utf-8-sig")

print("Manual evaluation template created:", out)
print("Fine-tuned file used:", fine_file)
