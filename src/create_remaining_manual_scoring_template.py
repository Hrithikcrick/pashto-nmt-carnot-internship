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
    print("Missing:", baseline_file)
    raise SystemExit

if fine_file is None:
    print("No fine-tuned prediction file found.")
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
base_pred = find_col(base, ["prediction"])
fine_pred = find_col(fine, ["prediction"])

n = min(len(base), len(fine), 50)

out = pd.DataFrame({
    "sample_id": list(range(1, n + 1)),
    "pashto_source": base[pcol].astype(str).head(n) if pcol else "",
    "english_reference": base[refcol].astype(str).head(n) if refcol else "",
    "baseline_prediction": base[base_pred].astype(str).head(n) if base_pred else "",
    "finetuned_prediction": fine[fine_pred].astype(str).head(n) if fine_pred else "",
    "better_output": "",
    "meaning_score_1_to_5": "",
    "fluency_score_1_to_5": "",
    "completeness_score_1_to_5": "",
    "named_entity_correct_yes_no": "",
    "main_error_type": "",
    "comment": ""
})

out_file = "outputs/tables/remaining_manual_scoring_template.csv"
out.to_csv(out_file, index=False, encoding="utf-8-sig")

print("Manual scoring template created:")
print(out_file)
print("Open this CSV and fill scores manually.")
