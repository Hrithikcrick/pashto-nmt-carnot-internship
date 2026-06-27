import os
import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)

baseline_file = "outputs/tables/week4_baseline_predictions.csv"
fine_file = "outputs/tables/week4_lora_10k_predictions.csv"

if not os.path.exists(fine_file):
    fine_file = "outputs/tables/week4_lora_3000_predictions.csv"

if not os.path.exists(fine_file):
    fine_file = "outputs/tables/week4_lora_1000_predictions.csv"

if not os.path.exists(fine_file):
    fine_file = "outputs/tables/week4_lora_500_predictions.csv"

if not os.path.exists(fine_file):
    fine_file = "outputs/tables/week4_lora_trial_20_predictions.csv"

if not os.path.exists(baseline_file):
    print("Missing baseline prediction file:", baseline_file)
    raise SystemExit

if not os.path.exists(fine_file):
    print("Missing fine-tuned prediction file.")
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
predcol_base = find_col(base, ["prediction"])
predcol_fine = find_col(fine, ["prediction"])

n = min(len(base), len(fine), 50)

template = pd.DataFrame({
    "sample_id": list(range(1, n + 1)),
    "pashto_source": base[pcol].astype(str).head(n) if pcol else "",
    "english_reference": base[refcol].astype(str).head(n) if refcol else "",
    "baseline_prediction": base[predcol_base].astype(str).head(n) if predcol_base else "",
    "finetuned_prediction": fine[predcol_fine].astype(str).head(n) if predcol_fine else "",
    "better_output_baseline_or_finetuned": "",
    "meaning_preservation_score_1_to_5": "",
    "fluency_score_1_to_5": "",
    "completeness_score_1_to_5": "",
    "named_entity_correct_yes_no": "",
    "main_error_type": "",
    "manual_comment": ""
})

out = "outputs/tables/manual_evaluation_baseline_vs_finetuned_template.csv"
template.to_csv(out, index=False, encoding="utf-8-sig")

print("Manual evaluation template created:")
print(out)
print("Fill this file manually for human evaluation.")
