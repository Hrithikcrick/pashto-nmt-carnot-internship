from pathlib import Path
import pandas as pd


tables = Path("outputs/tables")

baseline_path = tables / "week4_baseline_predictions.csv"
original_lora_path = tables / "week4_lora_10k_predictions.csv"
semantic_lora_path = tables / "remaining_semantic_lora_8000_predictions.csv"

required_files = [
    baseline_path,
    original_lora_path,
    semantic_lora_path,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

baseline = pd.read_csv(
    baseline_path,
    encoding="utf-8-sig",
)

original_lora = pd.read_csv(
    original_lora_path,
    encoding="utf-8-sig",
)

semantic_lora = pd.read_csv(
    semantic_lora_path,
    encoding="utf-8-sig",
)

required_columns = {
    "pashto",
    "english",
    "prediction",
}

for name, dataframe in [
    ("Baseline", baseline),
    ("Original LoRA", original_lora),
    ("Semantic LoRA", semantic_lora),
]:
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}"
        )

if not (
    len(baseline)
    == len(original_lora)
    == len(semantic_lora)
):
    raise ValueError(
        "Prediction files contain different numbers of rows."
    )

baseline_pashto = (
    baseline["pashto"]
    .fillna("")
    .astype(str)
    .str.strip()
)

original_pashto = (
    original_lora["pashto"]
    .fillna("")
    .astype(str)
    .str.strip()
)

semantic_pashto = (
    semantic_lora["pashto"]
    .fillna("")
    .astype(str)
    .str.strip()
)

baseline_english = (
    baseline["english"]
    .fillna("")
    .astype(str)
    .str.strip()
)

original_english = (
    original_lora["english"]
    .fillna("")
    .astype(str)
    .str.strip()
)

semantic_english = (
    semantic_lora["english"]
    .fillna("")
    .astype(str)
    .str.strip()
)

if not baseline_pashto.equals(original_pashto):
    raise ValueError(
        "Baseline and original LoRA Pashto rows do not match."
    )

if not baseline_pashto.equals(semantic_pashto):
    raise ValueError(
        "Baseline and semantic LoRA Pashto rows do not match."
    )

if not baseline_english.equals(original_english):
    raise ValueError(
        "Baseline and original LoRA references do not match."
    )

if not baseline_english.equals(semantic_english):
    raise ValueError(
        "Baseline and semantic LoRA references do not match."
    )

combined = pd.DataFrame(
    {
        "source": baseline_pashto,
        "reference": baseline_english,
        "baseline_prediction": (
            baseline["prediction"]
            .fillna("")
            .astype(str)
            .str.strip()
        ),
        "original_lora_prediction": (
            original_lora["prediction"]
            .fillna("")
            .astype(str)
            .str.strip()
        ),
        "semantic_lora_prediction": (
            semantic_lora["prediction"]
            .fillna("")
            .astype(str)
            .str.strip()
        ),
    }
)

output_path = (
    tables
    / "conference_predictions_combined.csv"
)

combined.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print("Combined conference input created successfully.")
print(f"File: {output_path}")
print(f"Rows: {len(combined)}")
print(f"Columns: {combined.columns.tolist()}")
