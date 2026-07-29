import pandas as pd
from pathlib import Path

input_path = Path(
    "outputs/tables/research_human_evaluation_form.csv"
)

output_path = Path(
    "outputs/tables/research_human_evaluation_filled.csv"
)

if not input_path.exists():
    raise FileNotFoundError(
        f"File not found: {input_path}"
    )

df = pd.read_csv(
    input_path,
    encoding="utf-8-sig"
)

old_error_columns = [
    "missing_information",
    "hallucination",
    "named_entity_error",
]

df = df.drop(
    columns=[
        column
        for column in old_error_columns
        if column in df.columns
    ],
    errors="ignore",
)

new_columns = [
    "missing_information_a",
    "missing_information_b",
    "missing_information_c",
    "hallucination_a",
    "hallucination_b",
    "hallucination_c",
    "named_entity_error_a",
    "named_entity_error_b",
    "named_entity_error_c",
]

for column in new_columns:
    df[column] = ""

ordered_columns = [
    "sample_id",
    "pashto_source",
    "english_reference",
    "system_a_output",
    "system_b_output",
    "system_c_output",
    "best_system",
    "adequacy_a_1_to_5",
    "adequacy_b_1_to_5",
    "adequacy_c_1_to_5",
    "fluency_a_1_to_5",
    "fluency_b_1_to_5",
    "fluency_c_1_to_5",
    "missing_information_a",
    "missing_information_b",
    "missing_information_c",
    "hallucination_a",
    "hallucination_b",
    "hallucination_c",
    "named_entity_error_a",
    "named_entity_error_b",
    "named_entity_error_c",
    "comments",
]

ordered_columns = [
    column
    for column in ordered_columns
    if column in df.columns
]

df = df[ordered_columns]

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("Improved human-evaluation form created.")
print(f"File: {output_path}")
print(f"Samples: {len(df)}")
