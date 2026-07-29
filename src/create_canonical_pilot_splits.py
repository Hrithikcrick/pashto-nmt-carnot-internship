import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


TRAIN_INPUT = Path("data/train_high_quality_10k.csv")
TEST_INPUT = Path("data/gold_test_candidates_100.csv")

OUTPUT_DIRECTORY = Path("data/splits")
REPORT_DIRECTORY = Path("reports")

TRAIN_OUTPUT = OUTPUT_DIRECTORY / "pilot_train.csv"
VALIDATION_OUTPUT = OUTPUT_DIRECTORY / "pilot_validation.csv"
TEST_OUTPUT = OUTPUT_DIRECTORY / "pilot_test.csv"

SUMMARY_OUTPUT = REPORT_DIRECTORY / "canonical_pilot_split_summary.csv"
DETAIL_OUTPUT = REPORT_DIRECTORY / "canonical_pilot_split_details.json"
REPORT_OUTPUT = REPORT_DIRECTORY / "canonical_pilot_split_report.md"

SEED = 42
TRAIN_FRACTION = 0.90


SOURCE_CANDIDATES = [
    "pashto",
    "source",
    "src",
    "pbt",
    "pbt_arab",
    "input",
]

TARGET_CANDIDATES = [
    "english",
    "english_reference",
    "reference",
    "target",
    "tgt",
    "eng",
    "eng_latn",
]


def find_column(dataframe, candidates):
    column_map = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in column_map:
            return column_map[candidate]

    return None


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    for character in [
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    ]:
        text = text.replace(character, "")

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def load_parallel_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    source_column = find_column(
        dataframe,
        SOURCE_CANDIDATES,
    )

    target_column = find_column(
        dataframe,
        TARGET_CANDIDATES,
    )

    if source_column is None:
        raise ValueError(
            f"No Pashto/source column detected in {path}. "
            f"Columns: {list(dataframe.columns)}"
        )

    if target_column is None:
        raise ValueError(
            f"No English/target column detected in {path}. "
            f"Columns: {list(dataframe.columns)}"
        )

    selected = dataframe[
        [
            source_column,
            target_column,
        ]
    ].copy()

    selected.columns = [
        "pashto",
        "english",
    ]

    selected["pashto"] = (
        selected["pashto"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    selected["english"] = (
        selected["english"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    selected = selected[
        (selected["pashto"] != "")
        & (selected["english"] != "")
    ].copy()

    selected["normalized_pashto"] = (
        selected["pashto"]
        .map(normalize_text)
    )

    selected["normalized_english"] = (
        selected["english"]
        .map(normalize_text)
    )

    selected["pair_key"] = (
        selected["normalized_pashto"]
        + "\u241f"
        + selected["normalized_english"]
    )

    return selected.reset_index(drop=True)


def calculate_overlap(first, second):
    first_pairs = set(first["pair_key"])
    second_pairs = set(second["pair_key"])

    first_sources = set(
        first["normalized_pashto"]
    )

    second_sources = set(
        second["normalized_pashto"]
    )

    first_targets = set(
        first["normalized_english"]
    )

    second_targets = set(
        second["normalized_english"]
    )

    return {
        "exact_pair_overlap": len(
            first_pairs.intersection(
                second_pairs
            )
        ),
        "source_overlap": len(
            first_sources.intersection(
                second_sources
            )
        ),
        "target_overlap": len(
            first_targets.intersection(
                second_targets
            )
        ),
    }


def clean_for_saving(dataframe):
    return (
        dataframe[
            [
                "pashto",
                "english",
            ]
        ]
        .reset_index(drop=True)
    )


def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("CREATING CANONICAL PILOT SPLITS")
    print("=" * 80)

    train_original = load_parallel_file(
        TRAIN_INPUT
    )

    test_original = load_parallel_file(
        TEST_INPUT
    )

    train_original_rows = len(
        train_original
    )

    test_original_rows = len(
        test_original
    )

    train_pool = (
        train_original
        .drop_duplicates(
            subset=["pair_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    test_data = (
        test_original
        .drop_duplicates(
            subset=["pair_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    train_duplicate_pairs_removed = (
        train_original_rows
        - len(train_pool)
    )

    test_duplicate_pairs_removed = (
        test_original_rows
        - len(test_data)
    )

    test_pair_keys = set(
        test_data["pair_key"]
    )

    test_sources = set(
        test_data["normalized_pashto"]
    )

    exact_test_overlap_mask = (
        train_pool["pair_key"]
        .isin(test_pair_keys)
    )

    source_test_overlap_mask = (
        train_pool["normalized_pashto"]
        .isin(test_sources)
    )

    exact_test_overlap_removed = int(
        exact_test_overlap_mask.sum()
    )

    source_test_overlap_removed = int(
        (
            source_test_overlap_mask
            & ~exact_test_overlap_mask
        ).sum()
    )

    train_pool = train_pool[
        ~source_test_overlap_mask
    ].reset_index(drop=True)

    unique_sources = (
        train_pool["normalized_pashto"]
        .drop_duplicates()
        .sample(
            frac=TRAIN_FRACTION,
            random_state=SEED,
        )
        .tolist()
    )

    train_source_set = set(
        unique_sources
    )

    train_data = train_pool[
        train_pool["normalized_pashto"]
        .isin(train_source_set)
    ].copy()

    validation_data = train_pool[
        ~train_pool["normalized_pashto"]
        .isin(train_source_set)
    ].copy()

    train_data = (
        train_data
        .sample(
            frac=1,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    validation_data = (
        validation_data
        .sample(
            frac=1,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    if train_data.empty:
        raise ValueError(
            "Training split is empty after source-group splitting."
        )

    if validation_data.empty:
        raise ValueError(
            "Validation split is empty after source-group splitting."
        )

    comparisons = {
        "train_vs_validation": calculate_overlap(
            train_data,
            validation_data,
        ),
        "train_vs_test": calculate_overlap(
            train_data,
            test_data,
        ),
        "validation_vs_test": calculate_overlap(
            validation_data,
            test_data,
        ),
    }

    for comparison, values in comparisons.items():
        if values["exact_pair_overlap"] != 0:
            raise ValueError(
                f"Exact pair leakage remains in {comparison}"
            )

        if values["source_overlap"] != 0:
            raise ValueError(
                f"Pashto source leakage remains in {comparison}"
            )

    clean_for_saving(
        train_data
    ).to_csv(
        TRAIN_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    clean_for_saving(
        validation_data
    ).to_csv(
        VALIDATION_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    clean_for_saving(
        test_data
    ).to_csv(
        TEST_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = [
        {
            "split": "train",
            "file": str(TRAIN_OUTPUT),
            "rows": len(train_data),
            "unique_pairs": train_data[
                "pair_key"
            ].nunique(),
            "seed": SEED,
        },
        {
            "split": "validation",
            "file": str(VALIDATION_OUTPUT),
            "rows": len(validation_data),
            "unique_pairs": validation_data[
                "pair_key"
            ].nunique(),
            "seed": SEED,
        },
        {
            "split": "test",
            "file": str(TEST_OUTPUT),
            "rows": len(test_data),
            "unique_pairs": test_data[
                "pair_key"
            ].nunique(),
            "seed": "",
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    details = {
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "training_input": str(TRAIN_INPUT),
        "test_input": str(TEST_INPUT),
        "training_input_usable_rows": (
            train_original_rows
        ),
        "test_input_usable_rows": (
            test_original_rows
        ),
        "training_duplicate_pairs_removed": (
            train_duplicate_pairs_removed
        ),
        "test_duplicate_pairs_removed": (
            test_duplicate_pairs_removed
        ),
        "exact_test_pairs_removed_from_training": (
            exact_test_overlap_removed
        ),
        "additional_same_source_rows_removed_from_training": (
            source_test_overlap_removed
        ),
        "final_train_rows": len(train_data),
        "final_validation_rows": len(
            validation_data
        ),
        "final_test_rows": len(test_data),
        "comparisons": comparisons,
    }

    DETAIL_OUTPUT.write_text(
        json.dumps(
            details,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    markdown = [
        "# Canonical Pilot Dataset Splits",
        "",
        "Fixed dataset splits were created for reproducible experiments.",
        "",
        f"- Random seed: `{SEED}`",
        f"- Training proportion: `{TRAIN_FRACTION}`",
        f"- Train rows: `{len(train_data)}`",
        f"- Validation rows: `{len(validation_data)}`",
        f"- Test rows: `{len(test_data)}`",
        f"- Training duplicate pairs removed: `{train_duplicate_pairs_removed}`",
        f"- Test duplicate pairs removed: `{test_duplicate_pairs_removed}`",
        f"- Exact test pairs removed from the training pool: `{exact_test_overlap_removed}`",
        f"- Additional same-source rows removed: `{source_test_overlap_removed}`",
        "",
        "## Leakage verification",
        "",
        "| Comparison | Exact-pair overlap | Pashto-source overlap | English-target overlap |",
        "|---|---:|---:|---:|",
    ]

    for comparison, values in comparisons.items():
        markdown.append(
            f"| {comparison} | "
            f"{values['exact_pair_overlap']} | "
            f"{values['source_overlap']} | "
            f"{values['target_overlap']} |"
        )

    markdown.extend(
        [
            "",
            "Exact-pair and Pashto-source overlaps must remain zero.",
            "English-target overlap is reported but is not automatically treated as leakage because common target sentences can occur independently.",
        ]
    )

    REPORT_OUTPUT.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print()
    print("Canonical splits created successfully.")
    print()
    print(summary.to_string(index=False))

    print()
    print("Leakage verification:")

    for comparison, values in comparisons.items():
        print(
            comparison,
            values,
        )

    print()
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Details: {DETAIL_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
