import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


SEARCH_DIRECTORIES = [
    Path("data"),
    Path("outputs/tables"),
]

REPORT_DIRECTORY = Path("reports")

DATASET_REPORT = (
    REPORT_DIRECTORY
    / "leakage_audit_dataset_files.csv"
)

SUMMARY_REPORT = (
    REPORT_DIRECTORY
    / "leakage_audit_summary.csv"
)

EXAMPLES_REPORT = (
    REPORT_DIRECTORY
    / "leakage_overlap_examples.csv"
)

MARKDOWN_REPORT = (
    REPORT_DIRECTORY
    / "leakage_audit_report.md"
)

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
}

SOURCE_COLUMN_CANDIDATES = [
    "pashto",
    "pashto_source",
    "source_pashto",
    "source",
    "src",
    "source_text",
    "source_sentence",
    "pashto_text",
    "pbt",
    "pbt_arab",
    "sentence_pbt",
    "translation_pbt",
]

TARGET_COLUMN_CANDIDATES = [
    "english",
    "english_reference",
    "reference_english",
    "reference",
    "target",
    "tgt",
    "target_text",
    "target_sentence",
    "english_text",
    "eng",
    "eng_latn",
    "sentence_eng",
    "translation_eng",
]

ZERO_WIDTH_CHARACTERS = [
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
]


def normalize_column_name(name):
    name = str(name).strip().lower()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name,
    )

    return name.strip("_")


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    for character in ZERO_WIDTH_CHARACTERS:
        text = text.replace(character, "")

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    text = re.sub(
        r"^[\W_]+|[\W_]+$",
        "",
        text,
        flags=re.UNICODE,
    )

    return text


def read_dataset(path):
    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
        )

    if extension == ".tsv":
        return pd.read_csv(
            path,
            sep="\t",
            encoding="utf-8-sig",
            low_memory=False,
        )

    if extension == ".jsonl":
        return pd.read_json(
            path,
            lines=True,
        )

    if extension == ".json":
        return pd.read_json(path)

    if extension == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported extension: {extension}"
    )


def find_column(columns, candidates, kind):
    normalized_columns = {
        normalize_column_name(column): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    for normalized_name, original_name in (
        normalized_columns.items()
    ):
        if "prediction" in normalized_name:
            continue

        if kind == "source":
            keywords = [
                "pashto",
                "pbt",
                "source",
                "src",
            ]

        else:
            keywords = [
                "english",
                "eng",
                "reference",
                "target",
                "tgt",
            ]

        if any(
            keyword in normalized_name
            for keyword in keywords
        ):
            return original_name

    return None


def classify_split(path):
    name = path.name.lower()
    full_path = str(path).lower()

    if (
        "devtest" in name
        or "test" in name
        or "gold" in name
        or "evaluation" in name
        or "prediction" in name
    ):
        return "test"

    if (
        "validation" in name
        or "valid" in name
        or re.search(
            r"(^|[_\-.])val([_\-.]|$)",
            name,
        )
        or re.search(
            r"(^|[_\-.])dev([_\-.]|$)",
            name,
        )
    ):
        return "validation"

    if (
        "semantic" in name
        or "high_quality" in name
        or "high-quality" in name
        or "filtered_train" in name
        or "training_subset" in name
    ):
        return "derived_train"

    if (
        "train" in name
        or "training" in name
    ):
        return "train"

    if (
        "raw" in name
        or "cleaned" in name
        or "clean_dataset" in name
        or "full_dataset" in name
        or "corpus" in name
    ):
        return "pool"

    if "outputs/tables" in full_path:
        return "unknown_output"

    return "unknown"


def load_parallel_file(path):
    dataframe = read_dataset(path)

    source_column = find_column(
        dataframe.columns,
        SOURCE_COLUMN_CANDIDATES,
        "source",
    )

    target_column = find_column(
        dataframe.columns,
        TARGET_COLUMN_CANDIDATES,
        "target",
    )

    if source_column is None:
        raise ValueError(
            "No Pashto/source column detected."
        )

    if target_column is None:
        raise ValueError(
            "No English/reference column detected."
        )

    selected = dataframe[
        [
            source_column,
            target_column,
        ]
    ].copy()

    selected.columns = [
        "source",
        "target",
    ]

    selected["original_row"] = (
        selected.index + 1
    )

    selected["source"] = (
        selected["source"]
        .fillna("")
        .astype(str)
    )

    selected["target"] = (
        selected["target"]
        .fillna("")
        .astype(str)
    )

    selected["normalized_source"] = (
        selected["source"]
        .map(normalize_text)
    )

    selected["normalized_target"] = (
        selected["target"]
        .map(normalize_text)
    )

    selected = selected[
        (
            selected["normalized_source"]
            != ""
        )
        &
        (
            selected["normalized_target"]
            != ""
        )
    ].copy()

    selected["pair_key"] = (
        selected["normalized_source"]
        + "\u241f"
        + selected["normalized_target"]
    )

    return (
        selected,
        source_column,
        target_column,
        len(dataframe),
    )


def collect_files():
    files = []

    for directory in SEARCH_DIRECTORIES:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                files.append(path)

    return sorted(set(files))


def build_group(records, group_name):
    group_parts = []

    for dataframe in records:
        if dataframe is None:
            continue

        if dataframe.empty:
            continue

        if "audit_group" not in dataframe.columns:
            continue

        available_groups = (
            dataframe["audit_group"]
            .dropna()
            .astype(str)
        )

        if available_groups.empty:
            continue

        if available_groups.iloc[0] == group_name:
            group_parts.append(dataframe)

    if not group_parts:
        return pd.DataFrame()

    return pd.concat(
        group_parts,
        ignore_index=True,
    )


def create_overlap_examples(
    group_a,
    group_b,
    comparison_name,
    key_column,
    overlap_type,
    maximum_examples=100,
):
    if group_a.empty or group_b.empty:
        return []

    keys_a = set(group_a[key_column])
    keys_b = set(group_b[key_column])

    common_keys = keys_a.intersection(
        keys_b
    )

    examples = []

    for key in list(common_keys)[
        :maximum_examples
    ]:
        row_a = group_a[
            group_a[key_column] == key
        ].iloc[0]

        row_b = group_b[
            group_b[key_column] == key
        ].iloc[0]

        examples.append(
            {
                "comparison": comparison_name,
                "overlap_type": overlap_type,
                "file_a": row_a["file"],
                "row_a": row_a["original_row"],
                "file_b": row_b["file"],
                "row_b": row_b["original_row"],
                "source_a": row_a["source"],
                "target_a": row_a["target"],
                "source_b": row_b["source"],
                "target_b": row_b["target"],
            }
        )

    return examples


def compare_groups(
    group_a,
    group_b,
    name_a,
    name_b,
):
    comparison = (
        f"{name_a}_vs_{name_b}"
    )

    if group_a.empty or group_b.empty:
        return {
            "comparison": comparison,
            "rows_a": len(group_a),
            "rows_b": len(group_b),
            "unique_pairs_a": 0,
            "unique_pairs_b": 0,
            "exact_pair_overlap": 0,
            "source_overlap": 0,
            "target_overlap": 0,
            "status": "missing_split",
        }

    pair_overlap = len(
        set(group_a["pair_key"])
        .intersection(
            set(group_b["pair_key"])
        )
    )

    source_overlap = len(
        set(group_a["normalized_source"])
        .intersection(
            set(
                group_b[
                    "normalized_source"
                ]
            )
        )
    )

    target_overlap = len(
        set(group_a["normalized_target"])
        .intersection(
            set(
                group_b[
                    "normalized_target"
                ]
            )
        )
    )

    if pair_overlap == 0:
        status = "PASS"
    else:
        status = "LEAKAGE_FOUND"

    return {
        "comparison": comparison,
        "rows_a": len(group_a),
        "rows_b": len(group_b),
        "unique_pairs_a": (
            group_a["pair_key"].nunique()
        ),
        "unique_pairs_b": (
            group_b["pair_key"].nunique()
        ),
        "exact_pair_overlap": pair_overlap,
        "source_overlap": source_overlap,
        "target_overlap": target_overlap,
        "status": status,
    }


def main():
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = collect_files()

    if not files:
        raise FileNotFoundError(
            "No dataset files were found."
        )

    dataset_rows = []
    record_frames = []

    print("=" * 100)
    print("TRAIN–VALIDATION–TEST LEAKAGE AUDIT")
    print("=" * 100)
    print()

    for index, path in enumerate(
        files,
        start=1,
    ):
        split = classify_split(path)

        print(
            f"[{index}/{len(files)}] {path}"
        )

        try:
            (
                dataframe,
                source_column,
                target_column,
                original_rows,
            ) = load_parallel_file(path)

            if split in {
                "train",
                "derived_train",
            }:
                audit_group = "train"

            elif split == "validation":
                audit_group = "validation"

            elif split == "test":
                audit_group = "test"

            else:
                audit_group = split

            dataframe["file"] = str(path)
            dataframe["detected_split"] = split
            dataframe["audit_group"] = (
                audit_group
            )

            record_frames.append(dataframe)

            duplicate_pairs = (
                len(dataframe)
                - dataframe[
                    "pair_key"
                ].nunique()
            )

            dataset_rows.append(
                {
                    "file": str(path),
                    "detected_split": split,
                    "audit_group": audit_group,
                    "original_rows": (
                        original_rows
                    ),
                    "usable_parallel_rows": (
                        len(dataframe)
                    ),
                    "unique_pairs": (
                        dataframe[
                            "pair_key"
                        ].nunique()
                    ),
                    "duplicate_pairs": (
                        duplicate_pairs
                    ),
                    "source_column": (
                        source_column
                    ),
                    "target_column": (
                        target_column
                    ),
                    "status": "included",
                    "error": "",
                }
            )

            print(
                f"    included: {split}, "
                f"rows={len(dataframe)}, "
                f"duplicates={duplicate_pairs}"
            )

        except Exception as error:
            dataset_rows.append(
                {
                    "file": str(path),
                    "detected_split": split,
                    "audit_group": "",
                    "original_rows": "",
                    "usable_parallel_rows": "",
                    "unique_pairs": "",
                    "duplicate_pairs": "",
                    "source_column": "",
                    "target_column": "",
                    "status": "skipped",
                    "error": str(error),
                }
            )

            print(
                f"    skipped: {error}"
            )

    if not record_frames:
        raise ValueError(
            "No usable Pashto–English parallel "
            "files were detected."
        )

    train_group = build_group(
        record_frames,
        "train",
    )

    validation_group = build_group(
        record_frames,
        "validation",
    )

    test_group = build_group(
        record_frames,
        "test",
    )

    comparisons = [
        (
            train_group,
            validation_group,
            "train",
            "validation",
        ),
        (
            train_group,
            test_group,
            "train",
            "test",
        ),
        (
            validation_group,
            test_group,
            "validation",
            "test",
        ),
    ]

    summary_rows = []
    example_rows = []

    for (
        group_a,
        group_b,
        name_a,
        name_b,
    ) in comparisons:
        summary = compare_groups(
            group_a,
            group_b,
            name_a,
            name_b,
        )

        summary_rows.append(summary)

        comparison_name = (
            f"{name_a}_vs_{name_b}"
        )

        example_rows.extend(
            create_overlap_examples(
                group_a,
                group_b,
                comparison_name,
                "pair_key",
                "exact_pair",
            )
        )

        example_rows.extend(
            create_overlap_examples(
                group_a,
                group_b,
                comparison_name,
                "normalized_source",
                "same_source",
            )
        )

        example_rows.extend(
            create_overlap_examples(
                group_a,
                group_b,
                comparison_name,
                "normalized_target",
                "same_target",
            )
        )

    dataset_report = pd.DataFrame(
        dataset_rows
    )

    summary_report = pd.DataFrame(
        summary_rows
    )

    examples_report = pd.DataFrame(
        example_rows
    )

    dataset_report.to_csv(
        DATASET_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_report.to_csv(
        SUMMARY_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    examples_report.to_csv(
        EXAMPLES_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    critical_comparisons = (
        summary_report[
            summary_report["comparison"].isin(
                [
                    "train_vs_validation",
                    "train_vs_test",
                ]
            )
        ]
    )

    leakage_found = (
        critical_comparisons[
            "exact_pair_overlap"
        ]
        .fillna(0)
        .astype(int)
        .sum()
        > 0
    )

    if leakage_found:
        final_status = "LEAKAGE FOUND"
    else:
        final_status = "PASS"

    markdown = []

    markdown.append(
        "# Train–Validation–Test Leakage Audit"
    )

    markdown.append("")
    markdown.append(
        f"**Final status: {final_status}**"
    )

    markdown.append("")
    markdown.append(
        "Exact source–target pair overlap is treated "
        "as confirmed leakage. Source-only or target-only "
        "overlap is reported for further inspection."
    )

    markdown.append("")
    markdown.append("## Dataset files")
    markdown.append("")
    markdown.append(
        dataset_report.to_markdown(
            index=False
        )
    )

    markdown.append("")
    markdown.append("## Cross-split overlap")
    markdown.append("")
    markdown.append(
        summary_report.to_markdown(
            index=False
        )
    )

    markdown.append("")
    markdown.append("## Interpretation")
    markdown.append("")

    if leakage_found:
        markdown.append(
            "Exact training examples were found in "
            "validation or test data. New leak-free "
            "splits must be created before further "
            "fine-tuning or final evaluation."
        )
    else:
        markdown.append(
            "No exact source–target pair leakage was "
            "detected between training and evaluation "
            "splits."
        )

    markdown.append("")
    markdown.append(
        "Master cleaned pools are not treated as training "
        "splits because they may legitimately contain rows "
        "later assigned to train, validation and test."
    )

    MARKDOWN_REPORT.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("LEAKAGE AUDIT COMPLETED")
    print("=" * 100)
    print()
    print(summary_report.to_string(index=False))
    print()
    print(f"Final status: {final_status}")
    print(f"Dataset report: {DATASET_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")
    print(f"Examples: {EXAMPLES_REPORT}")
    print(f"Markdown report: {MARKDOWN_REPORT}")


if __name__ == "__main__":
    main()
