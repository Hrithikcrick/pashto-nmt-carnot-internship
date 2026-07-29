import hashlib
import json
import platform
import re
import sys
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd


TRAIN_FILE = Path("data/splits/pilot_train.csv")
VALIDATION_FILE = Path("data/splits/pilot_validation.csv")
TEST_FILE = Path("data/splits/pilot_test.csv")

OUTPUT_DIRECTORY = Path("reports/training_manifests")
OUTPUT_FILE = OUTPUT_DIRECTORY / "canonical_nllb_lora_seed42.json"

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SOURCE_LANGUAGE = "pbt_Arab"
TARGET_LANGUAGE = "eng_Latn"
SEED = 42


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


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_split(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required split file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "pashto",
        "english",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe[
        [
            "pashto",
            "english",
        ]
    ].copy()

    dataframe["pashto"] = (
        dataframe["pashto"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["english"] = (
        dataframe["english"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    blank_rows = int(
        (
            (dataframe["pashto"] == "")
            | (dataframe["english"] == "")
        ).sum()
    )

    if blank_rows > 0:
        raise ValueError(
            f"{path} contains {blank_rows} blank rows."
        )

    dataframe["normalized_pashto"] = (
        dataframe["pashto"]
        .map(normalize_text)
    )

    dataframe["normalized_english"] = (
        dataframe["english"]
        .map(normalize_text)
    )

    dataframe["pair_key"] = (
        dataframe["normalized_pashto"]
        + "\u241f"
        + dataframe["normalized_english"]
    )

    duplicate_pairs = int(
        dataframe["pair_key"]
        .duplicated()
        .sum()
    )

    if duplicate_pairs > 0:
        raise ValueError(
            f"{path} contains {duplicate_pairs} "
            "duplicate sentence pairs."
        )

    information = {
        "path": str(path),
        "absolute_path": str(path.resolve()),
        "rows": len(dataframe),
        "unique_pashto_sources": int(
            dataframe[
                "normalized_pashto"
            ].nunique()
        ),
        "unique_english_targets": int(
            dataframe[
                "normalized_english"
            ].nunique()
        ),
        "unique_pairs": int(
            dataframe[
                "pair_key"
            ].nunique()
        ),
        "sha256": calculate_sha256(path),
    }

    return dataframe, information


def compare_splits(first, second):
    exact_pairs = len(
        set(first["pair_key"]).intersection(
            set(second["pair_key"])
        )
    )

    pashto_sources = len(
        set(
            first["normalized_pashto"]
        ).intersection(
            set(
                second["normalized_pashto"]
            )
        )
    )

    english_targets = len(
        set(
            first["normalized_english"]
        ).intersection(
            set(
                second["normalized_english"]
            )
        )
    )

    return {
        "exact_pair_overlap": exact_pairs,
        "pashto_source_overlap": pashto_sources,
        "english_target_overlap": english_targets,
    }


def get_package_versions():
    packages = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "pandas",
        "numpy",
        "sacrebleu",
        "unbabel-comet",
    ]

    results = {}

    for package in packages:
        try:
            results[package] = version(package)
        except PackageNotFoundError:
            results[package] = "not installed"

    return results


def main():
    print("=" * 80)
    print("RESEARCH TRAINING SETUP VALIDATION")
    print("=" * 80)

    train, train_information = load_split(
        TRAIN_FILE
    )

    validation, validation_information = (
        load_split(
            VALIDATION_FILE
        )
    )

    test, test_information = load_split(
        TEST_FILE
    )

    comparisons = {
        "train_vs_validation": compare_splits(
            train,
            validation,
        ),
        "train_vs_test": compare_splits(
            train,
            test,
        ),
        "validation_vs_test": compare_splits(
            validation,
            test,
        ),
    }

    for name, result in comparisons.items():
        if result["exact_pair_overlap"] != 0:
            raise ValueError(
                f"Exact-pair leakage detected in "
                f"{name}: "
                f"{result['exact_pair_overlap']}"
            )

        if result["pashto_source_overlap"] != 0:
            raise ValueError(
                f"Pashto-source leakage detected in "
                f"{name}: "
                f"{result['pashto_source_overlap']}"
            )

    manifest = {
        "status": "validated",
        "experiment_name": (
            "canonical_nllb_lora"
        ),
        "model": {
            "name": MODEL_NAME,
            "source_language": SOURCE_LANGUAGE,
            "target_language": TARGET_LANGUAGE,
        },
        "training_configuration": {
            "seed": SEED,
            "method": "LoRA",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "learning_rate": 0.0002,
            "epochs": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "maximum_sequence_length": 128,
        },
        "datasets": {
            "train": train_information,
            "validation": (
                validation_information
            ),
            "test": test_information,
        },
        "overlap_verification": comparisons,
        "environment": {
            "python_version": sys.version,
            "operating_system": (
                platform.platform()
            ),
            "package_versions": (
                get_package_versions()
            ),
        },
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Dataset sizes:")
    print(f"  Train:      {len(train)}")
    print(
        f"  Validation: {len(validation)}"
    )
    print(f"  Test:       {len(test)}")

    print()
    print("Leakage verification:")

    for name, result in comparisons.items():
        print(f"  {name}")
        print(
            "    Exact pair overlap: "
            f"{result['exact_pair_overlap']}"
        )
        print(
            "    Pashto overlap:     "
            f"{result['pashto_source_overlap']}"
        )
        print(
            "    English overlap:    "
            f"{result['english_target_overlap']}"
        )

    print()
    print(
        "FINAL RESULT: RESEARCH TRAINING "
        "SETUP IS VALID"
    )
    print(
        f"Manifest saved to: {OUTPUT_FILE}"
    )
    print(
        "No model was downloaded or trained."
    )


if __name__ == "__main__":
    main()
