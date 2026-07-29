import hashlib
import json
from pathlib import Path

import pandas as pd


SEARCH_DIRECTORIES = [
    Path("data"),
    Path("outputs/tables"),
]

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
}

OUTPUT_DIRECTORY = Path("reports")
OUTPUT_CSV = OUTPUT_DIRECTORY / "data_inventory.csv"
OUTPUT_JSON = OUTPUT_DIRECTORY / "data_inventory.json"


def calculate_sha256(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


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


def classify_file(path):
    name = path.name.lower()

    labels = []

    keywords = {
        "train": "training",
        "valid": "validation",
        "validation": "validation",
        "dev": "development",
        "test": "test",
        "clean": "cleaned",
        "semantic": "semantic-filtered",
        "prediction": "predictions",
        "baseline": "baseline",
        "lora": "lora",
        "raw": "raw",
        "filtered": "filtered",
        "remaining": "remaining-experiments",
    }

    for keyword, label in keywords.items():
        if keyword in name and label not in labels:
            labels.append(label)

    if not labels:
        labels.append("unclassified")

    return ", ".join(labels)


def inspect_file(path):
    result = {
        "file": str(path),
        "filename": path.name,
        "category": classify_file(path),
        "extension": path.suffix.lower(),
        "size_mb": round(
            path.stat().st_size / (1024 * 1024),
            4,
        ),
        "rows": None,
        "columns": None,
        "column_names": "",
        "sha256": "",
        "status": "success",
        "error": "",
    }

    try:
        dataframe = read_dataset(path)

        result["rows"] = len(dataframe)
        result["columns"] = len(dataframe.columns)
        result["column_names"] = " | ".join(
            str(column)
            for column in dataframe.columns
        )

    except Exception as error:
        result["status"] = "read_failed"
        result["error"] = str(error)

    try:
        result["sha256"] = calculate_sha256(path)

    except Exception as error:
        result["status"] = "hash_failed"
        result["error"] = str(error)

    return result


def main():
    files = []

    for directory in SEARCH_DIRECTORIES:
        if not directory.exists():
            continue

        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            "No supported dataset files were found inside "
            "data/ or outputs/tables/."
        )

    print("=" * 100)
    print("PASHTO NMT DATASET INVENTORY")
    print("=" * 100)
    print(f"Files found: {len(files)}")
    print()

    results = []

    for index, path in enumerate(files, start=1):
        print(
            f"[{index}/{len(files)}] Inspecting: {path}"
        )

        result = inspect_file(path)
        results.append(result)

        print(
            f"    category: {result['category']}"
        )
        print(
            f"    rows: {result['rows']}"
        )
        print(
            f"    columns: {result['columns']}"
        )
        print(
            f"    status: {result['status']}"
        )
        print()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory = pd.DataFrame(results)

    inventory.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            results,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "file",
        "category",
        "rows",
        "columns",
        "column_names",
        "status",
    ]

    print("=" * 100)
    print("INVENTORY COMPLETED")
    print("=" * 100)
    print()
    print(
        inventory[
            display_columns
        ].to_string(index=False)
    )
    print()
    print(f"CSV report: {OUTPUT_CSV}")
    print(f"JSON report: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
