import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Launch a reproducible Pashto-English "
            "NLLB LoRA experiment from a JSON configuration."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment JSON configuration.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate the configuration and print the "
            "training command without starting training."
        ),
    )

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help=(
            "Run the training script in validation-only mode."
        ),
    )

    return parser.parse_args()


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_configuration(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    configuration = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    required_sections = [
        "experiment",
        "model",
        "dataset",
        "training",
        "lora",
        "reproducibility",
        "output",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in configuration
    ]

    if missing_sections:
        raise ValueError(
            "Configuration is missing sections: "
            + ", ".join(missing_sections)
        )

    return configuration


def verify_dataset(
    label,
    path_string,
    expected_hash,
):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} dataset does not exist: {path}"
        )

    actual_hash = calculate_sha256(path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} dataset hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            f"File:     {path}"
        )

    return {
        "path": str(path),
        "sha256": actual_hash,
        "size_bytes": path.stat().st_size,
        "status": "verified",
    }


def build_training_command(configuration):
    training_script = Path(
        "src/finetune_nllb_lora.py"
    )

    if not training_script.exists():
        raise FileNotFoundError(
            f"Training script not found: {training_script}"
        )

    experiment_id = configuration[
        "experiment"
    ]["id"]

    model = configuration["model"]
    dataset = configuration["dataset"]
    training = configuration["training"]
    lora = configuration["lora"]
    reproducibility = configuration[
        "reproducibility"
    ]
    output = configuration["output"]

    command = [
        sys.executable,
        str(training_script),
        "--train_file",
        dataset["train_file"],
        "--validation_file",
        dataset["validation_file"],
        "--test_file",
        dataset["test_file"],
        "--model_name",
        model["name"],
        "--output_dir",
        output["model_directory"],
        "--run_name",
        experiment_id,
        "--seed",
        str(reproducibility["seed"]),
        "--epochs",
        str(training["epochs"]),
        "--batch_size",
        str(training["batch_size"]),
        "--gradient_accumulation_steps",
        str(
            training[
                "gradient_accumulation_steps"
            ]
        ),
        "--lr",
        str(training["learning_rate"]),
        "--max_length",
        str(
            training[
                "maximum_sequence_length"
            ]
        ),
        "--lora_r",
        str(lora["rank"]),
        "--lora_alpha",
        str(lora["alpha"]),
        "--lora_dropout",
        str(lora["dropout"]),
        "--logging_steps",
        str(training["logging_steps"]),
        "--save_total_limit",
        "2",
    ]

    if training.get(
        "gradient_checkpointing",
        False,
    ):
        command.append(
            "--gradient_checkpointing"
        )

    return command


def display_command(command):
    print()

    print("Training command:")

    print(
        " ".join(
            shlex.quote(part)
            for part in command
        )
    )


def save_run_record(
    configuration_path,
    configuration,
    dataset_verification,
    command,
    mode,
):
    experiment_id = configuration[
        "experiment"
    ]["id"]

    output_directory = Path(
        "reports/experiment_launches"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{experiment_id}_{mode}.json"
    )

    record = {
        "experiment_id": experiment_id,
        "mode": mode,
        "status": "validated",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration_file": str(
            configuration_path
        ),
        "seed": configuration[
            "reproducibility"
        ]["seed"],
        "dataset_verification": (
            dataset_verification
        ),
        "command": command,
    }

    output_path.write_text(
        json.dumps(
            record,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


def main():
    args = parse_arguments()

    configuration_path = Path(
        args.config
    )

    configuration = load_configuration(
        configuration_path
    )

    experiment_id = configuration[
        "experiment"
    ]["id"]

    seed = configuration[
        "reproducibility"
    ]["seed"]

    dataset = configuration["dataset"]

    print("=" * 80)
    print("PASHTO NMT EXPERIMENT LAUNCHER")
    print("=" * 80)

    print()
    print(f"Experiment: {experiment_id}")
    print(f"Seed:       {seed}")
    print(
        f"Model:      "
        f"{configuration['model']['name']}"
    )
    print(f"Config:     {configuration_path}")

    print()
    print("Verifying fixed datasets...")

    dataset_verification = {
        "train": verify_dataset(
            "Training",
            dataset["train_file"],
            dataset["train_sha256"],
        ),
        "validation": verify_dataset(
            "Validation",
            dataset[
                "validation_file"
            ],
            dataset[
                "validation_sha256"
            ],
        ),
        "test": verify_dataset(
            "Test",
            dataset["test_file"],
            dataset["test_sha256"],
        ),
    }

    for split, result in (
        dataset_verification.items()
    ):
        print(
            f"  {split}: verified"
        )
        print(
            f"    file: {result['path']}"
        )
        print(
            f"    hash: {result['sha256']}"
        )

    command = build_training_command(
        configuration
    )

    if args.validate_only:
        command.append(
            "--validate_only"
        )

    display_command(command)

    if args.dry_run:
        mode = "dry_run"

        record_path = save_run_record(
            configuration_path,
            configuration,
            dataset_verification,
            command,
            mode,
        )

        print()
        print("=" * 80)
        print("DRY RUN COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print(
            "Configuration, files, hashes, "
            "and command were verified."
        )

        print(
            "Model training was not started."
        )

        print(
            f"Dry-run record: {record_path}"
        )

        return

    mode = (
        "validation_only"
        if args.validate_only
        else "training"
    )

    record_path = save_run_record(
        configuration_path,
        configuration,
        dataset_verification,
        command,
        mode,
    )

    print()
    print(
        f"Launch record saved to: "
        f"{record_path}"
    )

    print()
    print(
        "Starting subprocess..."
    )

    completed_process = subprocess.run(
        command,
        check=False,
    )

    if completed_process.returncode != 0:
        raise SystemExit(
            completed_process.returncode
        )

    print()
    print(
        "Experiment command completed successfully."
    )


if __name__ == "__main__":
    main()
