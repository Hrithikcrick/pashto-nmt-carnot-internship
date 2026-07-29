import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path


SEEDS = [13, 42, 2026]

CONFIG_DIRECTORY = Path("configs/experiments")
REPORT_DIRECTORY = Path("reports")
SUMMARY_FILE = REPORT_DIRECTORY / "seed_experiment_config_summary.csv"

TRAIN_FILE = Path("data/splits/pilot_train.csv")
VALIDATION_FILE = Path("data/splits/pilot_validation.csv")
TEST_FILE = Path("data/splits/pilot_test.csv")


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def verify_required_files():
    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required canonical split files are missing:\n"
            + "\n".join(missing_files)
        )


def build_base_configuration():
    return {
        "study": {
            "name": "extended_a_star_pashto_nmt",
            "stage": "multi_seed_confirmation",
            "status": "planned",
        },
        "model": {
            "name": "facebook/nllb-200-distilled-600M",
            "source_language": "pbt_Arab",
            "target_language": "eng_Latn",
        },
        "dataset": {
            "method": "canonical_pilot",
            "train_file": str(TRAIN_FILE),
            "validation_file": str(VALIDATION_FILE),
            "test_file": str(TEST_FILE),
            "train_sha256": calculate_sha256(TRAIN_FILE),
            "validation_sha256": calculate_sha256(
                VALIDATION_FILE
            ),
            "test_sha256": calculate_sha256(TEST_FILE),
        },
        "training": {
            "epochs": 2.0,
            "learning_rate": 0.0002,
            "batch_size": 2,
            "evaluation_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "maximum_sequence_length": 128,
            "logging_steps": 50,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "mixed_precision": "auto",
            "gradient_checkpointing": False,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "v_proj",
            ],
            "task_type": "SEQ_2_SEQ_LM",
        },
        "generation": {
            "num_beams": 4,
            "maximum_new_tokens": 128,
            "early_stopping": True,
        },
        "evaluation": {
            "metrics": [
                "BLEU",
                "chrF++",
                "TER",
                "COMET",
            ],
            "comet_model": "Unbabel/wmt22-comet-da",
            "bootstrap_samples": 5000,
            "confidence_level": 0.95,
        },
    }


def verify_configurations(configurations):
    if not configurations:
        raise ValueError(
            "No experiment configurations were provided."
        )

    seeds = [
        configuration["reproducibility"]["seed"]
        for configuration in configurations
    ]

    output_directories = [
        configuration["output"]["model_directory"]
        for configuration in configurations
    ]

    experiment_ids = [
        configuration["experiment"]["id"]
        for configuration in configurations
    ]

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Duplicate seeds were detected."
        )

    if len(output_directories) != len(
        set(output_directories)
    ):
        raise ValueError(
            "Duplicate model output directories were detected."
        )

    if len(experiment_ids) != len(
        set(experiment_ids)
    ):
        raise ValueError(
            "Duplicate experiment IDs were detected."
        )

    ignored_sections = {
        "experiment",
        "reproducibility",
        "output",
    }

    def comparable_configuration(configuration):
        return {
            key: deepcopy(value)
            for key, value in configuration.items()
            if key not in ignored_sections
        }

    reference_configuration = (
        comparable_configuration(
            configurations[0]
        )
    )

    for configuration in configurations[1:]:
        current_configuration = (
            comparable_configuration(
                configuration
            )
        )

        if (
            current_configuration
            != reference_configuration
        ):
            raise ValueError(
                "Hyperparameters or datasets differ "
                "across seed configurations."
            )

    print(
        "Verified that all configurations use "
        "identical datasets and hyperparameters."
    )


def main():
    verify_required_files()

    CONFIG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_configuration = build_base_configuration()
    configurations = []
    summary_rows = []

    print("=" * 80)
    print("CREATING MULTI-SEED EXPERIMENT CONFIGURATIONS")
    print("=" * 80)

    for seed in SEEDS:
        experiment_id = (
            f"canonical_nllb_lora_seed{seed}"
        )

        configuration = deepcopy(
            base_configuration
        )

        configuration["experiment"] = {
            "id": experiment_id,
            "description": (
                "NLLB-200 LoRA training on fixed "
                "source-grouped leak-free canonical splits"
            ),
        }

        configuration["reproducibility"] = {
            "seed": seed,
            "data_seed": seed,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }

        configuration["output"] = {
            "model_directory": (
                f"models/research/{experiment_id}"
            ),
            "prediction_file": (
                "outputs/predictions/"
                f"{experiment_id}_predictions.csv"
            ),
            "metrics_file": (
                "outputs/metrics/"
                f"{experiment_id}_metrics.json"
            ),
            "training_manifest": (
                "reports/training_manifests/"
                f"{experiment_id}.json"
            ),
            "training_log": (
                "logs/"
                f"{experiment_id}.log"
            ),
        }

        config_path = (
            CONFIG_DIRECTORY
            / f"{experiment_id}.json"
        )

        config_path.write_text(
            json.dumps(
                configuration,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        configurations.append(
            configuration
        )

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "seed": seed,
                "model": configuration[
                    "model"
                ]["name"],
                "dataset_method": configuration[
                    "dataset"
                ]["method"],
                "train_file": configuration[
                    "dataset"
                ]["train_file"],
                "validation_file": configuration[
                    "dataset"
                ]["validation_file"],
                "test_file": configuration[
                    "dataset"
                ]["test_file"],
                "epochs": configuration[
                    "training"
                ]["epochs"],
                "learning_rate": configuration[
                    "training"
                ]["learning_rate"],
                "batch_size": configuration[
                    "training"
                ]["batch_size"],
                "gradient_accumulation_steps": configuration[
                    "training"
                ][
                    "gradient_accumulation_steps"
                ],
                "lora_rank": configuration[
                    "lora"
                ]["rank"],
                "lora_alpha": configuration[
                    "lora"
                ]["alpha"],
                "lora_dropout": configuration[
                    "lora"
                ]["dropout"],
                "model_output": configuration[
                    "output"
                ]["model_directory"],
                "config_file": str(config_path),
                "status": "planned",
            }
        )

        print(
            f"Created: {config_path}"
        )

    verify_configurations(
        configurations
    )

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 80)
    print("CONFIGURATION VALIDATION PASSED")
    print("=" * 80)
    print(f"Experiments created: {len(configurations)}")
    print(f"Seeds: {SEEDS}")
    print(f"Summary: {SUMMARY_FILE}")
    print()
    print("No model was downloaded or trained.")


if __name__ == "__main__":
    main()
