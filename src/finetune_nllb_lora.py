import argparse
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


PASHTO_COLUMNS = [
    "pashto",
    "ps",
    "source",
    "src",
    "pbt",
    "pbt_arab",
    "input",
]

ENGLISH_COLUMNS = [
    "english",
    "en",
    "target",
    "tgt",
    "eng",
    "eng_latn",
    "reference",
    "english_reference",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible LoRA fine-tuning of NLLB for "
            "Pashto-to-English machine translation."
        )
    )

    parser.add_argument(
        "--train_file",
        default="data/splits/pilot_train.csv",
    )

    parser.add_argument(
        "--validation_file",
        default="data/splits/pilot_validation.csv",
    )

    parser.add_argument(
        "--test_file",
        default="data/splits/pilot_test.csv",
    )

    parser.add_argument(
        "--model_name",
        default="facebook/nllb-200-distilled-600M",
    )

    parser.add_argument(
        "--output_dir",
        default="models/research_nllb_lora_seed42",
    )

    parser.add_argument(
        "--run_name",
        default="canonical_nllb_lora",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_validation_rows",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help=(
            "Validate the fixed splits and save the run manifest "
            "without loading or training the model."
        ),
    )

    return parser.parse_args()


def find_column(dataframe, candidates):
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

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


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_parallel_file(path_string, maximum_rows=0):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    pashto_column = find_column(
        dataframe,
        PASHTO_COLUMNS,
    )

    english_column = find_column(
        dataframe,
        ENGLISH_COLUMNS,
    )

    if pashto_column is None:
        raise ValueError(
            f"No Pashto column detected in {path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    if english_column is None:
        raise ValueError(
            f"No English column detected in {path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe[
        [
            pashto_column,
            english_column,
        ]
    ].copy()

    dataframe.columns = [
        "pashto",
        "english",
    ]

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

    dataframe = dataframe[
        (dataframe["pashto"] != "")
        & (dataframe["english"] != "")
    ].reset_index(drop=True)

    if maximum_rows > 0:
        dataframe = (
            dataframe
            .head(maximum_rows)
            .reset_index(drop=True)
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
        dataframe["pair_key"].duplicated().sum()
    )

    if duplicate_pairs > 0:
        raise ValueError(
            f"{path} contains {duplicate_pairs} duplicate "
            "source-target pairs. Fix the canonical split first."
        )

    information = {
        "path": str(path),
        "absolute_path": str(path.resolve()),
        "rows": len(dataframe),
        "pashto_column": pashto_column,
        "english_column": english_column,
        "unique_pashto_sources": int(
            dataframe["normalized_pashto"].nunique()
        ),
        "unique_english_targets": int(
            dataframe["normalized_english"].nunique()
        ),
        "unique_pairs": int(
            dataframe["pair_key"].nunique()
        ),
        "sha256": calculate_sha256(path),
    }

    return dataframe, information


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
        "pashto_source_overlap": len(
            first_sources.intersection(
                second_sources
            )
        ),
        "english_target_overlap": len(
            first_targets.intersection(
                second_targets
            )
        ),
    }


def verify_splits(
    train_dataframe,
    validation_dataframe,
    test_dataframe,
):
    comparisons = {
        "train_vs_validation": calculate_overlap(
            train_dataframe,
            validation_dataframe,
        ),
        "train_vs_test": calculate_overlap(
            train_dataframe,
            test_dataframe,
        ),
        "validation_vs_test": calculate_overlap(
            validation_dataframe,
            test_dataframe,
        ),
    }

    for comparison_name, values in comparisons.items():
        if values["exact_pair_overlap"] != 0:
            raise ValueError(
                f"Exact-pair leakage detected in "
                f"{comparison_name}: "
                f"{values['exact_pair_overlap']}"
            )

        if values["pashto_source_overlap"] != 0:
            raise ValueError(
                f"Pashto-source leakage detected in "
                f"{comparison_name}: "
                f"{values['pashto_source_overlap']}"
            )

    return comparisons


def configure_reproducibility(seed):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except (AttributeError, TypeError):
        pass


def package_versions():
    packages = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "pandas",
        "numpy",
    ]

    results = {}

    for package in packages:
        try:
            results[package] = version(package)
        except PackageNotFoundError:
            results[package] = "not installed"

    return results


def save_json(path, content):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            content,
            indent=4,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def clean_for_training(dataframe):
    return (
        dataframe[
            [
                "pashto",
                "english",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )


def main():
    args = parse_arguments()

    configure_reproducibility(
        args.seed
    )

    print("=" * 80)
    print("REPRODUCIBLE PASHTO-ENGLISH NLLB LORA TRAINING")
    print("=" * 80)
    print()
    print(f"Run name: {args.run_name}")
    print(f"Seed: {args.seed}")
    print(f"Model: {args.model_name}")
    print()

    train_dataframe, train_information = (
        load_parallel_file(
            args.train_file,
            args.max_train_rows,
        )
    )

    validation_dataframe, validation_information = (
        load_parallel_file(
            args.validation_file,
            args.max_validation_rows,
        )
    )

    test_dataframe, test_information = (
        load_parallel_file(
            args.test_file,
            0,
        )
    )

    comparisons = verify_splits(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

    manifest_path = Path(
        "reports/training_manifests"
    ) / (
        f"{args.run_name}_seed{args.seed}.json"
    )

    manifest = {
        "status": "validated",
        "run_name": args.run_name,
        "seed": args.seed,
        "model_name": args.model_name,
        "arguments": vars(args),
        "datasets": {
            "train": train_information,
            "validation": validation_information,
            "test": test_information,
        },
        "overlap_verification": comparisons,
        "environment": {
            "packages": package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),
        },
    }

    save_json(
        manifest_path,
        manifest,
    )

    print("Dataset sizes:")
    print(
        f"  Train      : {len(train_dataframe)}"
    )
    print(
        f"  Validation : {len(validation_dataframe)}"
    )
    print(
        f"  Test       : {len(test_dataframe)}"
    )

    print()
    print("Leakage verification:")

    for comparison_name, values in comparisons.items():
        print(
            f"  {comparison_name}: {values}"
        )

    print()
    print(
        f"Training manifest saved to: "
        f"{manifest_path}"
    )

    if args.validate_only:
        print()
        print(
            "VALIDATION COMPLETED. "
            "MODEL TRAINING WAS NOT STARTED."
        )
        return

    train_dataset = Dataset.from_pandas(
        clean_for_training(
            train_dataframe
        ),
        preserve_index=False,
    )

    validation_dataset = Dataset.from_pandas(
        clean_for_training(
            validation_dataframe
        ),
        preserve_index=False,
    )

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        src_lang="pbt_Arab",
        tgt_lang="eng_Latn",
    )

    print("Loading NLLB model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name
    )

    english_token_id = (
        tokenizer.convert_tokens_to_ids(
            "eng_Latn"
        )
    )

    model.generation_config.forced_bos_token_id = (
        english_token_id
    )

    lora_configuration = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(
        model,
        lora_configuration,
    )

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    def preprocess(batch):
        return tokenizer(
            batch["pashto"],
            text_target=batch["english"],
            max_length=args.max_length,
            truncation=True,
        )

    print()
    print("Tokenizing training split...")

    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    print("Tokenizing validation split...")

    tokenized_validation = (
        validation_dataset.map(
            preprocess,
            batched=True,
            remove_columns=(
                validation_dataset.column_names
            ),
        )
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_arguments = (
        Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            run_name=(
                f"{args.run_name}_seed{args.seed}"
            ),
            learning_rate=args.lr,
            per_device_train_batch_size=(
                args.batch_size
            ),
            per_device_eval_batch_size=(
                args.batch_size
            ),
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_total_limit=(
                args.save_total_limit
            ),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            predict_with_generate=False,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            dataloader_num_workers=0,
            gradient_checkpointing=(
                args.gradient_checkpointing
            ),
        )
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print()
    print("Starting LoRA fine-tuning...")

    start_time = time.time()

    training_result = trainer.train()

    elapsed_seconds = (
        time.time() - start_time
    )

    trainer.save_model(
        args.output_dir
    )

    tokenizer.save_pretrained(
        args.output_dir
    )

    trainer.save_metrics(
        "train",
        training_result.metrics,
    )

    trainer.save_state()

    manifest["status"] = "training_completed"
    manifest["training"] = {
        "elapsed_seconds": elapsed_seconds,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
        "train_metrics": (
            training_result.metrics
        ),
        "output_directory": args.output_dir,
    }

    save_json(
        manifest_path,
        manifest,
    )

    print()
    print("Fine-tuning completed.")
    print(f"Model saved at: {args.output_dir}")
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
