import argparse
import json
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reload a trained NLLB LoRA adapter and run "
            "a small Pashto-to-English inference check."
        )
    )

    parser.add_argument(
        "--adapter_dir",
        default="models/research/smoke_nllb_lora_seed42",
    )

    parser.add_argument(
        "--base_model",
        default="facebook/nllb-200-distilled-600M",
    )

    parser.add_argument(
        "--test_file",
        default="data/splits/pilot_test.csv",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )

    return parser.parse_args()


def load_test_data(path_string, samples):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"Test file not found: {path}"
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
            f"Missing test columns: "
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

    dataframe = dataframe[
        (dataframe["pashto"] != "")
        & (dataframe["english"] != "")
    ].reset_index(drop=True)

    if samples > 0:
        dataframe = (
            dataframe
            .head(samples)
            .reset_index(drop=True)
        )

    if dataframe.empty:
        raise ValueError(
            "No usable test rows were found."
        )

    return dataframe


def generate_predictions(
    dataframe,
    tokenizer,
    model,
    device,
    batch_size,
    max_source_length,
    max_new_tokens,
    num_beams,
):
    predictions = []

    total_rows = len(dataframe)

    for start in range(
        0,
        total_rows,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total_rows,
        )

        batch_sources = (
            dataframe[
                "pashto"
            ]
            .iloc[start:end]
            .tolist()
        )

        encoded = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=(
                    tokenizer.convert_tokens_to_ids(
                        "eng_Latn"
                    )
                ),
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        predictions.extend(
            prediction.strip()
            for prediction in decoded
        )

        print(
            f"Translated {end}/{total_rows}"
        )

    return predictions


def calculate_metrics(
    references,
    predictions,
):
    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
        tokenize="13a",
    )

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
        word_order=2,
    )

    ter = sacrebleu.corpus_ter(
        predictions,
        [references],
    )

    empty_outputs = sum(
        1
        for prediction in predictions
        if not prediction.strip()
    )

    return {
        "samples": len(predictions),
        "BLEU": round(
            float(bleu.score),
            6,
        ),
        "chrF++": round(
            float(chrf.score),
            6,
        ),
        "TER": round(
            float(ter.score),
            6,
        ),
        "empty_outputs": empty_outputs,
    }


def main():
    args = parse_arguments()

    adapter_directory = Path(
        args.adapter_dir
    )

    if not adapter_directory.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: "
            f"{adapter_directory}"
        )

    adapter_file = (
        adapter_directory
        / "adapter_model.safetensors"
    )

    if not adapter_file.exists():
        raise FileNotFoundError(
            f"LoRA adapter file not found: "
            f"{adapter_file}"
        )

    print("=" * 80)
    print("SMOKE LORA INFERENCE CHECK")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {adapter_directory}")

    dataframe = load_test_data(
        args.test_file,
        args.samples,
    )

    print(
        f"Test samples: {len(dataframe)}"
    )

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_directory,
        src_lang="pbt_Arab",
        tgt_lang="eng_Latn",
    )

    print("Loading base NLLB model...")

    base_model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            args.base_model
        )
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_directory,
    )

    model = model.to(device)
    model.eval()

    start_time = time.time()

    predictions = generate_predictions(
        dataframe=dataframe,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_source_length=(
            args.max_source_length
        ),
        max_new_tokens=(
            args.max_new_tokens
        ),
        num_beams=args.num_beams,
    )

    elapsed_seconds = (
        time.time() - start_time
    )

    dataframe["prediction"] = predictions

    references = (
        dataframe["english"]
        .astype(str)
        .tolist()
    )

    metrics = calculate_metrics(
        references,
        predictions,
    )

    metrics.update(
        {
            "status": (
                "inference_completed"
            ),
            "purpose": (
                "pipeline_smoke_check"
            ),
            "base_model": args.base_model,
            "adapter_directory": (
                str(adapter_directory)
            ),
            "device": str(device),
            "batch_size": args.batch_size,
            "num_beams": args.num_beams,
            "elapsed_seconds": round(
                elapsed_seconds,
                4,
            ),
            "seconds_per_sentence": round(
                elapsed_seconds
                / len(dataframe),
                4,
            ),
        }
    )

    prediction_directory = Path(
        "outputs/predictions"
    )

    metric_directory = Path(
        "outputs/metrics"
    )

    report_directory = Path(
        "reports"
    )

    prediction_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        prediction_directory
        / (
            "smoke_nllb_lora_seed42_"
            "predictions.csv"
        )
    )

    metrics_path = (
        metric_directory
        / (
            "smoke_nllb_lora_seed42_"
            "metrics.json"
        )
    )

    report_path = (
        report_directory
        / (
            "smoke_nllb_lora_seed42_"
            "report.md"
        )
    )

    dataframe.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# NLLB LoRA Smoke-Test Report",
        "",
        "## Purpose",
        "",
        (
            "This experiment verifies that the saved "
            "LoRA adapter can be reloaded and used for "
            "Pashto-to-English generation."
        ),
        "",
        "It is not a final translation-quality experiment.",
        "",
        "## Training check",
        "",
        "- Training rows: 100",
        "- Validation rows: 20",
        "- Training duration: approximately 34 seconds",
        "- Epoch fraction: 0.1",
        "- Device: CPU",
        "",
        "## Inference check",
        "",
        f"- Test samples: {metrics['samples']}",
        f"- BLEU: {metrics['BLEU']}",
        f"- chrF++: {metrics['chrF++']}",
        f"- TER: {metrics['TER']}",
        f"- Empty outputs: {metrics['empty_outputs']}",
        (
            f"- Inference duration: "
            f"{metrics['elapsed_seconds']} seconds"
        ),
        "",
        "## Interpretation",
        "",
        (
            "These scores must not be used as final "
            "research results because the adapter was "
            "trained on only 100 rows for 0.1 epoch."
        ),
        "",
        (
            "A successful run confirms the training, "
            "adapter-saving, adapter-loading, generation, "
            "and metric-computation pipeline."
        ),
    ]

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SMOKE INFERENCE COMPLETED")
    print("=" * 80)

    print()
    print(json.dumps(
        metrics,
        indent=4,
        ensure_ascii=False,
    ))

    print()
    print(
        f"Predictions: {prediction_path}"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Report: {report_path}"
    )

    print()
    print(
        "FINAL RESULT: TRAINING AND "
        "INFERENCE PIPELINE WORKS"
    )


if __name__ == "__main__":
    main()
