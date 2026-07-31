import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)


MODEL_CONFIGURATIONS = {
    "m2m100": {
        "display_name": "M2M100 418M",
        "model_name": "facebook/m2m100_418M",
        "prediction_column": "m2m100_prediction",
    },
    "mbart50": {
        "display_name": "mBART-50",
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "prediction_column": "mbart50_prediction",
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-shot M2M100 and mBART-50 "
            "for Pashto-to-English translation."
        )
    )

    parser.add_argument(
        "--input_file",
        default="outputs/tables/research_predictions_combined.csv",
    )

    parser.add_argument(
        "--models",
        choices=["m2m100", "mbart50", "all"],
        default="all",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
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
        "--with_comet",
        action="store_true",
    )

    return parser.parse_args()


def find_column(dataframe, candidates):
    lowered = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    return None


def load_evaluation_data(path_string, samples):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation input not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    source_column = find_column(
        dataframe,
        [
            "source",
            "pashto",
            "src",
        ],
    )

    reference_column = find_column(
        dataframe,
        [
            "reference",
            "english",
            "target",
        ],
    )

    if source_column is None:
        raise ValueError(
            f"No source column detected. "
            f"Available columns: {list(dataframe.columns)}"
        )

    if reference_column is None:
        raise ValueError(
            f"No reference column detected. "
            f"Available columns: {list(dataframe.columns)}"
        )

    selected = dataframe[
        [
            source_column,
            reference_column,
        ]
    ].copy()

    selected.columns = [
        "source",
        "reference",
    ]

    selected["source"] = (
        selected["source"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    selected["reference"] = (
        selected["reference"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    selected = selected[
        (selected["source"] != "")
        & (selected["reference"] != "")
    ].reset_index(drop=True)

    if samples > 0:
        selected = (
            selected
            .head(samples)
            .reset_index(drop=True)
        )

    if selected.empty:
        raise ValueError(
            "No usable evaluation rows were found."
        )

    return selected


def load_model_and_tokenizer(model_key, device):
    configuration = MODEL_CONFIGURATIONS[
        model_key
    ]

    model_name = configuration[
        "model_name"
    ]

    if model_key == "m2m100":
        tokenizer = (
            M2M100Tokenizer
            .from_pretrained(model_name)
        )

        tokenizer.src_lang = "ps"

        model = (
            M2M100ForConditionalGeneration
            .from_pretrained(model_name)
        )

        forced_bos_token_id = (
            tokenizer.get_lang_id("en")
        )

    elif model_key == "mbart50":
        tokenizer = (
            MBart50TokenizerFast
            .from_pretrained(
                model_name,
                src_lang="ps_AF",
            )
        )

        tokenizer.src_lang = "ps_AF"

        model = (
            MBartForConditionalGeneration
            .from_pretrained(model_name)
        )

        forced_bos_token_id = (
            tokenizer.convert_tokens_to_ids(
                "en_XX"
            )
        )

        if forced_bos_token_id is None:
            raise ValueError(
                "Could not resolve mBART English language token."
            )

    else:
        raise ValueError(
            f"Unsupported model key: {model_key}"
        )

    model = model.to(device)
    model.eval()

    return (
        tokenizer,
        model,
        forced_bos_token_id,
    )


def generate_predictions(
    dataframe,
    tokenizer,
    model,
    forced_bos_token_id,
    device,
    batch_size,
    num_beams,
    max_source_length,
    max_new_tokens,
):
    predictions = []

    total = len(dataframe)

    start_time = time.time()

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        batch_sources = (
            dataframe["source"]
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
                    forced_bos_token_id
                ),
                num_beams=num_beams,
                max_new_tokens=(
                    max_new_tokens
                ),
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        predictions.extend(
            prediction.strip()
            for prediction in decoded
        )

        print(
            f"Translated {end}/{total}"
        )

    elapsed_seconds = (
        time.time() - start_time
    )

    return predictions, elapsed_seconds


def calculate_automatic_metrics(
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
        "BLEU": float(bleu.score),
        "chrF++": float(chrf.score),
        "TER": float(ter.score),
        "empty_outputs": empty_outputs,
    }


def calculate_comet(
    sources,
    references,
    predictions,
):
    try:
        from comet import (
            download_model,
            load_from_checkpoint,
        )
    except ImportError as error:
        raise ImportError(
            "COMET is not installed in this environment."
        ) from error

    checkpoint_path = download_model(
        "Unbabel/wmt22-comet-da"
    )

    comet_model = load_from_checkpoint(
        checkpoint_path
    )

    evaluation_rows = [
        {
            "src": source,
            "mt": prediction,
            "ref": reference,
        }
        for source, prediction, reference in zip(
            sources,
            predictions,
            references,
        )
    ]

    result = comet_model.predict(
        evaluation_rows,
        batch_size=4,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar=True,
    )

    if hasattr(result, "system_score"):
        score = result.system_score
    elif isinstance(result, tuple):
        score = result[1]
    elif isinstance(result, dict):
        score = result.get(
            "system_score"
        )
    else:
        raise ValueError(
            "Unable to read COMET system score."
        )

    del comet_model
    gc.collect()

    return float(score)


def save_markdown_report(
    metrics_dataframe,
    output_path,
):
    lines = [
        "# External Multilingual Baseline Evaluation",
        "",
        (
            "M2M100 and mBART-50 were evaluated "
            "zero-shot on the same Pashto-English "
            "instances used for NLLB evaluation."
        ),
        "",
        metrics_dataframe.to_markdown(
            index=False
        ),
        "",
        "## Interpretation",
        "",
        (
            "These models are external zero-shot "
            "baselines. They were not fine-tuned on "
            "the Pashto-English training corpus."
        ),
        "",
        (
            "The comparison contextualizes the NLLB "
            "and LoRA results across independent "
            "multilingual translation architectures."
        ),
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    args = parse_arguments()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("EXTERNAL PASHTO-ENGLISH BASELINE EVALUATION")
    print("=" * 80)
    print(f"Device: {device}")

    evaluation_data = load_evaluation_data(
        args.input_file,
        args.samples,
    )

    if args.models == "all":
        selected_models = [
            "m2m100",
            "mbart50",
        ]
    else:
        selected_models = [
            args.models
        ]

    prediction_table = (
        evaluation_data.copy()
    )

    metric_rows = []

    for model_key in selected_models:
        configuration = (
            MODEL_CONFIGURATIONS[
                model_key
            ]
        )

        display_name = configuration[
            "display_name"
        ]

        print()
        print("=" * 80)
        print(f"MODEL: {display_name}")
        print("=" * 80)

        tokenizer, model, forced_token = (
            load_model_and_tokenizer(
                model_key,
                device,
            )
        )

        predictions, elapsed_seconds = (
            generate_predictions(
                dataframe=evaluation_data,
                tokenizer=tokenizer,
                model=model,
                forced_bos_token_id=(
                    forced_token
                ),
                device=device,
                batch_size=args.batch_size,
                num_beams=args.num_beams,
                max_source_length=(
                    args.max_source_length
                ),
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )
        )

        references = (
            evaluation_data[
                "reference"
            ].tolist()
        )

        metrics = (
            calculate_automatic_metrics(
                references,
                predictions,
            )
        )

        comet_score = None

        if args.with_comet:
            print()
            print(
                f"Calculating COMET for "
                f"{display_name}..."
            )

            comet_score = calculate_comet(
                evaluation_data[
                    "source"
                ].tolist(),
                references,
                predictions,
            )

        metrics.update(
            {
                "model": display_name,
                "model_checkpoint": (
                    configuration[
                        "model_name"
                    ]
                ),
                "evaluation_type": (
                    "zero-shot"
                ),
                "COMET": comet_score,
                "device": str(device),
                "num_beams": (
                    args.num_beams
                ),
                "elapsed_seconds": (
                    elapsed_seconds
                ),
                "seconds_per_sentence": (
                    elapsed_seconds
                    / len(predictions)
                ),
            }
        )

        metric_rows.append(metrics)

        prediction_table[
            configuration[
                "prediction_column"
            ]
        ] = predictions

        del model
        del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    output_table_directory = Path(
        "outputs/tables"
    )

    output_prediction_directory = Path(
        "outputs/predictions"
    )

    report_directory = Path(
        "reports"
    )

    output_table_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_prediction_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_table_directory
        / "external_baseline_metrics.csv"
    )

    predictions_path = (
        output_prediction_directory
        / "external_baseline_predictions.csv"
    )

    report_path = (
        report_directory
        / "external_baseline_evaluation.md"
    )

    metrics_dataframe = pd.DataFrame(
        metric_rows
    )

    ordered_columns = [
        "model",
        "model_checkpoint",
        "evaluation_type",
        "samples",
        "BLEU",
        "chrF++",
        "TER",
        "COMET",
        "empty_outputs",
        "device",
        "num_beams",
        "elapsed_seconds",
        "seconds_per_sentence",
    ]

    metrics_dataframe = (
        metrics_dataframe[
            ordered_columns
        ]
    )

    metrics_dataframe.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_table.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_markdown_report(
        metrics_dataframe,
        report_path,
    )

    print()
    print("=" * 80)
    print("EXTERNAL BASELINE EVALUATION COMPLETED")
    print("=" * 80)
    print()
    print(
        metrics_dataframe.to_string(
            index=False
        )
    )
    print()
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
