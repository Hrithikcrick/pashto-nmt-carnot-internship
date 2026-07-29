import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacrebleu
from comet import download_model, load_from_checkpoint


DEFAULT_INPUT = Path(
    "outputs/tables/research_predictions_combined.csv"
)

TABLE_DIRECTORY = Path("outputs/tables")
FIGURE_DIRECTORY = Path("outputs/figures")
REPORT_DIRECTORY = Path("reports")

SUMMARY_PATH = TABLE_DIRECTORY / "research_automatic_metrics.csv"
SENTENCE_PATH = TABLE_DIRECTORY / "research_sentence_scores.csv"
SIGNIFICANCE_PATH = TABLE_DIRECTORY / "research_metric_significance.csv"
DIAGNOSTICS_PATH = TABLE_DIRECTORY / "research_automatic_diagnostics.csv"
PAIRWISE_PATH = TABLE_DIRECTORY / "research_pairwise_wins.csv"

FIGURE_PATH = FIGURE_DIRECTORY / "research_neural_metric_comparison.png"
REPORT_PATH = REPORT_DIRECTORY / "research_automatic_evaluation.md"


MODEL_COLUMNS = {
    "Baseline NLLB": "baseline_prediction",
    "Original LoRA": "original_lora_prediction",
    "Semantic LoRA": "semantic_lora_prediction",
}


def clean_text(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def validate_input(dataframe):
    required_columns = [
        "source",
        "reference",
        *MODEL_COLUMNS.values(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Input file is missing columns: {missing_columns}"
        )


def corpus_metrics(references, predictions):
    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
        tokenize="13a",
    ).score

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
        word_order=2,
    ).score

    ter = sacrebleu.corpus_ter(
        predictions,
        [references],
    ).score

    return {
        "BLEU": float(bleu),
        "chrF++": float(chrf),
        "TER": float(ter),
    }


def sentence_metrics(reference, prediction):
    bleu_metric = sacrebleu.metrics.BLEU(
        tokenize="13a",
        effective_order=True,
    )

    chrf_metric = sacrebleu.metrics.CHRF(
        word_order=2,
    )

    ter_metric = sacrebleu.metrics.TER()

    bleu = bleu_metric.sentence_score(
        prediction,
        [reference],
    ).score

    chrf = chrf_metric.sentence_score(
        prediction,
        [reference],
    ).score

    ter = ter_metric.sentence_score(
        prediction,
        [reference],
    ).score

    return bleu, chrf, ter


def run_comet_prediction(model, data, batch_size):
    try:
        result = model.predict(
            data,
            batch_size=batch_size,
            gpus=0,
            progress_bar=True,
        )

    except TypeError:
        result = model.predict(
            data,
            batch_size=batch_size,
            accelerator="cpu",
            devices=1,
        )

    scores = [
        float(score)
        for score in result.scores
    ]

    system_score = float(result.system_score)

    return scores, system_score


def evaluate_reference_comet(
    dataframe,
    model_columns,
    batch_size,
):
    print()
    print("Loading reference-based COMET model...")

    model_path = download_model(
        "Unbabel/wmt22-comet-da"
    )

    model = load_from_checkpoint(model_path)

    sentence_scores = {}
    system_scores = {}

    for model_name, column in model_columns.items():
        print()
        print(f"COMET evaluation: {model_name}")

        data = []

        for _, row in dataframe.iterrows():
            data.append(
                {
                    "src": clean_text(row["source"]),
                    "mt": clean_text(row[column]),
                    "ref": clean_text(row["reference"]),
                }
            )

        scores, system_score = run_comet_prediction(
            model,
            data,
            batch_size,
        )

        sentence_scores[model_name] = scores
        system_scores[model_name] = system_score

        print(
            f"{model_name} COMET: "
            f"{system_score:.6f}"
        )

    del model

    return sentence_scores, system_scores


def evaluate_cometkiwi(
    dataframe,
    model_columns,
    batch_size,
):
    print()
    print("Attempting reference-free COMETKiwi evaluation...")

    sentence_scores = {}
    system_scores = {}

    try:
        model_path = download_model(
            "Unbabel/wmt22-cometkiwi-da"
        )

        model = load_from_checkpoint(model_path)

    except Exception as error:
        print()
        print("COMETKiwi could not be loaded.")
        print("The remaining evaluation will continue normally.")
        print(f"Reason: {error}")

        return sentence_scores, system_scores

    for model_name, column in model_columns.items():
        print()
        print(f"COMETKiwi evaluation: {model_name}")

        data = []

        for _, row in dataframe.iterrows():
            data.append(
                {
                    "src": clean_text(row["source"]),
                    "mt": clean_text(row[column]),
                }
            )

        try:
            scores, system_score = run_comet_prediction(
                model,
                data,
                batch_size,
            )

            sentence_scores[model_name] = scores
            system_scores[model_name] = system_score

            print(
                f"{model_name} COMETKiwi: "
                f"{system_score:.6f}"
            )

        except Exception as error:
            print(
                f"COMETKiwi failed for {model_name}: "
                f"{error}"
            )

    del model

    return sentence_scores, system_scores


def bootstrap_mean_difference(
    scores_a,
    scores_b,
    samples=5000,
    seed=42,
):
    scores_a = np.asarray(
        scores_a,
        dtype=float,
    )

    scores_b = np.asarray(
        scores_b,
        dtype=float,
    )

    if len(scores_a) != len(scores_b):
        raise ValueError(
            "Paired score lists must have equal lengths."
        )

    random_generator = np.random.default_rng(seed)
    size = len(scores_a)

    observed_difference = float(
        np.mean(scores_b - scores_a)
    )

    bootstrap_differences = []

    for _ in range(samples):
        indices = random_generator.integers(
            0,
            size,
            size=size,
        )

        difference = np.mean(
            scores_b[indices] - scores_a[indices]
        )

        bootstrap_differences.append(
            float(difference)
        )

    bootstrap_differences = np.asarray(
        bootstrap_differences
    )

    lower = float(
        np.percentile(
            bootstrap_differences,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_differences,
            97.5,
        )
    )

    probability_non_positive = float(
        np.mean(
            bootstrap_differences <= 0
        )
    )

    probability_non_negative = float(
        np.mean(
            bootstrap_differences >= 0
        )
    )

    p_value = float(
        min(
            1.0,
            2 * min(
                probability_non_positive,
                probability_non_negative,
            ),
        )
    )

    return {
        "difference": observed_difference,
        "ci_low": lower,
        "ci_high": upper,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def extract_numbers(text):
    return re.findall(
        r"\d+(?:[.,]\d+)*",
        clean_text(text),
    )


def word_tokens(text):
    return re.findall(
        r"\b[\w'-]+\b",
        clean_text(text).lower(),
        flags=re.UNICODE,
    )


def number_mismatch(reference, prediction):
    reference_numbers = extract_numbers(reference)
    prediction_numbers = extract_numbers(prediction)

    if not reference_numbers:
        return 0

    for number in reference_numbers:
        if number not in prediction_numbers:
            return 1

    return 0


def length_ratio(reference, prediction):
    reference_tokens = word_tokens(reference)
    prediction_tokens = word_tokens(prediction)

    if len(reference_tokens) == 0:
        return np.nan

    return len(prediction_tokens) / len(reference_tokens)


def length_outlier(reference, prediction):
    ratio = length_ratio(
        reference,
        prediction,
    )

    if pd.isna(ratio):
        return 0

    return int(
        ratio < 0.50
        or ratio > 2.00
    )


def repeated_token_alert(prediction):
    tokens = word_tokens(prediction)

    if len(tokens) < 4:
        return 0

    unique_ratio = (
        len(set(tokens))
        / len(tokens)
    )

    return int(unique_ratio < 0.50)


def empty_output(prediction):
    return int(
        clean_text(prediction) == ""
    )


def exact_reference_match(reference, prediction):
    return int(
        clean_text(reference).lower()
        == clean_text(prediction).lower()
    )


def create_diagnostics(
    dataframe,
    model_columns,
):
    rows = []

    for model_name, column in model_columns.items():
        number_errors = []
        length_ratios = []
        length_alerts = []
        repetition_alerts = []
        empty_alerts = []
        exact_matches = []

        for _, row in dataframe.iterrows():
            reference = row["reference"]
            prediction = row[column]

            number_errors.append(
                number_mismatch(
                    reference,
                    prediction,
                )
            )

            length_ratios.append(
                length_ratio(
                    reference,
                    prediction,
                )
            )

            length_alerts.append(
                length_outlier(
                    reference,
                    prediction,
                )
            )

            repetition_alerts.append(
                repeated_token_alert(
                    prediction
                )
            )

            empty_alerts.append(
                empty_output(
                    prediction
                )
            )

            exact_matches.append(
                exact_reference_match(
                    reference,
                    prediction,
                )
            )

        valid_ratios = [
            value
            for value in length_ratios
            if not pd.isna(value)
        ]

        rows.append(
            {
                "model": model_name,
                "samples": len(dataframe),
                "mean_length_ratio": (
                    float(np.mean(valid_ratios))
                    if valid_ratios
                    else np.nan
                ),
                "number_mismatch_rate_percent": (
                    float(np.mean(number_errors) * 100)
                ),
                "length_outlier_rate_percent": (
                    float(np.mean(length_alerts) * 100)
                ),
                "repetition_alert_rate_percent": (
                    float(np.mean(repetition_alerts) * 100)
                ),
                "empty_output_rate_percent": (
                    float(np.mean(empty_alerts) * 100)
                ),
                "exact_reference_match_percent": (
                    float(np.mean(exact_matches) * 100)
                ),
            }
        )

    return pd.DataFrame(rows)


def create_pairwise_results(
    score_dictionary,
    metric_name,
):
    model_names = list(score_dictionary.keys())
    rows = []

    for first_index in range(len(model_names)):
        for second_index in range(
            first_index + 1,
            len(model_names),
        ):
            model_a = model_names[first_index]
            model_b = model_names[second_index]

            scores_a = np.asarray(
                score_dictionary[model_a],
                dtype=float,
            )

            scores_b = np.asarray(
                score_dictionary[model_b],
                dtype=float,
            )

            wins_a = int(
                np.sum(scores_a > scores_b)
            )

            wins_b = int(
                np.sum(scores_b > scores_a)
            )

            ties = int(
                np.sum(
                    np.isclose(
                        scores_a,
                        scores_b,
                        atol=1e-8,
                    )
                )
            )

            comparison = bootstrap_mean_difference(
                scores_a,
                scores_b,
            )

            rows.append(
                {
                    "metric": metric_name,
                    "system_a": model_a,
                    "system_b": model_b,
                    "system_a_wins": wins_a,
                    "system_b_wins": wins_b,
                    "ties": ties,
                    "mean_difference_b_minus_a": (
                        comparison["difference"]
                    ),
                    "difference_95_ci_low": (
                        comparison["ci_low"]
                    ),
                    "difference_95_ci_high": (
                        comparison["ci_high"]
                    ),
                    "p_value": (
                        comparison["p_value"]
                    ),
                    "significant_at_0.05": (
                        comparison[
                            "significant_at_0.05"
                        ]
                    ),
                }
            )

    return rows


def create_figure(summary):
    metrics_to_plot = []

    if "COMET" in summary.columns:
        metrics_to_plot.append("COMET")

    if (
        "COMETKiwi_QE" in summary.columns
        and summary["COMETKiwi_QE"].notna().any()
    ):
        metrics_to_plot.append(
            "COMETKiwi_QE"
        )

    if not metrics_to_plot:
        return

    plot_data = summary.set_index(
        "model"
    )[metrics_to_plot]

    plot_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.ylabel("Automatic neural metric score")
    plt.xlabel("Translation system")
    plt.title(
        "Automatic Neural Evaluation"
    )
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
    )

    plt.close()


def create_report(
    summary,
    significance,
    diagnostics,
    pairwise,
    cometkiwi_available,
):
    lines = []

    lines.append(
        "# Automatic Evaluation of Pashto-to-English NMT"
    )

    lines.append("")
    lines.append(
        "No human evaluation was conducted in this experiment."
    )

    lines.append("")
    lines.append(
        "The systems were evaluated automatically using "
        "BLEU, chrF++, TER and reference-based COMET."
    )

    if cometkiwi_available:
        lines.append("")
        lines.append(
            "Reference-free COMETKiwi quality estimation "
            "was also calculated."
        )

    lines.append("")
    lines.append(
        "Automatic diagnostic indicators were used to detect "
        "number mismatches, unusually short or long outputs, "
        "repetition and empty translations."
    )

    lines.append("")
    lines.append("## Main automatic results")
    lines.append("")
    lines.append(
        summary.to_markdown(index=False)
    )

    lines.append("")
    lines.append("## Bootstrap significance")
    lines.append("")
    lines.append(
        significance.to_markdown(index=False)
    )

    lines.append("")
    lines.append("## Pairwise sentence-level wins")
    lines.append("")
    lines.append(
        pairwise.to_markdown(index=False)
    )

    lines.append("")
    lines.append("## Automatic diagnostic indicators")
    lines.append("")
    lines.append(
        diagnostics.to_markdown(index=False)
    )

    lines.append("")
    lines.append("## Interpretation rule")
    lines.append("")
    lines.append(
        "For BLEU, chrF++ and COMET, a higher score is better. "
        "For TER, a lower score is better."
    )

    lines.append("")
    lines.append("## Limitation")
    lines.append("")
    lines.append(
        "Automatic metrics estimate translation quality but do "
        "not replace evaluation by qualified human annotators. "
        "The absence of human evaluation should therefore be "
        "stated explicitly in the paper."
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--skip-cometkiwi",
        action="store_true",
    )

    arguments = parser.parse_args()

    input_path = Path(arguments.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    TABLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    validate_input(dataframe)

    required_columns = [
        "source",
        "reference",
        *MODEL_COLUMNS.values(),
    ]

    dataframe = dataframe.dropna(
        subset=required_columns,
    ).reset_index(drop=True)

    for column in required_columns:
        dataframe[column] = (
            dataframe[column]
            .map(clean_text)
        )

    print()
    print("=" * 80)
    print("AUTOMATIC NMT EVALUATION")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Samples: {len(dataframe)}")

    reference_comet_scores, reference_comet_system = (
        evaluate_reference_comet(
            dataframe,
            MODEL_COLUMNS,
            arguments.batch_size,
        )
    )

    kiwi_sentence_scores = {}
    kiwi_system_scores = {}

    if not arguments.skip_cometkiwi:
        (
            kiwi_sentence_scores,
            kiwi_system_scores,
        ) = evaluate_cometkiwi(
            dataframe,
            MODEL_COLUMNS,
            arguments.batch_size,
        )

    references = dataframe[
        "reference"
    ].tolist()

    summary_rows = []

    sentence_output = pd.DataFrame(
        {
            "sample_id": range(
                1,
                len(dataframe) + 1,
            ),
            "source": dataframe["source"],
            "reference": dataframe["reference"],
        }
    )

    sentence_metric_dictionaries = {
        "BLEU": {},
        "chrF++": {},
        "TER": {},
        "COMET": reference_comet_scores,
    }

    if kiwi_sentence_scores:
        sentence_metric_dictionaries[
            "COMETKiwi_QE"
        ] = kiwi_sentence_scores

    for model_name, column in MODEL_COLUMNS.items():
        predictions = dataframe[
            column
        ].tolist()

        corpus = corpus_metrics(
            references,
            predictions,
        )

        sentence_bleu = []
        sentence_chrf = []
        sentence_ter = []

        for reference, prediction in zip(
            references,
            predictions,
        ):
            bleu, chrf, ter = sentence_metrics(
                reference,
                prediction,
            )

            sentence_bleu.append(bleu)
            sentence_chrf.append(chrf)
            sentence_ter.append(ter)

        sentence_metric_dictionaries[
            "BLEU"
        ][model_name] = sentence_bleu

        sentence_metric_dictionaries[
            "chrF++"
        ][model_name] = sentence_chrf

        sentence_metric_dictionaries[
            "TER"
        ][model_name] = sentence_ter

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        sentence_output[
            f"{safe_name}_prediction"
        ] = predictions

        sentence_output[
            f"{safe_name}_sentence_bleu"
        ] = sentence_bleu

        sentence_output[
            f"{safe_name}_sentence_chrf"
        ] = sentence_chrf

        sentence_output[
            f"{safe_name}_sentence_ter"
        ] = sentence_ter

        sentence_output[
            f"{safe_name}_comet"
        ] = reference_comet_scores[
            model_name
        ]

        if model_name in kiwi_sentence_scores:
            sentence_output[
                f"{safe_name}_cometkiwi_qe"
            ] = kiwi_sentence_scores[
                model_name
            ]

        summary_rows.append(
            {
                "model": model_name,
                "samples": len(predictions),
                "BLEU": corpus["BLEU"],
                "chrF++": corpus["chrF++"],
                "TER": corpus["TER"],
                "COMET": reference_comet_system[
                    model_name
                ],
                "COMETKiwi_QE": (
                    kiwi_system_scores.get(
                        model_name,
                        np.nan,
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    diagnostics = create_diagnostics(
        dataframe,
        MODEL_COLUMNS,
    )

    significance_rows = []
    pairwise_rows = []

    baseline_name = "Baseline NLLB"

    for metric_name, score_dictionary in (
        sentence_metric_dictionaries.items()
    ):
        if baseline_name not in score_dictionary:
            continue

        for model_name in score_dictionary:
            if model_name == baseline_name:
                continue

            baseline_scores = score_dictionary[
                baseline_name
            ]

            model_scores = score_dictionary[
                model_name
            ]

            if metric_name == "TER":
                result = bootstrap_mean_difference(
                    model_scores,
                    baseline_scores,
                )

                reported_difference = (
                    np.mean(model_scores)
                    - np.mean(baseline_scores)
                )

            else:
                result = bootstrap_mean_difference(
                    baseline_scores,
                    model_scores,
                )

                reported_difference = (
                    np.mean(model_scores)
                    - np.mean(baseline_scores)
                )

            significance_rows.append(
                {
                    "metric": metric_name,
                    "baseline": baseline_name,
                    "comparison_model": model_name,
                    "mean_difference_model_minus_baseline": (
                        float(reported_difference)
                    ),
                    "bootstrap_95_ci_low": (
                        result["ci_low"]
                    ),
                    "bootstrap_95_ci_high": (
                        result["ci_high"]
                    ),
                    "p_value": result["p_value"],
                    "significant_at_0.05": (
                        result[
                            "significant_at_0.05"
                        ]
                    ),
                }
            )

        pairwise_rows.extend(
            create_pairwise_results(
                score_dictionary,
                metric_name,
            )
        )

    significance = pd.DataFrame(
        significance_rows
    )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    numeric_summary_columns = [
        "BLEU",
        "chrF++",
        "TER",
        "COMET",
        "COMETKiwi_QE",
    ]

    for column in numeric_summary_columns:
        if column in summary.columns:
            summary[column] = (
                summary[column].round(6)
            )

    numeric_significance_columns = (
        significance.select_dtypes(
            include="number"
        ).columns
    )

    significance[
        numeric_significance_columns
    ] = significance[
        numeric_significance_columns
    ].round(6)

    numeric_pairwise_columns = (
        pairwise.select_dtypes(
            include="number"
        ).columns
    )

    pairwise[
        numeric_pairwise_columns
    ] = pairwise[
        numeric_pairwise_columns
    ].round(6)

    numeric_diagnostic_columns = (
        diagnostics.select_dtypes(
            include="number"
        ).columns
    )

    diagnostics[
        numeric_diagnostic_columns
    ] = diagnostics[
        numeric_diagnostic_columns
    ].round(6)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    sentence_output.to_csv(
        SENTENCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    significance.to_csv(
        SIGNIFICANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics.to_csv(
        DIAGNOSTICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pairwise.to_csv(
        PAIRWISE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    create_figure(summary)

    create_report(
        summary,
        significance,
        diagnostics,
        pairwise,
        bool(kiwi_system_scores),
    )

    print()
    print("=" * 80)
    print("AUTOMATIC EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Sentence scores: {SENTENCE_PATH}")
    print(f"Significance: {SIGNIFICANCE_PATH}")
    print(f"Diagnostics: {DIAGNOSTICS_PATH}")
    print(f"Pairwise wins: {PAIRWISE_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print(f"Report: {REPORT_PATH}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
