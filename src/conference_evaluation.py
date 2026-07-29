import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacrebleu


REFERENCE_NAMES = [
    "reference",
    "target",
    "english_reference",
    "reference_english",
    "gold",
    "gold_reference",
    "target_text",
    "english",
]

SOURCE_NAMES = [
    "source",
    "pashto",
    "source_pashto",
    "pashto_source",
    "source_text",
]

MODEL_PATTERNS = {
    "Baseline NLLB": [
        "baseline_prediction",
        "baseline_translation",
        "baseline_output",
        "nllb_baseline",
        "baseline",
    ],
    "Original LoRA": [
        "original_lora_prediction",
        "lora_prediction",
        "finetuned_prediction",
        "fine_tuned_prediction",
        "original_lora",
    ],
    "Semantic LoRA": [
        "semantic_lora_prediction",
        "semantic_filtered_prediction",
        "semantic_prediction",
        "filtered_lora_prediction",
        "semantic_lora",
    ],
}


def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def find_column(columns, candidates):
    normalized = {normalize_name(column): column for column in columns}

    for candidate in candidates:
        candidate = normalize_name(candidate)

        if candidate in normalized:
            return normalized[candidate]

    for candidate in candidates:
        candidate = normalize_name(candidate)

        for normalized_name, original_name in normalized.items():
            if candidate in normalized_name:
                return original_name

    return None


def find_model_columns(dataframe, reference_column, source_column):
    model_columns = {}

    for model_name, patterns in MODEL_PATTERNS.items():
        column = find_column(dataframe.columns, patterns)

        if column is not None:
            model_columns[model_name] = column

    excluded = {
        reference_column,
        source_column,
    }

    prediction_candidates = []

    for column in dataframe.columns:
        if column in excluded:
            continue

        normalized = normalize_name(column)

        if any(
            word in normalized
            for word in ["prediction", "translation", "hypothesis", "output"]
        ):
            prediction_candidates.append(column)

    used_columns = set(model_columns.values())

    for column in prediction_candidates:
        if column in used_columns:
            continue

        model_name = column.replace("_", " ").title()
        model_columns[model_name] = column

    return model_columns


def corpus_scores(references, predictions):
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

    return {
        "BLEU": float(bleu),
        "chrF++": float(chrf),
    }


def sentence_chrf(reference, prediction):
    return sacrebleu.sentence_chrf(
        prediction,
        [reference],
        word_order=2,
    ).score


def bootstrap_scores(
    references,
    predictions,
    samples=1000,
    seed=42,
):
    random_generator = np.random.default_rng(seed)
    size = len(references)

    bleu_values = []
    chrf_values = []

    for _ in range(samples):
        indices = random_generator.integers(
            low=0,
            high=size,
            size=size,
        )

        sampled_references = [
            references[index]
            for index in indices
        ]

        sampled_predictions = [
            predictions[index]
            for index in indices
        ]

        scores = corpus_scores(
            sampled_references,
            sampled_predictions,
        )

        bleu_values.append(scores["BLEU"])
        chrf_values.append(scores["chrF++"])

    return {
        "BLEU_low": float(np.percentile(bleu_values, 2.5)),
        "BLEU_high": float(np.percentile(bleu_values, 97.5)),
        "chrF_low": float(np.percentile(chrf_values, 2.5)),
        "chrF_high": float(np.percentile(chrf_values, 97.5)),
    }


def paired_bootstrap_test(
    references,
    predictions_a,
    predictions_b,
    samples=2000,
    seed=42,
):
    random_generator = np.random.default_rng(seed)
    size = len(references)

    full_a = corpus_scores(references, predictions_a)
    full_b = corpus_scores(references, predictions_b)

    observed_bleu_difference = (
        full_b["BLEU"] - full_a["BLEU"]
    )

    observed_chrf_difference = (
        full_b["chrF++"] - full_a["chrF++"]
    )

    bleu_differences = []
    chrf_differences = []

    for _ in range(samples):
        indices = random_generator.integers(
            low=0,
            high=size,
            size=size,
        )

        sampled_references = [
            references[index]
            for index in indices
        ]

        sampled_a = [
            predictions_a[index]
            for index in indices
        ]

        sampled_b = [
            predictions_b[index]
            for index in indices
        ]

        scores_a = corpus_scores(
            sampled_references,
            sampled_a,
        )

        scores_b = corpus_scores(
            sampled_references,
            sampled_b,
        )

        bleu_differences.append(
            scores_b["BLEU"] - scores_a["BLEU"]
        )

        chrf_differences.append(
            scores_b["chrF++"] - scores_a["chrF++"]
        )

    bleu_differences = np.array(bleu_differences)
    chrf_differences = np.array(chrf_differences)

    bleu_p_value = 2 * min(
        np.mean(bleu_differences <= 0),
        np.mean(bleu_differences >= 0),
    )

    chrf_p_value = 2 * min(
        np.mean(chrf_differences <= 0),
        np.mean(chrf_differences >= 0),
    )

    return {
        "BLEU_difference": float(
            observed_bleu_difference
        ),
        "BLEU_p_value": float(
            min(1.0, bleu_p_value)
        ),
        "BLEU_difference_low": float(
            np.percentile(bleu_differences, 2.5)
        ),
        "BLEU_difference_high": float(
            np.percentile(bleu_differences, 97.5)
        ),
        "chrF_difference": float(
            observed_chrf_difference
        ),
        "chrF_p_value": float(
            min(1.0, chrf_p_value)
        ),
        "chrF_difference_low": float(
            np.percentile(chrf_differences, 2.5)
        ),
        "chrF_difference_high": float(
            np.percentile(chrf_differences, 97.5)
        ),
    }


def build_sentence_analysis(
    dataframe,
    reference_column,
    source_column,
    model_columns,
):
    result = pd.DataFrame()

    if source_column is not None:
        result["source"] = dataframe[source_column].astype(str)

    result["reference"] = dataframe[
        reference_column
    ].astype(str)

    for model_name, column in model_columns.items():
        safe_name = normalize_name(model_name)

        predictions = dataframe[column].astype(str).tolist()
        references = dataframe[
            reference_column
        ].astype(str).tolist()

        result[f"{safe_name}_prediction"] = predictions

        result[f"{safe_name}_sentence_chrf"] = [
            sentence_chrf(reference, prediction)
            for reference, prediction in zip(
                references,
                predictions,
            )
        ]

    return result


def create_blinded_human_evaluation(
    dataframe,
    reference_column,
    source_column,
    model_columns,
    output_path,
    sample_size=50,
    seed=42,
):
    size = min(sample_size, len(dataframe))

    sampled = dataframe.sample(
        n=size,
        random_state=seed,
    ).reset_index(drop=True)

    random_generator = random.Random(seed)

    rows = []

    for index, row in sampled.iterrows():
        systems = list(model_columns.items())
        random_generator.shuffle(systems)

        output_row = {
            "sample_id": index + 1,
        }

        if source_column is not None:
            output_row["pashto_source"] = str(
                row[source_column]
            )

        output_row["english_reference"] = str(
            row[reference_column]
        )

        mapping = {}

        labels = ["System A", "System B", "System C"]

        for label, (model_name, column) in zip(
            labels,
            systems,
        ):
            output_row[f"{normalize_name(label)}_output"] = str(
                row[column]
            )

            mapping[label] = model_name

        output_row["best_system"] = ""
        output_row["adequacy_a_1_to_5"] = ""
        output_row["adequacy_b_1_to_5"] = ""
        output_row["adequacy_c_1_to_5"] = ""
        output_row["fluency_a_1_to_5"] = ""
        output_row["fluency_b_1_to_5"] = ""
        output_row["fluency_c_1_to_5"] = ""
        output_row["missing_information"] = ""
        output_row["hallucination"] = ""
        output_row["named_entity_error"] = ""
        output_row["comments"] = ""
        output_row["_hidden_mapping"] = json.dumps(
            mapping,
            ensure_ascii=False,
        )

        rows.append(output_row)

    blinded = pd.DataFrame(rows)
    blinded.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def create_metric_plot(
    metrics_dataframe,
    output_path,
):
    models = metrics_dataframe["Model"].tolist()
    bleu_values = metrics_dataframe["BLEU"].tolist()
    chrf_values = metrics_dataframe["chrF++"].tolist()

    x_positions = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x_positions - width / 2,
        bleu_values,
        width,
        label="BLEU",
    )

    plt.bar(
        x_positions + width / 2,
        chrf_values,
        width,
        label="chrF++",
    )

    plt.xticks(
        x_positions,
        models,
        rotation=15,
    )

    plt.ylabel("Score")
    plt.title("Baseline and Fine-Tuned Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_markdown_report(
    metrics_dataframe,
    significance_dataframe,
    input_path,
    output_path,
):
    lines = []

    lines.append(
        "# Statistical Evaluation of Pashto-to-English NMT"
    )
    lines.append("")
    lines.append(
        f"Input predictions: `{input_path}`"
    )
    lines.append("")
    lines.append(
        "The same reference sentences were used for all systems."
    )
    lines.append("")
    lines.append("## Corpus-level results")
    lines.append("")

    lines.append(
        metrics_dataframe.to_markdown(index=False)
    )

    lines.append("")
    lines.append(
        "Confidence intervals were calculated using paired "
        "bootstrap resampling."
    )
    lines.append("")
    lines.append("## Paired significance tests")
    lines.append("")

    if len(significance_dataframe) > 0:
        lines.append(
            significance_dataframe.to_markdown(index=False)
        )
    else:
        lines.append(
            "At least two model-output columns are required."
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- A p-value below 0.05 is commonly treated as "
        "evidence that the difference is statistically significant."
    )
    lines.append(
        "- Statistical significance does not automatically mean "
        "that the improvement is practically large."
    )
    lines.append(
        "- Automatic metrics should be accompanied by blinded "
        "human evaluation of adequacy, fluency, missing content, "
        "named entities, and hallucination."
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "outputs/tables/"
            "remaining_checkpoint_comparison.csv"
        ),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--human-samples",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    arguments = parser.parse_args()

    input_path = Path(arguments.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    tables_directory = Path("outputs/tables")
    figures_directory = Path("outputs/figures")
    reports_directory = Path("reports")

    tables_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(input_path)

    reference_column = find_column(
        dataframe.columns,
        REFERENCE_NAMES,
    )

    source_column = find_column(
        dataframe.columns,
        SOURCE_NAMES,
    )

    if reference_column is None:
        print("Available columns:")
        print(list(dataframe.columns))

        raise ValueError(
            "Reference column could not be detected."
        )

    model_columns = find_model_columns(
        dataframe,
        reference_column,
        source_column,
    )

    if len(model_columns) < 2:
        print("Available columns:")
        print(list(dataframe.columns))

        raise ValueError(
            "At least two prediction columns are required."
        )

    print(f"Reference column: {reference_column}")
    print(f"Source column: {source_column}")
    print(f"Model columns: {model_columns}")

    clean_dataframe = dataframe.copy()

    required_columns = [
        reference_column,
        *model_columns.values(),
    ]

    clean_dataframe = clean_dataframe.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    references = clean_dataframe[
        reference_column
    ].astype(str).tolist()

    metric_rows = []
    model_predictions = {}

    for model_name, column in model_columns.items():
        predictions = clean_dataframe[
            column
        ].astype(str).tolist()

        model_predictions[model_name] = predictions

        scores = corpus_scores(
            references,
            predictions,
        )

        confidence_intervals = bootstrap_scores(
            references,
            predictions,
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )

        metric_rows.append(
            {
                "Model": model_name,
                "Samples": len(references),
                "BLEU": round(scores["BLEU"], 4),
                "BLEU 95% CI Low": round(
                    confidence_intervals["BLEU_low"],
                    4,
                ),
                "BLEU 95% CI High": round(
                    confidence_intervals["BLEU_high"],
                    4,
                ),
                "chrF++": round(scores["chrF++"], 4),
                "chrF 95% CI Low": round(
                    confidence_intervals["chrF_low"],
                    4,
                ),
                "chrF 95% CI High": round(
                    confidence_intervals["chrF_high"],
                    4,
                ),
            }
        )

    metrics_dataframe = pd.DataFrame(metric_rows)

    significance_rows = []

    model_names = list(model_predictions.keys())

    baseline_name = model_names[0]

    if "Baseline NLLB" in model_predictions:
        baseline_name = "Baseline NLLB"

    for model_name in model_names:
        if model_name == baseline_name:
            continue

        test_result = paired_bootstrap_test(
            references,
            model_predictions[baseline_name],
            model_predictions[model_name],
            samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )

        significance_rows.append(
            {
                "System A": baseline_name,
                "System B": model_name,
                "BLEU Difference B-A": round(
                    test_result["BLEU_difference"],
                    4,
                ),
                "BLEU p-value": round(
                    test_result["BLEU_p_value"],
                    4,
                ),
                "chrF Difference B-A": round(
                    test_result["chrF_difference"],
                    4,
                ),
                "chrF p-value": round(
                    test_result["chrF_p_value"],
                    4,
                ),
                "Significant at 0.05": (
                    test_result["chrF_p_value"] < 0.05
                ),
            }
        )

    significance_dataframe = pd.DataFrame(
        significance_rows
    )

    sentence_analysis = build_sentence_analysis(
        clean_dataframe,
        reference_column,
        source_column,
        model_columns,
    )

    metrics_path = (
        tables_directory
        / "conference_model_metrics_with_ci.csv"
    )

    significance_path = (
        tables_directory
        / "conference_paired_bootstrap_significance.csv"
    )

    sentence_path = (
        tables_directory
        / "conference_sentence_level_analysis.csv"
    )

    human_path = (
        tables_directory
        / "conference_blinded_human_evaluation.csv"
    )

    figure_path = (
        figures_directory
        / "conference_model_comparison.png"
    )

    report_path = (
        reports_directory
        / "conference_statistical_evaluation.md"
    )

    metrics_dataframe.to_csv(
        metrics_path,
        index=False,
    )

    significance_dataframe.to_csv(
        significance_path,
        index=False,
    )

    sentence_analysis.to_csv(
        sentence_path,
        index=False,
        encoding="utf-8-sig",
    )

    create_blinded_human_evaluation(
        clean_dataframe,
        reference_column,
        source_column,
        model_columns,
        human_path,
        sample_size=arguments.human_samples,
        seed=arguments.seed,
    )

    create_metric_plot(
        metrics_dataframe,
        figure_path,
    )

    create_markdown_report(
        metrics_dataframe,
        significance_dataframe,
        input_path,
        report_path,
    )

    print()
    print("Evaluation completed.")
    print(f"Metrics: {metrics_path}")
    print(f"Significance: {significance_path}")
    print(f"Sentence analysis: {sentence_path}")
    print(f"Human evaluation: {human_path}")
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")
    print()
    print(metrics_dataframe.to_string(index=False))

    if len(significance_dataframe) > 0:
        print()
        print(significance_dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
