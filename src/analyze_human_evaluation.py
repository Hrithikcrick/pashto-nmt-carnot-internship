import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FORM_PATH = Path(
    "outputs/tables/research_human_evaluation_filled.csv"
)

KEY_PATH = Path(
    "outputs/tables/research_human_evaluation_key.csv"
)

SUMMARY_PATH = Path(
    "outputs/tables/research_human_evaluation_summary.csv"
)

LONG_PATH = Path(
    "outputs/tables/research_human_evaluation_long.csv"
)

REPORT_PATH = Path(
    "reports/research_human_evaluation.md"
)

FIGURE_PATH = Path(
    "outputs/figures/research_human_evaluation.png"
)


def normalize_best_system(value):
    value = str(value).strip().upper()

    replacements = {
        "SYSTEM A": "A",
        "SYSTEM B": "B",
        "SYSTEM C": "C",
    }

    return replacements.get(value, value)


def yes_to_number(value):
    value = str(value).strip().lower()

    if value in {"yes", "y", "1", "true"}:
        return 1

    if value in {"no", "n", "0", "false"}:
        return 0

    return None


def main():
    if not FORM_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation form not found: {FORM_PATH}"
        )

    if not KEY_PATH.exists():
        raise FileNotFoundError(
            f"Hidden key not found: {KEY_PATH}"
        )

    form = pd.read_csv(
        FORM_PATH,
        encoding="utf-8-sig"
    )

    key = pd.read_csv(
        KEY_PATH,
        encoding="utf-8-sig"
    )

    data = form.merge(
        key,
        on="sample_id",
        how="inner"
    )

    score_columns = [
        "adequacy_a_1_to_5",
        "adequacy_b_1_to_5",
        "adequacy_c_1_to_5",
        "fluency_a_1_to_5",
        "fluency_b_1_to_5",
        "fluency_c_1_to_5",
    ]

    for column in score_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    completed = data.dropna(
        subset=[
            "best_system",
            *score_columns,
        ]
    ).copy()

    completed["best_system"] = (
        completed["best_system"]
        .apply(normalize_best_system)
    )

    completed = completed[
        completed["best_system"].isin(
            ["A", "B", "C"]
        )
    ]

    for column in score_columns:
        invalid = completed[
            ~completed[column].between(1, 5)
        ]

        if len(invalid) > 0:
            raise ValueError(
                f"Scores in {column} must be between 1 and 5."
            )

    if len(completed) == 0:
        raise ValueError(
            "No completed evaluation rows were found."
        )

    if len(completed) < 30:
        raise ValueError(
            f"Only {len(completed)} samples are completed. "
            "Complete at least 30 samples before running analysis."
        )

    long_rows = []

    for _, row in completed.iterrows():
        mapping = json.loads(
            row["_hidden_mapping"]
        )

        selected_label = (
            f"System {row['best_system']}"
        )

        for letter in ["A", "B", "C"]:
            lower = letter.lower()
            system_label = f"System {letter}"
            model_name = mapping[system_label]

            adequacy = float(
                row[f"adequacy_{lower}_1_to_5"]
            )

            fluency = float(
                row[f"fluency_{lower}_1_to_5"]
            )

            missing_information = yes_to_number(
                row.get(
                    f"missing_information_{lower}",
                    ""
                )
            )

            hallucination = yes_to_number(
                row.get(
                    f"hallucination_{lower}",
                    ""
                )
            )

            named_entity_error = yes_to_number(
                row.get(
                    f"named_entity_error_{lower}",
                    ""
                )
            )

            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "system_position": letter,
                    "model": model_name,
                    "adequacy": adequacy,
                    "fluency": fluency,
                    "average_human_score": (
                        adequacy + fluency
                    ) / 2,
                    "selected_as_best": int(
                        selected_label == system_label
                    ),
                    "missing_information": (
                        missing_information
                    ),
                    "hallucination": hallucination,
                    "named_entity_error": (
                        named_entity_error
                    ),
                }
            )

    long_df = pd.DataFrame(long_rows)

    summary = (
        long_df.groupby("model")
        .agg(
            evaluated_samples=(
                "sample_id",
                "count"
            ),
            mean_adequacy=(
                "adequacy",
                "mean"
            ),
            adequacy_std=(
                "adequacy",
                "std"
            ),
            mean_fluency=(
                "fluency",
                "mean"
            ),
            fluency_std=(
                "fluency",
                "std"
            ),
            mean_human_score=(
                "average_human_score",
                "mean"
            ),
            best_system_votes=(
                "selected_as_best",
                "sum"
            ),
            missing_information_rate=(
                "missing_information",
                "mean"
            ),
            hallucination_rate=(
                "hallucination",
                "mean"
            ),
            named_entity_error_rate=(
                "named_entity_error",
                "mean"
            ),
        )
        .reset_index()
    )

    total_evaluations = len(completed)

    summary["best_system_percentage"] = (
        summary["best_system_votes"]
        / total_evaluations
        * 100
    )

    percentage_columns = [
        "missing_information_rate",
        "hallucination_rate",
        "named_entity_error_rate",
    ]

    for column in percentage_columns:
        summary[column] = (
            summary[column] * 100
        )

    numeric_columns = summary.select_dtypes(
        include="number"
    ).columns

    summary[numeric_columns] = (
        summary[numeric_columns].round(4)
    )

    LONG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    long_df.to_csv(
        LONG_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    plot_data = summary.set_index("model")[
        [
            "mean_adequacy",
            "mean_fluency",
        ]
    ]

    plot_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("Mean human score (1–5)")
    plt.xlabel("Translation system")
    plt.title(
        "Blinded Human Evaluation Results"
    )
    plt.xticks(rotation=15)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH,
        dpi=300
    )
    plt.close()

    best_model = summary.sort_values(
        by="mean_human_score",
        ascending=False
    ).iloc[0]

    report = []

    report.append(
        "# Blinded Human Evaluation"
    )

    report.append("")
    report.append(
        f"Completed evaluation samples: {len(completed)}"
    )

    report.append("")
    report.append(
        "Each translation was evaluated for adequacy "
        "and fluency using a five-point scale."
    )

    report.append("")
    report.append(
        "System identities were hidden during evaluation "
        "and decoded only after scoring."
    )

    report.append("")
    report.append("## Results")
    report.append("")
    report.append(
        summary.to_markdown(index=False)
    )

    report.append("")
    report.append("## Main finding")
    report.append("")

    report.append(
        f"The highest mean human-evaluation score was "
        f"obtained by **{best_model['model']}**, with a "
        f"mean score of "
        f"{best_model['mean_human_score']:.4f}/5."
    )

    report.append("")
    report.append("## Evaluation limitation")
    report.append("")

    report.append(
        "This evaluation should be described as a "
        "single-annotator blinded evaluation unless "
        "another independent evaluator scores the same "
        "translation samples."
    )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print()
    print("Human evaluation analysis completed.")
    print(f"Completed samples: {len(completed)}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Detailed results: {LONG_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
