import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sacrebleu.metrics import CHRF


PROJECT_METRICS_FILE = Path(
    "outputs/tables/research_automatic_metrics.csv"
)

EXTERNAL_METRICS_FILE = Path(
    "outputs/tables/external_baseline_metrics.csv"
)

PREDICTIONS_FILE = Path(
    "outputs/tables/research_predictions_combined.csv"
)

OUTPUT_TABLE_DIRECTORY = Path(
    "outputs/tables"
)

FIGURE_DIRECTORY = Path(
    "paper/figures"
)

GENERATED_LATEX_DIRECTORY = Path(
    "paper/generated"
)

REPORT_DIRECTORY = Path(
    "reports"
)


def latex_escape(value):
    text = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for original, replacement in replacements.items():
        text = text.replace(
            original,
            replacement,
        )

    return text


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_metrics():
    require_file(
        PROJECT_METRICS_FILE
    )

    require_file(
        EXTERNAL_METRICS_FILE
    )

    project_metrics = pd.read_csv(
        PROJECT_METRICS_FILE,
        encoding="utf-8-sig",
    )

    external_metrics = pd.read_csv(
        EXTERNAL_METRICS_FILE,
        encoding="utf-8-sig",
    )

    required_project_columns = [
        "model",
        "samples",
        "BLEU",
        "chrF++",
        "TER",
        "COMET",
    ]

    for column in required_project_columns:
        if column not in project_metrics.columns:
            raise ValueError(
                f"Missing project metric column: {column}"
            )

    required_external_columns = [
        "model",
        "samples",
        "BLEU",
        "chrF++",
        "TER",
    ]

    for column in required_external_columns:
        if column not in external_metrics.columns:
            raise ValueError(
                f"Missing external metric column: {column}"
            )

    if "COMET" not in external_metrics.columns:
        external_metrics["COMET"] = np.nan

    project_table = project_metrics[
        [
            "model",
            "samples",
            "BLEU",
            "chrF++",
            "TER",
            "COMET",
        ]
    ].copy()

    project_table["setting"] = (
        "NLLB baseline or LoRA adaptation"
    )

    external_table = external_metrics[
        [
            "model",
            "samples",
            "BLEU",
            "chrF++",
            "TER",
            "COMET",
        ]
    ].copy()

    external_table["setting"] = (
        "External zero-shot baseline"
    )

    combined = pd.concat(
        [
            project_table,
            external_table,
        ],
        ignore_index=True,
    )

    preferred_order = [
        "Baseline NLLB",
        "Original LoRA",
        "Semantic LoRA",
        "M2M100 418M",
        "mBART-50",
    ]

    order_map = {
        model_name: index
        for index, model_name
        in enumerate(preferred_order)
    }

    combined["order"] = (
        combined["model"]
        .map(order_map)
        .fillna(len(preferred_order))
    )

    combined = (
        combined
        .sort_values("order")
        .drop(columns=["order"])
        .reset_index(drop=True)
    )

    combined["BLEU"] = pd.to_numeric(
        combined["BLEU"],
        errors="coerce",
    )

    combined["chrF++"] = pd.to_numeric(
        combined["chrF++"],
        errors="coerce",
    )

    combined["TER"] = pd.to_numeric(
        combined["TER"],
        errors="coerce",
    )

    combined["COMET"] = pd.to_numeric(
        combined["COMET"],
        errors="coerce",
    )

    output_path = (
        OUTPUT_TABLE_DIRECTORY
        / "instructor_all_model_metrics.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return combined, output_path


def select_qualitative_examples():
    require_file(
        PREDICTIONS_FILE
    )

    dataframe = pd.read_csv(
        PREDICTIONS_FILE,
        encoding="utf-8-sig",
    )

    required_columns = [
        "source",
        "reference",
        "baseline_prediction",
        "original_lora_prediction",
        "semantic_lora_prediction",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(missing_columns)
        )

    chrf_metric = CHRF(
        word_order=2
    )

    def sentence_chrf(
        reference,
        prediction,
    ):
        return chrf_metric.sentence_score(
            str(prediction),
            [str(reference)],
        ).score

    dataframe[
        "baseline_sentence_chrf"
    ] = dataframe.apply(
        lambda row: sentence_chrf(
            row["reference"],
            row["baseline_prediction"],
        ),
        axis=1,
    )

    dataframe[
        "original_lora_sentence_chrf"
    ] = dataframe.apply(
        lambda row: sentence_chrf(
            row["reference"],
            row["original_lora_prediction"],
        ),
        axis=1,
    )

    dataframe[
        "semantic_lora_sentence_chrf"
    ] = dataframe.apply(
        lambda row: sentence_chrf(
            row["reference"],
            row["semantic_lora_prediction"],
        ),
        axis=1,
    )

    dataframe[
        "semantic_minus_baseline_chrf"
    ] = (
        dataframe[
            "semantic_lora_sentence_chrf"
        ]
        - dataframe[
            "baseline_sentence_chrf"
        ]
    )

    strong_improvements = (
        dataframe
        .sort_values(
            "semantic_minus_baseline_chrf",
            ascending=False,
        )
        .head(3)
        .copy()
    )

    useful_regressions = (
        dataframe
        .sort_values(
            "semantic_minus_baseline_chrf",
            ascending=True,
        )
        .head(2)
        .copy()
    )

    strong_improvements[
        "example_category"
    ] = "Improvement candidate"

    useful_regressions[
        "example_category"
    ] = "Regression candidate"

    selected = pd.concat(
        [
            strong_improvements,
            useful_regressions,
        ],
        ignore_index=True,
    )

    selected.insert(
        0,
        "example_id",
        range(
            1,
            len(selected) + 1,
        ),
    )

    output_path = (
        OUTPUT_TABLE_DIRECTORY
        / "instructor_qualitative_examples.csv"
    )

    selected.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return selected, output_path


def create_quality_figure(metrics):
    model_names = (
        metrics["model"]
        .astype(str)
        .tolist()
    )

    positions = np.arange(
        len(model_names)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    bleu_bars = axis.bar(
        positions - width / 2,
        metrics["BLEU"],
        width,
        label="BLEU",
    )

    chrf_bars = axis.bar(
        positions + width / 2,
        metrics["chrF++"],
        width,
        label="chrF++",
    )

    axis.set_title(
        "Pashto-to-English Model Comparison"
    )

    axis.set_ylabel(
        "Translation quality score"
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        model_names,
        rotation=18,
        ha="right",
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.bar_label(
        bleu_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )

    axis.bar_label(
        chrf_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "instructor_model_quality_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_ter_figure(metrics):
    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    bars = axis.bar(
        metrics["model"],
        metrics["TER"],
    )

    axis.set_title(
        "Translation Edit Rate Across Models"
    )

    axis.set_ylabel(
        "TER — lower is better"
    )

    axis.tick_params(
        axis="x",
        rotation=18,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "instructor_model_ter_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_comet_figure(metrics):
    comet_metrics = metrics[
        metrics["COMET"].notna()
    ].copy()

    if comet_metrics.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    bars = axis.bar(
        comet_metrics["model"],
        comet_metrics["COMET"],
    )

    axis.set_title(
        "COMET Neural Evaluation Across Models"
    )

    axis.set_ylabel(
        "COMET score — higher is better"
    )

    axis.tick_params(
        axis="x",
        rotation=18,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.bar_label(
        bars,
        fmt="%.4f",
        padding=3,
        fontsize=8,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "instructor_model_comet_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_pipeline_figure():
    figure, axis = plt.subplots(
        figsize=(16, 7)
    )

    axis.set_xlim(
        0,
        16,
    )

    axis.set_ylim(
        0,
        7,
    )

    axis.axis(
        "off"
    )

    stages = [
        (
            0.3,
            4.4,
            2.0,
            1.3,
            "Raw Corpus\n93,498 pairs",
        ),
        (
            2.8,
            4.4,
            2.0,
            1.3,
            "Rule-Based Cleaning\n90,978 pairs",
        ),
        (
            5.3,
            4.4,
            2.0,
            1.3,
            "Leakage-Controlled\nFixed Splits",
        ),
        (
            7.8,
            4.4,
            2.0,
            1.3,
            "Semantic Filtering\n8K / 4K / 2K",
        ),
        (
            10.3,
            4.4,
            2.0,
            1.3,
            "NLLB + LoRA\nAdaptation",
        ),
        (
            12.8,
            4.4,
            2.8,
            1.3,
            "Evaluation\nBLEU • chrF++\nTER • COMET",
        ),
    ]

    for (
        x,
        y,
        width,
        height,
        label,
    ) in stages:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=(
                "round,pad=0.04,"
                "rounding_size=0.10"
            ),
            linewidth=1.5,
            edgecolor="black",
            facecolor="white",
        )

        axis.add_patch(
            box
        )

        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for index in range(
        len(stages) - 1
    ):
        current = stages[index]
        following = stages[index + 1]

        start_x = (
            current[0]
            + current[2]
        )

        start_y = (
            current[1]
            + current[3] / 2
        )

        end_x = (
            following[0]
        )

        end_y = (
            following[1]
            + following[3] / 2
        )

        axis.annotate(
            "",
            xy=(
                end_x,
                end_y,
            ),
            xytext=(
                start_x,
                start_y,
            ),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.8,
            },
        )

    lower_stages = [
        (
            2.0,
            1.1,
            3.0,
            1.4,
            "External Baselines\nM2M100 and mBART-50",
        ),
        (
            6.4,
            1.1,
            3.0,
            1.4,
            "Qualitative Analysis\nImprovements and Regressions",
        ),
        (
            10.8,
            1.1,
            3.0,
            1.4,
            "Hindi Extension\nDirect and Pivot Translation",
        ),
    ]

    for (
        x,
        y,
        width,
        height,
        label,
    ) in lower_stages:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=(
                "round,pad=0.04,"
                "rounding_size=0.10"
            ),
            linewidth=1.5,
            edgecolor="black",
            facecolor="white",
        )

        axis.add_patch(
            box
        )

        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    axis.annotate(
        "",
        xy=(
            3.5,
            2.5,
        ),
        xytext=(
            13.3,
            4.4,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
            "connectionstyle": (
                "arc3,rad=0.18"
            ),
        },
    )

    axis.annotate(
        "",
        xy=(
            7.9,
            2.5,
        ),
        xytext=(
            14.0,
            4.4,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
        },
    )

    axis.annotate(
        "",
        xy=(
            12.3,
            2.5,
        ),
        xytext=(
            14.6,
            4.4,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
            "connectionstyle": (
                "arc3,rad=-0.18"
            ),
        },
    )

    axis.set_title(
        (
            "Extended Quality-Aware "
            "Pashto NMT Research Pipeline"
        ),
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "instructor_extended_pipeline.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_metrics_latex_table(
    metrics,
):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Standardized Pashto--English "
            r"evaluation across NLLB, LoRA, and "
            r"external multilingual baselines.}"
        ),
        r"\label{tab:extended-model-comparison}",
        r"\small",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        (
            r"\textbf{Model} & "
            r"\textbf{BLEU} & "
            r"\textbf{chrF++} & "
            r"\textbf{TER} & "
            r"\textbf{COMET} & "
            r"\textbf{Samples} & "
            r"\textbf{Setting} \\"
        ),
        r"\midrule",
    ]

    for _, row in metrics.iterrows():
        comet_value = (
            "--"
            if pd.isna(row["COMET"])
            else f"{row['COMET']:.4f}"
        )

        lines.append(
            (
                f"{latex_escape(row['model'])} & "
                f"{row['BLEU']:.2f} & "
                f"{row['chrF++']:.2f} & "
                f"{row['TER']:.2f} & "
                f"{comet_value} & "
                f"{int(row['samples'])} & "
                f"{latex_escape(row['setting'])} "
                r"\\"
            )
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\vspace{2pt}\parbox{0.95\textwidth}"
                r"{\footnotesize TER is lower-is-better. "
                r"M2M100 and mBART-50 are external "
                r"zero-shot baselines and were not "
                r"fine-tuned on the project corpus.}"
            ),
            r"\end{table*}",
        ]
    )

    output_path = (
        GENERATED_LATEX_DIRECTORY
        / "instructor_model_comparison_table.tex"
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


def create_qualitative_latex_table(
    examples,
):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Qualitative comparison of "
            r"baseline NLLB, original LoRA, and "
            r"semantic-filtered LoRA outputs.}"
        ),
        r"\label{tab:qualitative-comparison}",
        r"\footnotesize",
        (
            r"\begin{tabularx}{\textwidth}"
            r"{p{0.16\textwidth}X}"
        ),
        r"\toprule",
    ]

    for _, row in examples.iterrows():
        example_id = int(
            row["example_id"]
        )

        delta = float(
            row[
                "semantic_minus_baseline_chrf"
            ]
        )

        lines.extend(
            [
                (
                    r"\multicolumn{2}{l}{"
                    r"\textbf{Example "
                    f"{example_id}: "
                    f"{latex_escape(row['example_category'])}"
                    r"}} \\"
                ),
                (
                    r"Pashto source & "
                    f"{latex_escape(row['source'])} "
                    r"\\"
                ),
                (
                    r"Reference & "
                    f"{latex_escape(row['reference'])} "
                    r"\\"
                ),
                (
                    r"NLLB baseline & "
                    f"{latex_escape(row['baseline_prediction'])} "
                    r"\\"
                ),
                (
                    r"Original LoRA & "
                    f"{latex_escape(row['original_lora_prediction'])} "
                    r"\\"
                ),
                (
                    r"Semantic LoRA & "
                    f"{latex_escape(row['semantic_lora_prediction'])} "
                    r"\\"
                ),
                (
                    r"chrF++ change & "
                    f"{delta:+.2f} "
                    r"(Semantic LoRA minus baseline) \\"
                ),
                r"\midrule",
            ]
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
        ]
    )

    output_path = (
        GENERATED_LATEX_DIRECTORY
        / "instructor_qualitative_examples.tex"
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


def create_completion_report(
    metrics_path,
    examples_path,
    generated_files,
):
    report_lines = [
        "# Instructor Feedback Completion Report",
        "",
        "## Requested improvements",
        "",
        "### 1. Improve figures and diagrams",
        "",
        "Completed:",
        "",
        "- Informative end-to-end research pipeline diagram",
        "- BLEU and chrF++ model comparison figure",
        "- TER model comparison figure",
        "- COMET model comparison figure where scores are available",
        "",
        "### 2. Add qualitative translation examples",
        "",
        "Completed:",
        "",
        "- Baseline NLLB outputs",
        "- Original LoRA outputs",
        "- Semantic-filtered LoRA outputs",
        "- Automatically selected improvement candidates",
        "- Automatically selected regression candidates",
        "",
        "These examples still require careful bilingual interpretation.",
        "",
        "### 3. Compare with additional models",
        "",
        "Completed:",
        "",
        "- M2M100 418M zero-shot comparison",
        "- mBART-50 zero-shot comparison",
        "",
        "IndicTrans2 is documented as applicable only to the English-to-Hindi pivot stage.",
        "",
        "## Additional research improvements completed",
        "",
        "- BLEU, chrF++, TER, and COMET evaluation",
        "- Paired bootstrap significance analysis",
        "- Automatic translation diagnostics",
        "- Fixed source-grouped leak-free splits",
        "- Dataset checksums and environment records",
        "- Reproducible experiment manifests",
        "- Multi-seed experiment configurations",
        "- Successful LoRA smoke training and inference checks",
        "",
        "## Generated files",
        "",
        f"- Combined metrics: `{metrics_path}`",
        f"- Qualitative examples: `{examples_path}`",
    ]

    for generated_file in generated_files:
        if generated_file is not None:
            report_lines.append(
                f"- `{generated_file}`"
            )

    output_path = (
        REPORT_DIRECTORY
        / "instructor_feedback_completion.md"
    )

    output_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    return output_path


def main():
    OUTPUT_TABLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    GENERATED_LATEX_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("BUILDING INSTRUCTOR IMPROVEMENT ASSETS")
    print("=" * 80)

    metrics, metrics_path = (
        load_metrics()
    )

    examples, examples_path = (
        select_qualitative_examples()
    )

    pipeline_figure = (
        create_pipeline_figure()
    )

    quality_figure = (
        create_quality_figure(
            metrics
        )
    )

    ter_figure = (
        create_ter_figure(
            metrics
        )
    )

    comet_figure = (
        create_comet_figure(
            metrics
        )
    )

    metrics_latex = (
        create_metrics_latex_table(
            metrics
        )
    )

    qualitative_latex = (
        create_qualitative_latex_table(
            examples
        )
    )

    generated_files = [
        pipeline_figure,
        quality_figure,
        ter_figure,
        comet_figure,
        metrics_latex,
        qualitative_latex,
    ]

    completion_report = (
        create_completion_report(
            metrics_path,
            examples_path,
            generated_files,
        )
    )

    print()
    print("=" * 80)
    print("INSTRUCTOR IMPROVEMENT ASSETS COMPLETED")
    print("=" * 80)

    print()
    print(
        metrics[
            [
                "model",
                "BLEU",
                "chrF++",
                "TER",
                "COMET",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Metrics: {metrics_path}")
    print(f"Examples: {examples_path}")
    print(f"Report: {completion_report}")

    for generated_file in generated_files:
        if generated_file is not None:
            print(
                f"Generated: {generated_file}"
            )


if __name__ == "__main__":
    main()
