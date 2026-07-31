import re
import shutil
from pathlib import Path

import pandas as pd


PAPER_PATH = Path("paper/main.tex")
OUTPUT_PATH = Path("paper/main_instructor_revised.tex")
BACKUP_PATH = Path("paper/main_before_instructor_revision.tex")

METRICS_PATH = Path(
    "outputs/tables/research_automatic_metrics.csv"
)

SIGNIFICANCE_PATH = Path(
    "outputs/tables/research_metric_significance.csv"
)

EXTERNAL_METRICS_PATH = Path(
    "outputs/tables/external_baseline_metrics.csv"
)

REQUIRED_ASSETS = [
    Path(
        "paper/generated/"
        "instructor_model_comparison_table.tex"
    ),
    Path(
        "paper/generated/"
        "instructor_qualitative_examples.tex"
    ),
    Path(
        "paper/figures/"
        "instructor_extended_pipeline.png"
    ),
    Path(
        "paper/figures/"
        "instructor_model_quality_comparison.png"
    ),
    Path(
        "paper/figures/"
        "instructor_model_ter_comparison.png"
    ),
]


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def get_model_metrics(dataframe, model_name):
    selected = dataframe[
        dataframe["model"] == model_name
    ]

    if selected.empty:
        raise ValueError(
            f"Model not found in metrics: {model_name}"
        )

    row = selected.iloc[0]

    return {
        "bleu": float(row["BLEU"]),
        "chrf": float(row["chrF++"]),
        "ter": float(row["TER"]),
        "comet": float(row["COMET"]),
        "samples": int(row["samples"]),
    }


def get_significance(
    dataframe,
    metric_name,
):
    if dataframe is None:
        return {
            "difference": None,
            "p_value": None,
            "significant": False,
        }

    selected = dataframe[
        (
            dataframe["metric"]
            == metric_name
        )
        &
        (
            dataframe[
                "comparison_model"
            ]
            == "Semantic LoRA"
        )
    ]

    if selected.empty:
        return {
            "difference": None,
            "p_value": None,
            "significant": False,
        }

    row = selected.iloc[0]

    significant_value = str(
        row["significant_at_0.05"]
    ).strip().lower()

    return {
        "difference": float(
            row[
                "mean_difference_model_minus_baseline"
            ]
        ),
        "p_value": float(
            row["p_value"]
        ),
        "significant": (
            significant_value == "true"
        ),
    }


def replace_section(
    text,
    current_title,
    following_title,
    replacement,
):
    pattern = (
        rf"\\section\{{"
        rf"{re.escape(current_title)}"
        rf"\}}"
        rf".*?"
        rf"(?="
        rf"\\section\{{"
        rf"{re.escape(following_title)}"
        rf"\}}"
        rf")"
    )

    updated, count = re.subn(
        pattern,
        lambda match: replacement.strip() + "\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )

    if count != 1:
        raise ValueError(
            f"Could not replace section: "
            f"{current_title}. "
            f"Matches found: {count}"
        )

    return updated


def replace_conclusion(
    text,
    replacement,
):
    pattern = (
        r"\\section\{Conclusion\}"
        r".*?"
        r"(?=\\section\*\{Repository\})"
    )

    updated, count = re.subn(
        pattern,
        lambda match: replacement.strip() + "\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )

    if count != 1:
        raise ValueError(
            "Could not replace Conclusion section."
        )

    return updated


def add_bibliography_entries(text):
    entries = r"""
\bibitem{m2m100}
A. Fan, S. Bhosale, H. Schwenk, Z. Ma, A. El-Kishky,
S. Goyal, M. Baines, O. Celebi, G. Wenzek,
V. Chaudhary, N. Goyal, T. Birch, V. Liptchinsky,
S. Edunov, E. Grave, M. Auli, and A. Joulin,
``Beyond English-Centric Multilingual Machine Translation,''
\textit{Journal of Machine Learning Research},
vol. 22, no. 107, pp. 1--48, 2021.

\bibitem{mbart50}
Y. Tang, C. Tran, X. Li, P.-J. Chen, N. Goyal,
V. Chaudhary, J. Gu, and A. Fan,
``Multilingual Translation with Extensible Multilingual
Pretraining and Finetuning,''
\textit{arXiv preprint arXiv:2008.00401}, 2020.

\bibitem{comet}
R. Rei, C. Stewart, A. C. Farinha, and A. Lavie,
``COMET: A Neural Framework for MT Evaluation,''
in \textit{Proceedings of EMNLP}, 2020,
pp. 2685--2702.
"""

    if r"\bibitem{m2m100}" not in text:
        text = text.replace(
            r"\end{thebibliography}",
            entries.strip()
            + "\n\n"
            + r"\end{thebibliography}",
            1,
        )

    return text


def main():
    require_file(PAPER_PATH)
    require_file(METRICS_PATH)
    require_file(EXTERNAL_METRICS_PATH)

    for asset in REQUIRED_ASSETS:
        require_file(asset)

    metrics_dataframe = pd.read_csv(
        METRICS_PATH,
        encoding="utf-8-sig",
    )

    baseline = get_model_metrics(
        metrics_dataframe,
        "Baseline NLLB",
    )

    original_lora = get_model_metrics(
        metrics_dataframe,
        "Original LoRA",
    )

    semantic_lora = get_model_metrics(
        metrics_dataframe,
        "Semantic LoRA",
    )

    significance_dataframe = None

    if SIGNIFICANCE_PATH.exists():
        significance_dataframe = pd.read_csv(
            SIGNIFICANCE_PATH,
            encoding="utf-8-sig",
        )

    bleu_significance = get_significance(
        significance_dataframe,
        "BLEU",
    )

    chrf_significance = get_significance(
        significance_dataframe,
        "chrF++",
    )

    ter_significance = get_significance(
        significance_dataframe,
        "TER",
    )

    comet_significance = get_significance(
        significance_dataframe,
        "COMET",
    )

    text = PAPER_PATH.read_text(
        encoding="utf-8-sig"
    )

    shutil.copy2(
        PAPER_PATH,
        BACKUP_PATH,
    )

    if (
        r"\usepackage{tabularx}"
        not in text
    ):
        text = text.replace(
            r"\usepackage{array}",
            (
                r"\usepackage{array}"
                "\n"
                r"\usepackage{tabularx}"
            ),
            1,
        )

    revised_title = r"""
\title{
Quality-Aware Low-Resource Pashto Neural Machine Translation:
Semantic Filtering, LoRA Adaptation, Multilingual Baselines,
and Pivot-Based Hindi Translation
}
"""

    text, title_count = re.subn(
        (
            r"\\title\{"
            r".*?"
            r"\}"
            r"\s*"
            r"(?=\\author)"
        ),
        lambda match: revised_title.strip()
        + "\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )

    if title_count != 1:
        raise ValueError(
            "Could not replace the paper title."
        )

    abstract = rf"""
\begin{{abstract}}
Low-resource neural machine translation is constrained not
only by corpus size but also by duplication, weak alignment,
domain mismatch, and unreliable evaluation. This paper
presents a quality-aware Pashto--English translation pipeline
using NLLB-200, semantic sentence-pair filtering, and LoRA
parameter-efficient adaptation, together with direct and
pivot-based Hindi extensions. Starting from 93,498 raw
Pashto--English sentence pairs, rule-based cleaning retained
90,978 pairs. Standardized evaluation was conducted on the
same {baseline['samples']} Pashto--English examples for the
pretrained NLLB baseline, original LoRA, semantic-filtered
LoRA, M2M100, and mBART-50. The pretrained NLLB model
obtained BLEU {baseline['bleu']:.2f}, chrF++ {baseline['chrf']:.2f},
TER {baseline['ter']:.2f}, and COMET {baseline['comet']:.4f}.
Original LoRA obtained BLEU {original_lora['bleu']:.2f},
chrF++ {original_lora['chrf']:.2f}, TER
{original_lora['ter']:.2f}, and COMET
{original_lora['comet']:.4f}. Semantic-filtered LoRA achieved
the highest BLEU and chrF++ values among the NLLB variants
at {semantic_lora['bleu']:.2f} and {semantic_lora['chrf']:.2f},
respectively, while its COMET score of
{semantic_lora['comet']:.4f} remained approximately tied
with the pretrained baseline. The revised experimental
package additionally introduces informative pipeline and
metric visualizations, external multilingual baselines,
qualitative translation examples, paired bootstrap analysis,
automatic diagnostics, source-grouped leakage-controlled
splits, dataset checksums, experiment manifests, and
multi-seed configurations. The results support semantic
filtering as a useful data-selection strategy for LoRA
adaptation, but do not establish broad statistically
significant superiority over the strong multilingual baseline.
\end{{abstract}}
"""

    text, abstract_count = re.subn(
        (
            r"\\begin\{abstract\}"
            r".*?"
            r"\\end\{abstract\}"
        ),
        lambda match: abstract.strip(),
        text,
        count=1,
        flags=re.DOTALL,
    )

    if abstract_count != 1:
        raise ValueError(
            "Could not replace the abstract."
        )

    contributions = r"""
\section{Research Contributions}

The revised study makes the following contributions:

\begin{enumerate}
    \item A complete low-resource Pashto--English NMT
    pipeline based on NLLB-200 and LoRA.
    \item Rule-based cleaning of a 93,498-pair corpus,
    retaining 90,978 parallel sentence pairs.
    \item Semantic similarity filtering and filtered
    training subsets containing 8k, 4k, and 2k pairs.
    \item Standardized comparison of pretrained NLLB,
    original LoRA, and semantic-filtered LoRA on the
    same test instances.
    \item External zero-shot comparisons with M2M100
    and mBART-50 \cite{m2m100,mbart50}.
    \item Evaluation using BLEU, chrF++, TER, and
    COMET \cite{comet}.
    \item Paired bootstrap confidence intervals and
    significance analysis.
    \item Automatic diagnostics for number mismatch,
    output length, repetition, empty output, and exact
    reference matching.
    \item Informative research-pipeline and
    model-comparison figures.
    \item Qualitative baseline-versus-LoRA improvement
    and regression candidates.
    \item Fixed source-grouped splits with zero
    source-level and exact-pair leakage.
    \item Dataset hashes, environment records, training
    manifests, and reproducible experiment configurations.
    \item Controlled configurations for seeds 13, 42,
    and 2026.
    \item A successfully validated LoRA smoke-training
    and adapter-loading pipeline.
    \item Direct and pivot Hindi analysis, with
    IndicTrans2 retained only for the applicable
    English-to-Hindi pivot stage.
\end{enumerate}
"""

    text = replace_section(
        text,
        "Research Contributions",
        "Related Work",
        contributions,
    )

    baseline_section = rf"""
\section{{Baseline Evaluation}}

\subsection{{Standardized Automatic Metrics}}

The earlier preliminary baseline values were replaced by a
standardized evaluation computed from the same prediction
and reference files used for all NLLB-based systems.

\begin{{table}}[H]
\centering
\caption{{Standardized NLLB baseline evaluation.}}
\label{{tab:standardized-baseline}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Metric}} & \textbf{{Score}} \\
\midrule
BLEU & {baseline['bleu']:.2f} \\
chrF++ & {baseline['chrf']:.2f} \\
TER & {baseline['ter']:.2f} \\
COMET & {baseline['comet']:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

BLEU and chrF++ are higher-is-better metrics, whereas TER
is lower-is-better. COMET provides a complementary neural
evaluation signal. All reported systems use the same
{baseline['samples']}-sentence evaluation set.

\subsection{{Inference-Time Analysis}}

The previously measured CPU inference-time comparison is
retained as an engineering observation. It is not treated as
a translation-quality metric.

\begin{{table}}[H]
\centering
\caption{{Average CPU inference time.}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Translation Direction}} &
\textbf{{Time per Sentence}} \\
\midrule
Pashto-to-English & 4.46 sec \\
Pashto-to-Hindi Direct & 5.11 sec \\
Pashto-to-English-to-Hindi Pivot & 4.76 sec \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    text = replace_section(
        text,
        "Baseline Evaluation",
        "Dataset Quality Integration",
        baseline_section,
    )

    qualitative_framework = r"""
\section{Qualitative Error Analysis Framework}

Automatic metrics cannot fully explain omissions,
hallucinations, named-entity errors, number errors,
tense changes, literal translation, or poor fluency.
The project therefore defines a qualitative analysis
framework covering:

\begin{itemize}
    \item missing or omitted information;
    \item incorrect or changed meaning;
    \item incomplete translation;
    \item named-entity and number errors;
    \item tense and grammatical mismatch;
    \item literal or unnatural translation;
    \item hallucinated content;
    \item untranslated text and repetition.
\end{itemize}

A structured evaluation template was prepared for future
bilingual annotation. However, a formal human evaluation
has not yet been completed. The examples reported in this
paper are automatically selected comparison candidates and
must not be interpreted as human preference judgments.
"""

    text = replace_section(
        text,
        "Manual Error Analysis",
        "LoRA Fine-Tuning",
        qualitative_framework,
    )

    results_section = rf"""
\section{{Fine-Tuning Results}}

All NLLB-based systems were re-evaluated using the same
source sentences, English references, decoding outputs, and
metric implementations.

\begin{{table}}[H]
\centering
\caption{{Standardized NLLB and LoRA results.}}
\label{{tab:nllb-lora-results}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
\textbf{{System}} &
\textbf{{BLEU}} &
\textbf{{chrF++}} &
\textbf{{TER}} &
\textbf{{COMET}} \\
\midrule
Baseline NLLB &
{baseline['bleu']:.2f} &
{baseline['chrf']:.2f} &
{baseline['ter']:.2f} &
{baseline['comet']:.4f} \\
Original LoRA &
{original_lora['bleu']:.2f} &
{original_lora['chrf']:.2f} &
{original_lora['ter']:.2f} &
{original_lora['comet']:.4f} \\
Semantic LoRA &
{semantic_lora['bleu']:.2f} &
{semantic_lora['chrf']:.2f} &
{semantic_lora['ter']:.2f} &
{semantic_lora['comet']:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

Semantic-filtered LoRA achieved the highest BLEU and
chrF++ point estimates among the NLLB variants. Relative
to the pretrained baseline, it increased corpus BLEU by
{semantic_lora['bleu'] - baseline['bleu']:+.2f} and chrF++
by {semantic_lora['chrf'] - baseline['chrf']:+.2f}.
However, COMET changed by only
{semantic_lora['comet'] - baseline['comet']:+.6f}, which is
effectively a tie. Its TER was also higher than the baseline.

Original LoRA improved chrF++ but produced a slightly lower
COMET score and a higher TER. The results therefore indicate
modest and metric-dependent improvement rather than uniform
superiority across all evaluation measures.
"""

    text = replace_section(
        text,
        "Fine-Tuning Results",
        "Improved LoRA Experiments",
        results_section,
    )

    indictrans_section = r"""
\section{IndicTrans2 Applicability and Status}

IndicTrans2 was considered for the English-to-Hindi stage
of the pivot translation pipeline:

\begin{center}
Pashto $\rightarrow$ English using NLLB/LoRA
$\rightarrow$ Hindi using IndicTrans2.
\end{center}

IndicTrans2 is not used as a direct Pashto-to-English
baseline in this study. Native Windows installation of the
official processing toolkit was not completed because its
Cython extension required an unavailable compiler toolchain.
The quantitative Pashto-to-English architecture comparison
was therefore conducted using NLLB, M2M100, and mBART-50.

Until IndicTrans2 outputs are generated in a supported
Linux or Colab environment, it is reported only as an
applicable future English-to-Hindi pivot experiment.
No IndicTrans2 quality result is claimed in this paper.
"""

    text = replace_section(
        text,
        "IndicTrans2 Exploration",
        "Ablation Study",
        indictrans_section,
    )

    limitations_section = r"""
\section{Limitations}

The present study has the following limitations:

\begin{itemize}
    \item The standardized evaluation currently contains
    100 shared Pashto--English examples.
    \item Full three-seed training is not yet complete;
    seed configurations have been prepared but must still
    be executed.
    \item M2M100 and mBART-50 are evaluated zero-shot and
    have not yet received equal-budget LoRA adaptation.
    \item The qualitative examples are automatically
    selected and have not been validated by multiple
    bilingual annotators.
    \item BLEU, chrF++, TER, and COMET provide
    complementary but imperfect quality estimates.
    \item Hindi evaluation lacks a gold Hindi reference set.
    \item IndicTrans2 inference was not completed in the
    native Windows environment.
    \item CPU-only training limits the number of model,
    data-size, seed, and hyperparameter experiments.
    \item The current results do not establish
    state-of-the-art performance.
\end{itemize}
"""

    text = replace_section(
        text,
        "Limitations",
        "Remaining Work and Extended Results",
        limitations_section,
    )

    def format_significance(result, digits):
        if result["difference"] is None:
            return "not available"

        return (
            f"difference {result['difference']:+.{digits}f}, "
            f"$p={result['p_value']:.4f}$"
        )

    extended_section = rf"""
\section{{Extended Evaluation and Instructor-Requested Improvements}}

\subsection{{Informative Research Pipeline}}

The original screenshot-style summary figures were replaced
by an informative end-to-end research diagram connecting
corpus preparation, leakage control, semantic filtering,
LoRA adaptation, external baselines, automatic metrics,
qualitative analysis, and Hindi translation.

\begin{{figure*}}[t]
\centering
\includegraphics[
    width=0.98\textwidth
]{{figures/instructor_extended_pipeline.png}}
\caption{{Extended quality-aware Pashto NMT research
pipeline.}}
\label{{fig:extended-pipeline}}
\end{{figure*}}

\subsection{{Multilingual Architecture Comparison}}

To place NLLB in a broader multilingual context, M2M100
and mBART-50 were evaluated zero-shot on the same
Pashto--English instances \cite{{m2m100,mbart50}}.

\input{{generated/instructor_model_comparison_table.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[
    width=0.92\textwidth
]{{figures/instructor_model_quality_comparison.png}}
\caption{{BLEU and chrF++ comparison across the evaluated
NLLB, LoRA, M2M100, and mBART-50 systems.}}
\label{{fig:quality-comparison}}
\end{{figure*}}

\begin{{figure}}[t]
\centering
\includegraphics[
    width=\columnwidth
]{{figures/instructor_model_ter_comparison.png}}
\caption{{Translation Edit Rate comparison. Lower values
are better.}}
\label{{fig:ter-comparison}}
\end{{figure}}

\IfFileExists{{
    figures/instructor_model_comet_comparison.png
}}{{
\begin{{figure}}[t]
\centering
\includegraphics[
    width=\columnwidth
]{{figures/instructor_model_comet_comparison.png}}
\caption{{COMET neural evaluation for systems with
completed COMET scores.}}
\label{{fig:comet-comparison}}
\end{{figure}}
}}{{}}

\subsection{{Qualitative Translation Examples}}

The instructor-requested qualitative examples compare the
English reference, pretrained NLLB output, original LoRA
output, and semantic-filtered LoRA output. Improvement and
regression candidates were selected using sentence-level
chrF++ differences. They illustrate model behaviour but do
not replace bilingual human evaluation.

\input{{generated/instructor_qualitative_examples.tex}}

\subsection{{Statistical Analysis}}

For Semantic LoRA relative to the NLLB baseline, the
implemented paired sentence-level bootstrap analysis reported:

\begin{{itemize}}
    \item BLEU:
    {format_significance(bleu_significance, 4)};
    \item chrF++:
    {format_significance(chrf_significance, 4)};
    \item TER:
    {format_significance(ter_significance, 4)};
    \item COMET:
    {format_significance(comet_significance, 6)}.
\end{{itemize}}

Only the sentence-level BLEU comparison crossed the
0.05 threshold in the pilot analysis. chrF++, TER, and COMET
did not provide statistically significant evidence of general
superiority. Corpus-level metric differences and averages of
sentence-level metric differences are reported as distinct
quantities.

\subsection{{Additional Completed Engineering Work}}

Beyond the instructor's requested figures, qualitative
examples, and model comparisons, the extended project also
completed:

\begin{{itemize}}
    \item standardized BLEU, chrF++, TER, and COMET
    evaluation;
    \item pairwise wins, losses, and ties;
    \item automatic translation diagnostics;
    \item fixed source-grouped train, validation, and
    test splits;
    \item zero exact-pair and Pashto-source overlap;
    \item SHA-256 dataset checksums;
    \item package and runtime environment records;
    \item machine-readable training manifests;
    \item configurations for seeds 13, 42, and 2026;
    \item a successful CPU LoRA smoke-training run;
    \item successful adapter saving, reloading, and
    translation-generation checks.
\end{{itemize}}

The smoke experiment is an engineering validation and is
not included as a final translation-quality result.
"""

    text = replace_section(
        text,
        "Remaining Work and Extended Results",
        "Discussion",
        extended_section,
    )

    discussion_section = r"""
\section{Discussion}

The expanded evaluation changes the interpretation of the
initial experiments. Semantic-filtered LoRA obtains the best
BLEU and chrF++ point estimates among the NLLB variants,
but its almost unchanged COMET score indicates that the
improvement is not uniformly reflected by a neural semantic
metric. Its higher TER also shows that better character-level
overlap does not necessarily imply fewer reference edits.

The M2M100 and mBART-50 comparisons reduce dependence on
a single multilingual architecture. However, because these
systems are evaluated zero-shot, the comparison measures
pretrained transfer quality rather than equal-budget
adaptation. A stronger experiment should apply the same
data size, LoRA configuration, training steps, decoding
settings, and seeds to every compatible model.

The qualitative examples make the metric results easier to
interpret by exposing both improvement and regression
candidates. They must be reviewed by bilingual annotators
before being described as improvements in human-perceived
adequacy or fluency.

The present contribution is therefore best described as a
reproducible empirical analysis of semantic filtering for
parameter-efficient Pashto--English adaptation. The current
evidence does not justify claims of state-of-the-art
performance or universal statistically significant
superiority.
"""

    text = replace_section(
        text,
        "Discussion",
        "Conclusion",
        discussion_section,
    )

    conclusion_section = rf"""
\section{{Conclusion}}

This paper presented a quality-aware and reproducible
pipeline for low-resource Pashto neural machine translation.
Starting from 93,498 raw Pashto--English sentence pairs,
rule-based cleaning retained 90,978 pairs, after which
semantic filtering and LoRA adaptation were applied.

On the standardized {baseline['samples']}-sentence evaluation
set, pretrained NLLB achieved BLEU {baseline['bleu']:.2f},
chrF++ {baseline['chrf']:.2f}, TER {baseline['ter']:.2f},
and COMET {baseline['comet']:.4f}. Original LoRA achieved
BLEU {original_lora['bleu']:.2f} and chrF++
{original_lora['chrf']:.2f}, while semantic-filtered LoRA
achieved the strongest NLLB-variant BLEU and chrF++ scores
of {semantic_lora['bleu']:.2f} and
{semantic_lora['chrf']:.2f}. Its COMET score of
{semantic_lora['comet']:.4f} remained effectively tied with
the pretrained baseline.

The revised study additionally contributes an informative
pipeline diagram, model-level quality figures, qualitative
translation examples, M2M100 and mBART-50 comparisons,
paired bootstrap analysis, automatic diagnostics,
leakage-controlled splits, dataset hashes, training
manifests, and multi-seed experiment configurations.
IndicTrans2 is retained only for the applicable future
English-to-Hindi pivot stage.

Future work should complete the full multi-seed runs,
expand evaluation to a substantially larger independent
test set, apply equal-budget adaptation to a second model
family, and conduct blinded bilingual human evaluation.
These steps are required before making strong claims about
robustness, significance, and generalization.
"""

    text = replace_conclusion(
        text,
        conclusion_section,
    )

    text = add_bibliography_entries(
        text
    )

    screenshot_pattern = (
        r"\\safeimage"
        r"\{Screenshot[^}]*\}"
        r"\{[^}]*\}"
        r"\{.*?\}"
        r"\s*"
    )

    text, removed_screenshots = re.subn(
        screenshot_pattern,
        "",
        text,
        flags=re.DOTALL,
    )

    OUTPUT_PATH.write_text(
        text,
        encoding="utf-8",
    )

    stale_patterns = [
        "17.97",
        "36.49",
        "38--39",
        "38–39",
    ]

    stale_found = [
        pattern
        for pattern in stale_patterns
        if pattern in text
    ]

    print("=" * 80)
    print("INSTRUCTOR PAPER REVISION COMPLETED")
    print("=" * 80)
    print()
    print(f"Original paper: {PAPER_PATH}")
    print(f"Backup:         {BACKUP_PATH}")
    print(f"Revised paper:  {OUTPUT_PATH}")
    print(
        f"Screenshot summary figures removed: "
        f"{removed_screenshots}"
    )

    if stale_found:
        print()
        print(
            "WARNING: Old result text still found:"
        )

        for pattern in stale_found:
            print(f"  {pattern}")

    else:
        print()
        print(
            "No outdated 17.97, 36.49, "
            "or 38--39 result text remains."
        )

    print()
    print(
        "Compile paper/main_instructor_revised.tex"
    )


if __name__ == "__main__":
    main()
