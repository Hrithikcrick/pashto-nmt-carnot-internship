import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("docs", exist_ok=True)

cleaning = pd.DataFrame({
    "stage": ["Raw rows", "After duplicate removal", "Final clean rows"],
    "rows": [93498, 93418, 90978]
})
cleaning.to_csv("outputs/tables/dataset_cleaning_summary.csv", index=False)

plt.figure(figsize=(9, 5))
bars = plt.bar(cleaning["stage"], cleaning["rows"])
plt.title("Dataset Cleaning Summary")
plt.xlabel("Cleaning stage")
plt.ylabel("Number of sentence pairs")
plt.xticks(rotation=20, ha="right")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{int(y):,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/dataset_cleaning_summary.png", dpi=300)
plt.close()

split = pd.DataFrame({
    "split": ["Training", "Validation", "Test"],
    "rows": [72782, 9098, 9098]
})
split.to_csv("outputs/tables/train_val_test_split.csv", index=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(split["split"], split["rows"])
plt.title("Train-Validation-Test Split")
plt.xlabel("Dataset split")
plt.ylabel("Number of rows")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{int(y):,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/train_val_test_split.png", dpi=300)
plt.close()

metrics = pd.DataFrame({
    "metric": ["BLEU", "chrF"],
    "score": [17.97, 36.49]
})
metrics.to_csv("outputs/tables/baseline_metrics.csv", index=False)

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics["metric"], metrics["score"])
plt.title("Baseline NLLB Pashto-English Evaluation")
plt.xlabel("Metric")
plt.ylabel("Score")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/baseline_metrics.png", dpi=300)
plt.close()

time_df = pd.DataFrame({
    "translation_direction": [
        "Pashto-English",
        "Pashto-Hindi Direct",
        "Pashto-English-Hindi Pivot"
    ],
    "avg_time_sec": [4.46, 5.11, 4.76]
})
time_df.to_csv("outputs/tables/inference_time.csv", index=False)

plt.figure(figsize=(9, 5))
bars = plt.bar(time_df["translation_direction"], time_df["avg_time_sec"])
plt.title("Average Inference Time on CPU")
plt.xlabel("Translation direction")
plt.ylabel("Average time per sentence (seconds)")
plt.xticks(rotation=20, ha="right")
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.2f}s", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/figures/inference_time.png", dpi=300)
plt.close()

progress = pd.DataFrame({
    "task": [
        "Baseline implementation",
        "Dataset cleaning",
        "Train/validation/test split",
        "Baseline evaluation",
        "Manual gold test set",
        "Semantic filtering",
        "NLLB fine-tuning",
        "IndicTrans2 Hindi pipeline",
        "Final Streamlit demo"
    ],
    "progress_percent": [100, 100, 100, 100, 0, 0, 0, 0, 0]
})
progress.to_csv("outputs/tables/research_progress.csv", index=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(progress["task"], progress["progress_percent"])
plt.title("Research Project Progress Dashboard")
plt.xlabel("Completion percentage")
plt.ylabel("Task")
plt.xlim(0, 100)
for bar in bars:
    x = bar.get_width()
    plt.text(x + 1, bar.get_y() + bar.get_height()/2, f"{int(x)}%", va="center")
plt.tight_layout()
plt.savefig("outputs/figures/research_progress_dashboard.png", dpi=300)
plt.close()

comparison = pd.DataFrame({
    "experiment": [
        "E0",
        "E1",
        "E2",
        "E3",
        "E4"
    ],
    "model": [
        "Baseline NLLB 600M",
        "NLLB LoRA",
        "NLLB LoRA",
        "NLLB LoRA",
        "NLLB LoRA + Back Translation"
    ],
    "dataset": [
        "No fine-tuning",
        "10k clean pairs",
        "10k semantic-filtered pairs",
        "50k semantic-filtered pairs",
        "50k filtered + synthetic pairs"
    ],
    "bleu": [17.97, "", "", "", ""],
    "chrf": [36.49, "", "", "", ""],
    "avg_time_sec": [4.46, "", "", "", ""],
    "status": ["completed", "planned", "planned", "planned", "planned"]
})
comparison.to_csv("outputs/tables/model_comparison_plan.csv", index=False)

manual = []
for i in range(1, 51):
    manual.append({
        "id": i,
        "pashto": "",
        "reference_english": "",
        "model_english": "",
        "direct_hindi": "",
        "pivot_hindi": "",
        "meaning_score_1_to_5": "",
        "grammar_score_1_to_5": "",
        "fluency_score_1_to_5": "",
        "completeness_score_1_to_5": "",
        "hindi_naturalness_1_to_5": "",
        "error_type": "",
        "remarks": ""
    })

manual_df = pd.DataFrame(manual)
manual_df.to_csv("outputs/tables/manual_evaluation_template.csv", index=False, encoding="utf-8-sig")

report = """# Week 2 Research Analysis

## Summary

This week, the Pashto Neural Machine Translation project was moved from a simple baseline demo to a research-level dataset and evaluation foundation.

The current system uses the pretrained `facebook/nllb-200-distilled-600M` model for:

1. Pashto to English translation.
2. Direct Pashto to Hindi translation.
3. Pashto to English to Hindi pivot-based translation.

## Dataset Cleaning Result

The WMT20 Pashto-English dataset was cleaned using duplicate removal, URL removal, HTML cleaning, sentence length filtering, script validation, and length ratio filtering.

| Stage | Rows |
|---|---:|
| Raw rows | 93,498 |
| After duplicate removal | 93,418 |
| Final clean rows | 90,978 |
| Removed noisy rows | 2,440 |

![Dataset Cleaning Summary](../outputs/figures/dataset_cleaning_summary.png)

## Train-Validation-Test Split

| Split | Rows |
|---|---:|
| Training | 72,782 |
| Validation | 9,098 |
| Test | 9,098 |

![Train Validation Test Split](../outputs/figures/train_val_test_split.png)

## Baseline Evaluation

The baseline NLLB 600M model was evaluated on 100 Pashto-English test samples.

| Model | Direction | Samples | BLEU | chrF |
|---|---|---:|---:|---:|
| NLLB 600M | Pashto-English | 100 | 17.97 | 36.49 |

![Baseline Metrics](../outputs/figures/baseline_metrics.png)

## Inference Time Analysis

| Translation Direction | Average Time |
|---|---:|
| Pashto-English | 4.46 sec |
| Pashto-Hindi Direct | 5.11 sec |
| Pashto-English-Hindi Pivot | 4.76 sec |

![Inference Time](../outputs/figures/inference_time.png)

## Research Progress

![Research Progress Dashboard](../outputs/figures/research_progress_dashboard.png)

## Key Observations

1. The pretrained NLLB baseline can generate meaningful Pashto-English translations, but the quality is still moderate.
2. The BLEU score of 17.97 and chrF score of 36.49 show that fine-tuning is required.
3. The WMT20 dataset contains noisy and mismatched sentence pairs, so semantic filtering is important before fine-tuning.
4. Direct Pashto-to-Hindi translation is weaker in some cases compared to pivot-based translation.
5. CPU inference is slow, so GPU or optimized inference will be useful for future experiments.

## Next Research Steps

1. Create manually verified Pashto-English-Hindi gold test set.
2. Apply semantic similarity filtering using LaBSE or multilingual sentence embeddings.
3. Fine-tune NLLB on 10k clean pairs.
4. Fine-tune NLLB on 10k semantic-filtered pairs.
5. Fine-tune on 50k filtered pairs if resources allow.
6. Compare baseline and fine-tuned models using BLEU, chrF, inference time, and human scoring.
7. Add IndicTrans2 for English-to-Hindi translation.
8. Compare direct Hindi, pivot Hindi, and fine-tuned NLLB + IndicTrans2 pipeline.
"""

Path("docs/week2_research_analysis.md").write_text(report, encoding="utf-8")

readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")

extra = """
## Week 2 Analysis Dashboard

The Week 2 research analysis includes dataset cleaning summary, train-validation-test split, baseline BLEU/chrF score, inference time analysis, and research progress dashboard.

### Analysis Figures

![Dataset Cleaning Summary](outputs/figures/dataset_cleaning_summary.png)

![Train Validation Test Split](outputs/figures/train_val_test_split.png)

![Baseline Metrics](outputs/figures/baseline_metrics.png)

![Inference Time](outputs/figures/inference_time.png)

![Research Progress Dashboard](outputs/figures/research_progress_dashboard.png)

### Research Tables

- `outputs/tables/dataset_cleaning_summary.csv`
- `outputs/tables/train_val_test_split.csv`
- `outputs/tables/baseline_metrics.csv`
- `outputs/tables/inference_time.csv`
- `outputs/tables/model_comparison_plan.csv`
- `outputs/tables/manual_evaluation_template.csv`
"""

if "## Week 2 Analysis Dashboard" not in readme:
    readme_path.write_text(readme + "\n" + extra, encoding="utf-8")

print("Week 2 analysis graphs, tables, and markdown report created successfully.")
