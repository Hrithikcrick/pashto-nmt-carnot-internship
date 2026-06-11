# Next 2 Weeks Plan

## Week 3: Dataset Quality and Gold Test Set

Tasks:

1. Create manually verified Pashto-English-Hindi test set of 50 to 100 examples.
2. Use columns: id, domain, pashto, english_reference, hindi_reference, difficulty.
3. Cover domains like daily conversation, education, healthcare, government, travel, news, emergency, and technology.
4. Apply semantic similarity filtering using LaBSE, LASER, or multilingual sentence embeddings.
5. Try thresholds: 0.65, 0.70, 0.75, and 0.80.
6. Create filtered dataset versions.
7. Prepare dataset quality report with examples of noisy pairs removed.

Expected outputs:

- gold_test_100_template.csv
- semantic_filter.py
- filtered dataset versions
- dataset quality report

## Week 4: Fine-Tuning and Evaluation

Tasks:

1. Fine-tune NLLB on 10k cleaned Pashto-English pairs.
2. Fine-tune NLLB on 10k semantic-filtered pairs.
3. If GPU/time allows, fine-tune on 50k filtered pairs.
4. Use LoRA if full fine-tuning is heavy.
5. Compare baseline and fine-tuned models.
6. Evaluate using BLEU, chrF, inference time, and manual scoring.

Expected outputs:

- fine-tuning script
- trained adapter/model checkpoint
- baseline vs fine-tuned prediction CSV
- BLEU/chrF comparison table
- manual evaluation sheet
- Week 4 report
