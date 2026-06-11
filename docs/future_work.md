# Future Work

In the upcoming weeks, the project will move from baseline evaluation to research-level improvement.

The next stage will focus on creating a manually verified Pashto-English-Hindi test set, improving dataset quality using semantic similarity filtering, and fine-tuning the NLLB model on cleaned Pashto-English data.

The work will include comparison between baseline NLLB and fine-tuned NLLB using BLEU, chrF, inference time, and manual evaluation.

For Hindi translation, three approaches will be compared:

1. Direct Pashto to Hindi translation.
2. Pashto to English to Hindi pivot translation.
3. Fine-tuned Pashto to English translation followed by IndicTrans2 English to Hindi translation.

The final output will include result tables, error analysis, GitHub repository, final report, and Streamlit demo.
