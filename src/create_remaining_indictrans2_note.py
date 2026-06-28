from pathlib import Path
import os

os.makedirs("docs", exist_ok=True)

text = """# IndicTrans2 Exploration for English-to-Hindi Stage

This document records the remaining research plan for improving the Hindi stage of the Pashto-English-Hindi translation pipeline.

## Current Hindi Translation Setup

The current system supports two Hindi translation approaches:

1. Direct Pashto-to-Hindi translation using NLLB.
2. Pivot Pashto-to-English-to-Hindi translation using NLLB.

The direct-vs-pivot comparison file has already been generated:

`outputs/tables/remaining_direct_vs_pivot_hindi.csv`

## Current Pipeline

Pashto -> Hindi using NLLB

Pashto -> English using NLLB  
English -> Hindi using NLLB

## Planned Improved Pipeline

Pashto -> English using fine-tuned NLLB  
English -> Hindi using IndicTrans2

## Why IndicTrans2

IndicTrans2 is specifically designed for Indian language translation. It may produce more natural and fluent Hindi output compared with direct multilingual NLLB decoding.

## Evaluation Plan

1. Select the same Pashto examples from the gold test candidate set.
2. Generate English output using the fine-tuned Pashto-to-English model.
3. Translate English output into Hindi using IndicTrans2.
4. Compare three outputs:
   - Direct Pashto-to-Hindi using NLLB
   - Pivot Pashto-English-Hindi using NLLB
   - Pivot Pashto-English-Hindi using fine-tuned NLLB + IndicTrans2
5. Manually score Hindi output for:
   - meaning preservation
   - Hindi fluency
   - completeness
   - named entity correctness
   - hallucination
   - naturalness

## Expected Output

The expected final output will be a comparison table showing which Hindi strategy is better for each sample.

## Status

IndicTrans2 is marked as remaining exploration work. It should be marked completed only after IndicTrans2 inference is successfully run and outputs are compared manually.
"""

Path("docs/remaining_indictrans2_exploration.md").write_text(text, encoding="utf-8")

print("Created docs/remaining_indictrans2_exploration.md")
