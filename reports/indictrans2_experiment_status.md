# IndicTrans2 Applicability and Status

IndicTrans2 was considered for the English-to-Hindi stage of the pivot
translation pipeline.

Proposed pipeline:

Pashto -> English using NLLB or Semantic LoRA
English -> Hindi using IndicTrans2

IndicTrans2 was not used as a direct Pashto-to-English baseline because
it does not directly support Pashto. Its gated model access and native
Windows toolkit installation also prevented completion in the present
local environment.

The instructor-requested multilingual Pashto-to-English comparison was
therefore completed using:

- NLLB-200 distilled 600M
- M2M100 418M
- mBART-50
- Original NLLB LoRA
- Semantic-filtered NLLB LoRA

IndicTrans2 remains applicable as future work for the English-to-Hindi
pivot stage and is not included in the current quantitative results.
