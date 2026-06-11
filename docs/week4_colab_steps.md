# Week 4 Google Colab Steps

## Step 1

Open Google Colab and select:

Runtime -> Change runtime type -> T4 GPU

## Step 2

Clone repository:

!git clone https://github.com/Hrithikcrick/pashto-nmt-carnot-internship.git
%cd pashto-nmt-carnot-internship

## Step 3

Install libraries:

!pip install -q torch transformers sentencepiece datasets accelerate peft sacrebleu pandas matplotlib

## Step 4

Run trial fine-tuning:

!python src/finetune_nllb_lora.py --train_file data/train_high_quality_10k.csv --output_dir models/nllb_lora_trial --max_rows 500 --epochs 1 --batch_size 2

## Step 5

Run full fine-tuning:

!python src/finetune_nllb_lora.py --train_file data/train_high_quality_10k.csv --output_dir models/nllb_lora_pashto_en_10k --max_rows 10000 --epochs 2 --batch_size 2

## Step 6

Evaluate baseline:

!python src/evaluate_week4_model.py --model_dir facebook/nllb-200-distilled-600M --test_file data/gold_test_candidates_100.csv --output outputs/tables/week4_baseline_predictions.csv

## Step 7

Evaluate fine-tuned model:

!python src/evaluate_week4_model.py --model_dir models/nllb_lora_pashto_en_10k --test_file data/gold_test_candidates_100.csv --output outputs/tables/week4_finetuned_predictions.csv

## Step 8

Generate report:

!python src/generate_week4_report.py
