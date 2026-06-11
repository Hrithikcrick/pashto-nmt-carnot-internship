import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in low:
            return low[name]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/train_high_quality_10k.csv")
    parser.add_argument("--output_dir", default="models/nllb_lora_pashto_en_10k")
    parser.add_argument("--max_rows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    model_name = "facebook/nllb-200-distilled-600M"

    df = pd.read_csv(args.train_file)

    pcol = find_col(df, ["pashto", "ps", "source", "pbt", "input"])
    ecol = find_col(df, ["english", "en", "target", "eng", "reference", "english_reference"])

    if pcol is None or ecol is None:
        print("Columns found:", list(df.columns))
        print("Could not detect Pashto and English columns.")
        return

    df = df[[pcol, ecol]].dropna()
    df = df.rename(columns={pcol: "pashto", ecol: "english"})

    if args.max_rows > 0:
        df = df.head(args.max_rows)

    train_df = df.sample(frac=0.9, random_state=42)
    valid_df = df.drop(train_df.index)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="pbt_Arab", tgt_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def preprocess(batch):
        x = tokenizer(batch["pashto"], max_length=128, truncation=True)
        y = tokenizer(text_target=batch["english"], max_length=128, truncation=True)
        x["labels"] = y["input_ids"]
        return x

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    valid_tok = valid_ds.map(preprocess, batched=True, remove_columns=valid_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args_train = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args_train,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        processing_class=tokenizer,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Fine-tuning completed.")
    print("Saved at:", args.output_dir)

if __name__ == "__main__":
    main()
