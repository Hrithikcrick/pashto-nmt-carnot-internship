import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in low:
            return low[name]
    return None

def translate(text, tokenizer, model, device, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/gold_test_candidates_100.csv")
    parser.add_argument("--output", default="outputs/tables/remaining_direct_vs_pivot_hindi.csv")
    parser.add_argument("--max_rows", type=int, default=20)
    args = parser.parse_args()

    os.makedirs("outputs/tables", exist_ok=True)

    df = pd.read_csv(args.input)

    pcol = find_col(df, ["pashto", "ps", "source", "pbt", "input"])

    if pcol is None:
        print("Could not detect Pashto column.")
        print("Columns found:", list(df.columns))
        return

    df = df.dropna(subset=[pcol]).head(args.max_rows).copy()

    model_name = "facebook/nllb-200-distilled-600M"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)
    print("Loading model:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    pivot_english = []
    direct_hindi = []
    pivot_hindi = []

    for idx, row in df.iterrows():
        pashto_text = str(row[pcol])

        english_text = translate(
            pashto_text,
            tokenizer,
            model,
            device,
            src_lang="pbt_Arab",
            tgt_lang="eng_Latn"
        )

        hindi_direct_text = translate(
            pashto_text,
            tokenizer,
            model,
            device,
            src_lang="pbt_Arab",
            tgt_lang="hin_Deva"
        )

        hindi_pivot_text = translate(
            english_text,
            tokenizer,
            model,
            device,
            src_lang="eng_Latn",
            tgt_lang="hin_Deva"
        )

        pivot_english.append(english_text)
        direct_hindi.append(hindi_direct_text)
        pivot_hindi.append(hindi_pivot_text)

        print("=" * 80)
        print("Sample:", len(pivot_english))
        print("Pashto:", pashto_text)
        print("Pivot English:", english_text)
        print("Direct Hindi:", hindi_direct_text)
        print("Pivot Hindi:", hindi_pivot_text)

    df["pivot_english"] = pivot_english
    df["direct_hindi"] = direct_hindi
    df["pivot_hindi"] = pivot_hindi
    df["manual_better_direct_or_pivot"] = ""
    df["manual_hindi_fluency_score_1_to_5"] = ""
    df["manual_meaning_preservation_score_1_to_5"] = ""
    df["manual_comment"] = ""

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Direct vs pivot Hindi comparison completed.")
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
