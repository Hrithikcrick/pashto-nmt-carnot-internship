import os
import argparse
import pandas as pd
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import AutoPeftModelForSeq2SeqLM

def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in low:
            return low[name]
    return None

def load_model(model_dir, device):
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        return AutoPeftModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    return AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

def translate(text, tokenizer, model, device):
    tokenizer.src_lang = "pbt_Arab"
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128).to(device)
    bos = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=bos,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--test_file", default="data/gold_test_candidates_100.csv")
    parser.add_argument("--output", default="outputs/tables/week4_predictions.csv")
    args = parser.parse_args()

    os.makedirs("outputs/tables", exist_ok=True)

    df = pd.read_csv(args.test_file)

    pcol = find_col(df, ["pashto", "ps", "source", "pbt", "input"])
    ecol = find_col(df, ["english_reference", "english", "en", "target", "reference", "eng"])

    if pcol is None or ecol is None:
        print("Columns found:", list(df.columns))
        print("Could not detect columns.")
        return

    df = df.dropna(subset=[pcol, ecol]).head(100).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_model(args.model_dir, device)
    model.eval()

    preds = []

    for i, row in df.iterrows():
        pred = translate(row[pcol], tokenizer, model, device)
        preds.append(pred)
        print(i + 1, pred)

    df["prediction"] = preds

    refs = df[ecol].astype(str).tolist()
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    name = args.model_dir.replace("/", "_").replace("\\", "_")
    score_file = "outputs/tables/week4_scores_" + name + ".csv"

    score = pd.DataFrame([{
        "model": args.model_dir,
        "samples": len(df),
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2)
    }])

    score.to_csv(score_file, index=False)

    print(score)
    print("Saved prediction file:", args.output)
    print("Saved score file:", score_file)

if __name__ == "__main__":
    main()
