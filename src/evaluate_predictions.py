import argparse
import pandas as pd
import sacrebleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference_col", default="reference")
    parser.add_argument("--prediction_col", default="prediction")
    parser.add_argument("--output", default="outputs/tables/evaluation_result.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    refs = df[args.reference_col].astype(str).tolist()
    preds = df[args.prediction_col].astype(str).tolist()

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    result = pd.DataFrame([{
        "input_file": args.input,
        "samples": len(df),
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2)
    }])

    result.to_csv(args.output, index=False)
    print(result)

if __name__ == "__main__":
    main()
