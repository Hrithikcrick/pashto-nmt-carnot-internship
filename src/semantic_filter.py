import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--pashto_col", default="pashto")
    parser.add_argument("--english_col", default="english")
    parser.add_argument("--model", default="sentence-transformers/LaBSE")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df = df.dropna(subset=[args.pashto_col, args.english_col])
    df[args.pashto_col] = df[args.pashto_col].astype(str)
    df[args.english_col] = df[args.english_col].astype(str)

    model = SentenceTransformer(args.model)

    pashto_sentences = df[args.pashto_col].tolist()
    english_sentences = df[args.english_col].tolist()

    print("Encoding Pashto sentences...")
    p_emb = model.encode(pashto_sentences, batch_size=32, show_progress_bar=True)

    print("Encoding English sentences...")
    e_emb = model.encode(english_sentences, batch_size=32, show_progress_bar=True)

    scores = []
    for i in range(len(df)):
        score = cosine_similarity([p_emb[i]], [e_emb[i]])[0][0]
        scores.append(float(score))

    df["semantic_similarity"] = scores

    filtered = df[df["semantic_similarity"] >= args.threshold].copy()

    filtered.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("Original rows:", len(df))
    print("Filtered rows:", len(filtered))
    print("Removed rows:", len(df) - len(filtered))
    print("Threshold:", args.threshold)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
