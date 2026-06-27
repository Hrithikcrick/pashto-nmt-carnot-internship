import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in low:
            return low[name]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_high_quality_10k.csv")
    parser.add_argument("--output", default="data/train_semantic_filtered.csv")
    parser.add_argument("--score_output", default="outputs/tables/semantic_similarity_scores.csv")
    parser.add_argument("--figure", default="outputs/figures/semantic_similarity_distribution.png")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--max_rows", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Reading:", args.input)
    df = pd.read_csv(args.input)

    pcol = find_col(df, ["pashto", "ps", "source", "pbt", "input"])
    ecol = find_col(df, ["english", "en", "target", "eng", "reference", "english_reference"])

    if pcol is None or ecol is None:
        print("Could not detect Pashto/English columns.")
        print("Columns:", list(df.columns))
        return

    df = df[[pcol, ecol]].dropna().copy()
    df = df.rename(columns={pcol: "pashto", ecol: "english"})

    if args.max_rows > 0:
        df = df.head(args.max_rows)

    print("Rows used:", len(df))
    print("Loading embedding model:", args.model)
    model = SentenceTransformer(args.model)

    pashto_texts = df["pashto"].astype(str).tolist()
    english_texts = df["english"].astype(str).tolist()

    print("Encoding Pashto sentences...")
    emb_ps = model.encode(
        pashto_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Encoding English sentences...")
    emb_en = model.encode(
        english_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.sum(emb_ps * emb_en, axis=1)
    df["semantic_similarity"] = scores

    df = df.sort_values("semantic_similarity", ascending=False).reset_index(drop=True)
    df.to_csv(args.score_output, index=False, encoding="utf-8-sig")

    filtered = df.head(args.top_k).copy()
    filtered.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "input_file": args.input,
        "total_rows_scored": len(df),
        "filtered_rows_saved": len(filtered),
        "mean_similarity": round(float(np.mean(scores)), 4),
        "median_similarity": round(float(np.median(scores)), 4),
        "min_similarity": round(float(np.min(scores)), 4),
        "max_similarity": round(float(np.max(scores)), 4),
        "output_file": args.output
    }])

    summary.to_csv("outputs/tables/semantic_filtering_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(scores, bins=40)
    plt.title("Semantic Similarity Distribution")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Number of sentence pairs")
    plt.tight_layout()
    plt.savefig(args.figure, dpi=300)
    plt.close()

    top_preview = filtered.head(20)
    top_preview.to_csv("outputs/tables/semantic_filtered_top20_preview.csv", index=False, encoding="utf-8-sig")

    print("Semantic filtering completed.")
    print("Saved scored file:", args.score_output)
    print("Saved filtered file:", args.output)
    print("Saved summary: outputs/tables/semantic_filtering_summary.csv")
    print("Saved figure:", args.figure)

if __name__ == "__main__":
    main()
