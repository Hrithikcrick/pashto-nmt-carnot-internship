import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in low:
            return low[name]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_high_quality_10k.csv")
    parser.add_argument("--output", default="data/train_semantic_filtered_8000.csv")
    parser.add_argument("--max_rows", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(args.input)

    pcol = find_col(df, ["pashto", "ps", "source", "pbt", "input"])
    ecol = find_col(df, ["english", "en", "target", "eng", "reference", "english_reference"])

    if pcol is None or ecol is None:
        print("Could not detect Pashto and English columns.")
        print("Columns found:", list(df.columns))
        return

    df = df[[pcol, ecol]].dropna().copy()
    df = df.rename(columns={pcol: "pashto", ecol: "english"})

    if args.max_rows > 0:
        df = df.head(args.max_rows)

    print("Total rows for semantic scoring:", len(df))
    print("Loading multilingual embedding model:", args.model)

    model = SentenceTransformer(args.model)

    ps_texts = df["pashto"].astype(str).tolist()
    en_texts = df["english"].astype(str).tolist()

    print("Encoding Pashto sentences...")
    ps_emb = model.encode(
        ps_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Encoding English sentences...")
    en_emb = model.encode(
        en_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.sum(ps_emb * en_emb, axis=1)
    df["semantic_similarity"] = scores

    df = df.sort_values("semantic_similarity", ascending=False).reset_index(drop=True)
    df.to_csv("outputs/tables/week5_semantic_similarity_scores.csv", index=False, encoding="utf-8-sig")

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
        "filtered_output": args.output
    }])

    summary.to_csv("outputs/tables/week5_semantic_filtering_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(scores, bins=40)
    plt.title("Week 5 Semantic Similarity Distribution")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Sentence pair count")
    plt.tight_layout()
    plt.savefig("outputs/figures/week5_semantic_similarity_distribution.png", dpi=300)
    plt.close()

    top20 = filtered.head(20)
    top20.to_csv("outputs/tables/week5_top20_semantic_filtered_preview.csv", index=False, encoding="utf-8-sig")

    print("Semantic filtering completed.")
    print("Saved filtered data:", args.output)
    print("Saved summary: outputs/tables/week5_semantic_filtering_summary.csv")
    print("Saved graph: outputs/figures/week5_semantic_similarity_distribution.png")

if __name__ == "__main__":
    main()
