import pandas as pd

df = pd.DataFrame({
    "Model": [
        "Baseline NLLB 600M",
        "NLLB LoRA 10k clean",
        "NLLB LoRA 10k filtered",
        "NLLB LoRA 50k filtered"
    ],
    "Dataset": [
        "No fine-tuning",
        "10k clean",
        "10k semantic-filtered",
        "50k semantic-filtered"
    ],
    "BLEU": [17.97, "", "", ""],
    "chrF": [36.49, "", "", ""],
    "Avg Time": ["4.46 sec", "", "", ""],
    "Manual Score": ["", "", "", ""]
})

df.to_csv("outputs/result_table.csv", index=False)
print("result_table.csv created inside outputs folder")
