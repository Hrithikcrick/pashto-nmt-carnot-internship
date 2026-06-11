import pandas as pd

domains = [
    "daily_conversation",
    "education",
    "healthcare",
    "government",
    "travel",
    "news",
    "emergency",
    "technology"
]

rows = []

for i in range(1, 101):
    rows.append({
        "id": i,
        "domain": domains[(i - 1) % len(domains)],
        "pashto": "",
        "english_reference": "",
        "hindi_reference": "",
        "difficulty": ""
    })

df = pd.DataFrame(rows)
df.to_csv("data/gold_test_100_template.csv", index=False, encoding="utf-8-sig")

print("gold_test_100_template.csv created inside data folder")
