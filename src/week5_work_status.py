import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

rows = [
    {"task": "Semantic filtering script", "completion": 100},
    {"task": "Filtered dataset generation", "completion": 100},
    {"task": "Similarity graph", "completion": 100},
    {"task": "Manual evaluation template", "completion": 100},
    {"task": "Human scoring", "completion": 0},
    {"task": "Direct vs pivot Hindi test", "completion": 0},
    {"task": "IndicTrans2 integration", "completion": 0}
]

df = pd.DataFrame(rows)
df.to_csv("outputs/tables/week5_work_status.csv", index=False)

plt.figure(figsize=(10, 5))
bars = plt.barh(df["task"], df["completion"])
plt.title("Week 5 Research Work Status")
plt.xlabel("Completion percentage")
plt.ylabel("Task")
plt.xlim(0, 100)

for bar in bars:
    x = bar.get_width()
    plt.text(x + 1, bar.get_y() + bar.get_height()/2, f"{int(x)}%", va="center")

plt.tight_layout()
plt.savefig("outputs/figures/week5_work_status.png", dpi=300)
plt.close()

print("Week 5 work status graph created.")
