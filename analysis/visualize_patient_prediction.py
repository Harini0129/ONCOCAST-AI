
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------
# 1. Load patient prediction results
# ---------------------------------------

input_file = Path("outputs/patient_stage_probabilities.csv")

if not input_file.exists():
    raise FileNotFoundError(
        "Prediction results not found. Run patient_prediction.py first."
    )

df = pd.read_csv(input_file)

stage_columns = [
    "Stage I",
    "Stage II",
    "Stage III",
    "Stage IV"
]

# ---------------------------------------
# 2. Create stage progression chart
# ---------------------------------------

plt.figure(figsize=(10, 6))

for stage in stage_columns:
    plt.plot(
        df["Months"],
        df[stage],
        marker="o",
        linewidth=2,
        markersize=7,
        label=stage
    )

plt.title(
    "Patient-Specific Stage Occupancy Probabilities",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Time (Months)")
plt.ylabel("Probability (%)")

plt.xticks(df["Months"])
plt.ylim(0, 100)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Cancer Stage")

plt.tight_layout()

# ---------------------------------------
# 3. Save chart
# ---------------------------------------

output_file = Path("outputs/patient_stage_probability_chart.png")

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nChart successfully generated!")
print("Saved at:", output_file)
