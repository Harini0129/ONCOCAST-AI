
import numpy as np
import pandas as pd
from scipy.linalg import expm

rates = pd.read_csv("outputs/baseline_ctmc_rates.csv")[
    "estimated_rate_per_month"
].to_numpy()

q12, q13, q23, q24, q34 = rates

Q = np.array([
    [-(q12 + q13), q12, q13, 0.0],
    [0.0, -(q23 + q24), q23, q24],
    [0.0, 0.0, -q34, q34],
    [0.0, 0.0, 0.0, 0.0]
])

starting_stage = 1
start_index = starting_stage - 1

results = []

for months in [12, 24, 36, 60]:
    P = expm(Q * months)
    probabilities = P[start_index]

    results.append({
        "months": months,
        "Stage I": probabilities[0],
        "Stage II": probabilities[1],
        "Stage III": probabilities[2],
        "Stage IV": probabilities[3]
    })

output = pd.DataFrame(results)

print("\nSTAGE OCCUPANCY PROBABILITIES")
print(output.to_string(index=False))

output.to_csv(
    "outputs/stage_occupancy_probabilities.csv",
    index=False
)

print("\nSaved to outputs/stage_occupancy_probabilities.csv")
