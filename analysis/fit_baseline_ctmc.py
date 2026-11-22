
import numpy as np
import pandas as pd

from scipy.linalg import expm
from scipy.optimize import minimize


# ---------------------------------------
# 1. Load observed assessment intervals
# ---------------------------------------

df = pd.read_csv("data/transition_intervals.csv")

# Keep only valid forward transitions and positive durations
df = df[
    (df["duration_months"] > 0)
    & (df["start_stage"].between(1, 4))
    & (df["end_stage"].between(1, 4))
].copy()

durations = df["duration_months"].to_numpy(dtype=float)
starts = df["start_stage"].to_numpy(dtype=int) - 1
ends = df["end_stage"].to_numpy(dtype=int) - 1


# ---------------------------------------
# 2. Construct the CTMC generator matrix
# ---------------------------------------

def build_generator(rates):
    q12, q13, q23, q24, q34 = rates

    Q = np.array([
        [-(q12 + q13), q12, q13, 0.0],
        [0.0, -(q23 + q24), q23, q24],
        [0.0, 0.0, -q34, q34],
        [0.0, 0.0, 0.0, 0.0]
    ])

    return Q


# ---------------------------------------
# 3. Negative log-likelihood
# ---------------------------------------

def negative_log_likelihood(log_rates):
    rates = np.exp(log_rates)
    Q = build_generator(rates)

    total_log_likelihood = 0.0

    for duration, start, end in zip(durations, starts, ends):
        P = expm(Q * duration)

        probability = P[start, end]

        if probability <= 0 or not np.isfinite(probability):
            return 1e100

        total_log_likelihood += np.log(probability)

    return -total_log_likelihood


# ---------------------------------------
# 4. Estimate transition rates
# ---------------------------------------

initial_rates = np.array([
    0.01, 0.002, 0.01, 0.001, 0.02
])

result = minimize(
    negative_log_likelihood,
    np.log(initial_rates),
    method="L-BFGS-B",
    options={"maxiter": 1000}
)

if not result.success:
    print("WARNING: Optimizer did not fully converge.")
    print(result.message)

estimated_rates = np.exp(result.x)
Q_estimated = build_generator(estimated_rates)


# ---------------------------------------
# 5. Display results
# ---------------------------------------

transition_names = [
    "Stage I -> Stage II",
    "Stage I -> Stage III",
    "Stage II -> Stage III",
    "Stage II -> Stage IV",
    "Stage III -> Stage IV"
]

print("\n========== BASELINE CTMC RESULTS ==========")

for name, rate in zip(transition_names, estimated_rates):
    print(f"{name}: {rate:.6f} per month")

print("\nEstimated generator matrix Q:")
print(Q_estimated)

print("\nNegative log-likelihood:")
print(f"{result.fun:.4f}")

print("\nOptimizer convergence:")
print(result.success)


# ---------------------------------------
# 6. Save estimated rates
# ---------------------------------------

rates_df = pd.DataFrame({
    "transition": transition_names,
    "estimated_rate_per_month": estimated_rates
})

rates_df.to_csv(
    "outputs/baseline_ctmc_rates.csv",
    index=False
)

print("\nRates saved to outputs/baseline_ctmc_rates.csv")
