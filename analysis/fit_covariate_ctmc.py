
import numpy as np
import pandas as pd

from scipy.linalg import expm
from scipy.optimize import minimize


# ---------------------------------------
# 1. Load and prepare observed intervals
# ---------------------------------------

df = pd.read_csv("data/transition_intervals.csv")

df = df[
    (df["duration_months"] > 0)
    & (df["start_stage"].between(1, 4))
    & (df["end_stage"].between(1, 4))
].copy()

# Use a reproducible development sample to reduce fitting time.
sample_size = min(2000, len(df))
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

features = [
    "age",
    "tumor_size_cm",
    "grade",
    "node_positive"
]

# Standardize using the development sample.
means = df[features].mean()
stds = df[features].std().replace(0, 1)

X = ((df[features] - means) / stds).to_numpy(dtype=float)

durations = df["duration_months"].to_numpy(dtype=float)
starts = df["start_stage"].to_numpy(dtype=int) - 1
ends = df["end_stage"].to_numpy(dtype=int) - 1


# ---------------------------------------
# 2. Build patient-specific generator
# ---------------------------------------

def build_generator(rates):
    q12, q13, q23, q24, q34 = rates

    return np.array([
        [-(q12 + q13), q12, q13, 0.0],
        [0.0, -(q23 + q24), q23, q24],
        [0.0, 0.0, -q34, q34],
        [0.0, 0.0, 0.0, 0.0]
    ])


# ---------------------------------------
# 3. Negative log-likelihood
# ---------------------------------------

def negative_log_likelihood(params):
    # First five parameters are baseline log-rates.
    base_rates = np.exp(params[:5])

    # Four shared covariate effects.
    beta = params[5:]

    linear_effect = X @ beta
    multiplier = np.exp(np.clip(linear_effect, -10, 10))

    total_log_likelihood = 0.0

    for i in range(len(df)):
        patient_rates = base_rates * multiplier[i]
        Q = build_generator(patient_rates)

        P = expm(Q * durations[i])
        probability = P[starts[i], ends[i]]

        if probability <= 0 or not np.isfinite(probability):
            return 1e100

        total_log_likelihood += np.log(probability)

    return -total_log_likelihood


# ---------------------------------------
# 4. Fit the covariate-dependent model
# ---------------------------------------

initial_rates = np.array([
    0.012443,
    0.002171,
    0.016447,
    0.000963,
    0.022730
])

initial_params = np.concatenate([
    np.log(initial_rates),
    np.zeros(len(features))
])

print("Fitting covariate-dependent CTMC...")
print(f"Development intervals: {len(df)}")

result = minimize(
    negative_log_likelihood,
    initial_params,
    method="L-BFGS-B",
    options={"maxiter": 150}
)

estimated_rates = np.exp(result.x[:5])
estimated_beta = result.x[5:]

print("\n========== COVARIATE CTMC RESULTS ==========")

transition_names = [
    "Stage I -> Stage II",
    "Stage I -> Stage III",
    "Stage II -> Stage III",
    "Stage II -> Stage IV",
    "Stage III -> Stage IV"
]

print("\nBaseline rates at average covariate values:")
for name, rate in zip(transition_names, estimated_rates):
    print(f"{name}: {rate:.6f} per month")

print("\nShared covariate effects:")
for name, coefficient in zip(features, estimated_beta):
    print(f"{name}: {coefficient:.6f}")

print("\nNegative log-likelihood:")
print(f"{result.fun:.4f}")

print("\nOptimizer convergence:")
print(result.success)
print(result.message)


# ---------------------------------------
# 5. Save development model results
# ---------------------------------------

output = pd.DataFrame({
    "transition": transition_names,
    "baseline_rate_per_month": estimated_rates
})

output.to_csv(
    "outputs/covariate_ctmc_development_rates.csv",
    index=False
)

effects = pd.DataFrame({
    "covariate": features,
    "coefficient": estimated_beta,
    "hazard_multiplier": np.exp(estimated_beta)
})

effects.to_csv(
    "outputs/covariate_ctmc_effects.csv",
    index=False
)

print("\nDevelopment results saved in outputs/")
