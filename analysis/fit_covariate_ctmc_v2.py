
import numpy as np
import pandas as pd

from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split


# ---------------------------------------
# 1. Load observed intervals
# ---------------------------------------

df = pd.read_csv("data/transition_intervals.csv")

df = df[
    (df["duration_months"] > 0)
    & (df["start_stage"].between(1, 4))
    & (df["end_stage"].between(1, 4))
].copy()

# Split by patient to prevent data leakage.
patient_ids = df["patient_id"].unique()

train_ids, test_ids = train_test_split(
    patient_ids,
    test_size=0.20,
    random_state=42
)

train_df = df[df["patient_id"].isin(train_ids)].copy()
test_df = df[df["patient_id"].isin(test_ids)].copy()

# Reproduce the same interval sampling used in the
# previous evaluation for a fair comparison.
train_df = train_df.sample(
    n=min(10000, len(train_df)),
    random_state=42
).reset_index(drop=True)

test_df = test_df.sample(
    n=min(10000, len(test_df)),
    random_state=43
).reset_index(drop=True)


# ---------------------------------------
# 2. Prepare covariates
# ---------------------------------------

def prepare_features(data):
    treatment = data["treatment"]

    return np.column_stack([
        (data["age"].to_numpy(float) - 50) / 20,
        (data["tumor_size_cm"].to_numpy(float) - 2.5) / 2,
        data["grade"].to_numpy(float) - 2,
        data["node_positive"].to_numpy(float),
        data["HER2_positive"].to_numpy(float),
        (treatment == "Hormone").to_numpy(float),
        (treatment == "Chemotherapy").to_numpy(float),
        (treatment == "Combined").to_numpy(float)
    ])


feature_names = [
    "age_effect",
    "tumor_size_effect",
    "grade_effect",
    "node_positive",
    "HER2_positive",
    "treatment_Hormone",
    "treatment_Chemotherapy",
    "treatment_Combined"
]

X_train = prepare_features(train_df)
X_test = prepare_features(test_df)


def get_arrays(data):
    return (
        data["duration_months"].to_numpy(float),
        data["start_stage"].to_numpy(int) - 1,
        data["end_stage"].to_numpy(int) - 1
    )


train_durations, train_starts, train_ends = get_arrays(train_df)
test_durations, test_starts, test_ends = get_arrays(test_df)


# ---------------------------------------
# 3. Build generator matrix
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
# 4. Negative log-likelihood
# ---------------------------------------

def calculate_nll(params, durations, starts, ends, X=None):

    base_rates = np.exp(params[:5])

    if X is None:
        beta = None
    else:
        beta = params[5:]

    total_log_likelihood = 0.0

    for i in range(len(durations)):

        if X is None:
            rates = base_rates
        else:
            eta = np.dot(X[i], beta)
            multiplier = np.exp(np.clip(eta, -10, 10))
            rates = base_rates * multiplier

        Q = build_generator(rates)
        P = expm(Q * durations[i])

        probability = P[starts[i], ends[i]]

        if probability <= 0 or not np.isfinite(probability):
            return 1e100

        total_log_likelihood += np.log(probability)

    return -total_log_likelihood


# ---------------------------------------
# 5. Fit baseline model
# ---------------------------------------

initial_rates = np.array([
    0.012,
    0.002,
    0.015,
    0.001,
    0.020
])

initial_base = np.log(initial_rates)

print("Training patients:", len(train_ids))
print("Testing patients:", len(test_ids))
print("Training intervals:", len(train_df))
print("Testing intervals:", len(test_df))

print("\nFitting baseline CTMC...")

baseline_fit = minimize(
    calculate_nll,
    initial_base,
    args=(
        train_durations,
        train_starts,
        train_ends
    ),
    method="L-BFGS-B",
    options={"maxiter": 150}
)


# ---------------------------------------
# 6. Fit six-covariate model
# ---------------------------------------

initial_params = np.concatenate([
    initial_base,
    np.zeros(len(feature_names))
])

print("\nFitting six-covariate CTMC...")

covariate_fit = minimize(
    calculate_nll,
    initial_params,
    args=(
        train_durations,
        train_starts,
        train_ends,
        X_train
    ),
    method="L-BFGS-B",
    options={"maxiter": 150}
)


# ---------------------------------------
# 7. Evaluate on unseen patients
# ---------------------------------------

baseline_test_nll = calculate_nll(
    baseline_fit.x,
    test_durations,
    test_starts,
    test_ends
)

covariate_test_nll = calculate_nll(
    covariate_fit.x,
    test_durations,
    test_starts,
    test_ends,
    X_test
)

comparison = pd.DataFrame({
    "model": [
        "Baseline CTMC",
        "Six-covariate CTMC"
    ],
    "test_total_nll": [
        baseline_test_nll,
        covariate_test_nll
    ],
    "test_mean_nll_per_interval": [
        baseline_test_nll / len(test_df),
        covariate_test_nll / len(test_df)
    ],
    "converged": [
        baseline_fit.success,
        covariate_fit.success
    ]
})


# ---------------------------------------
# 8. Save estimated effects
# ---------------------------------------

estimated_rates = np.exp(covariate_fit.x[:5])
estimated_beta = covariate_fit.x[5:]

transition_names = [
    "Stage I -> Stage II",
    "Stage I -> Stage III",
    "Stage II -> Stage III",
    "Stage II -> Stage IV",
    "Stage III -> Stage IV"
]

rates_output = pd.DataFrame({
    "transition": transition_names,
    "baseline_rate_per_month": estimated_rates
})

effects_output = pd.DataFrame({
    "covariate": feature_names,
    "coefficient": estimated_beta,
    "rate_multiplier": np.exp(estimated_beta)
})

comparison.to_csv(
    "outputs/ctmc_v2_comparison.csv",
    index=False
)

rates_output.to_csv(
    "outputs/ctmc_v2_rates.csv",
    index=False
)

effects_output.to_csv(
    "outputs/ctmc_v2_effects.csv",
    index=False
)


# ---------------------------------------
# 9. Display results
# ---------------------------------------

print("\n========== SIX-COVARIATE CTMC RESULTS ==========")

print("\nEstimated baseline rates:")
print(rates_output.to_string(index=False))

print("\nEstimated covariate effects:")
print(effects_output.to_string(index=False))

print("\nHeld-out model comparison:")
print(comparison.to_string(index=False))

print("\nOptimizer convergence:")
print("Baseline:", baseline_fit.success)
print("Six-covariate:", covariate_fit.success)

print("\nResults saved in outputs/")
