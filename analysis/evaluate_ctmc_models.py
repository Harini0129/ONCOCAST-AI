
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

# Split by PATIENT, never by individual interval.
patient_ids = df["patient_id"].unique()

train_ids, test_ids = train_test_split(
    patient_ids,
    test_size=0.20,
    random_state=42
)

train_df = df[df["patient_id"].isin(train_ids)].copy()
test_df = df[df["patient_id"].isin(test_ids)].copy()

# Use manageable, reproducible interval samples.
# Patients remain separated between training and testing.
train_df = train_df.sample(
    n=min(2000, len(train_df)),
    random_state=42
).reset_index(drop=True)

test_df = test_df.sample(
    n=min(2000, len(test_df)),
    random_state=43
).reset_index(drop=True)

print("Training patients:", len(train_ids))
print("Testing patients:", len(test_ids))
print("Training intervals:", len(train_df))
print("Testing intervals:", len(test_df))


# ---------------------------------------
# 2. Prepare covariates
# ---------------------------------------

features = [
    "age",
    "tumor_size_cm",
    "grade",
    "node_positive"
]

# Calculate scaling values using training data only.
means = train_df[features].mean()
stds = train_df[features].std().replace(0, 1)

X_train = ((train_df[features] - means) / stds).to_numpy(float)
X_test = ((test_df[features] - means) / stds).to_numpy(float)


def get_arrays(data):
    durations = data["duration_months"].to_numpy(float)
    starts = data["start_stage"].to_numpy(int) - 1
    ends = data["end_stage"].to_numpy(int) - 1
    return durations, starts, ends


train_durations, train_starts, train_ends = get_arrays(train_df)
test_durations, test_starts, test_ends = get_arrays(test_df)


# ---------------------------------------
# 3. Generator matrix
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
# 4. Likelihood calculation
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
            multiplier = np.exp(
                np.clip(np.dot(X[i], beta), -10, 10)
            )
            rates = base_rates * multiplier

        Q = build_generator(rates)
        P = expm(Q * durations[i])

        probability = P[starts[i], ends[i]]

        if probability <= 0 or not np.isfinite(probability):
            return 1e100

        total_log_likelihood += np.log(probability)

    return -total_log_likelihood


# ---------------------------------------
# 5. Fit baseline model on training data
# ---------------------------------------

initial_rates = np.array([
    0.012443,
    0.002171,
    0.016447,
    0.000963,
    0.022730
])

initial_base = np.log(initial_rates)

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
# 6. Fit covariate model on same training data
# ---------------------------------------

initial_covariate = np.concatenate([
    initial_base,
    np.zeros(len(features))
])

print("Fitting covariate-dependent CTMC...")

covariate_fit = minimize(
    calculate_nll,
    initial_covariate,
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
# 7. Evaluate both models on unseen patients
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

results = pd.DataFrame({
    "model": [
        "Baseline CTMC",
        "Covariate-dependent CTMC"
    ],
    "test_total_nll": [
        baseline_test_nll,
        covariate_test_nll
    ],
    "test_mean_nll_per_interval": [
        baseline_test_nll / len(test_df),
        covariate_test_nll / len(test_df)
    ],
    "optimizer_converged": [
        baseline_fit.success,
        covariate_fit.success
    ]
})


# ---------------------------------------
# 8. Display and save results
# ---------------------------------------

print("\n========== FAIR MODEL COMPARISON ==========")
print(results.to_string(index=False))

print("\nTraining convergence:")
print("Baseline:", baseline_fit.success)
print("Covariate model:", covariate_fit.success)

results.to_csv(
    "outputs/ctmc_model_comparison.csv",
    index=False
)

print("\nResults saved to outputs/ctmc_model_comparison.csv")
print("Evaluation completed.")
