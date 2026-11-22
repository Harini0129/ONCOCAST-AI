
import numpy as np
import pandas as pd
from scipy.linalg import expm
from pathlib import Path


# ---------------------------------------
# 1. Load fitted CTMC model parameters
# ---------------------------------------

rates_file = Path("outputs/ctmc_v2_rates.csv")
effects_file = Path("outputs/ctmc_v2_effects.csv")

if not rates_file.exists() or not effects_file.exists():
    raise FileNotFoundError(
        "Model parameter files not found. Run fit_covariate_ctmc_v2.py first."
    )

rates_df = pd.read_csv(rates_file)
effects_df = pd.read_csv(effects_file)

base_rates = rates_df["baseline_rate_per_month"].to_numpy(float)
beta = effects_df["coefficient"].to_numpy(float)

feature_names = effects_df["covariate"].tolist()


# ---------------------------------------
# 2. Collect patient details
# ---------------------------------------

print("\n========== ONCOCAST AI ==========")
print("Patient-Specific Breast Cancer Progression Analysis")

age = float(input("\nEnter patient age: "))
tumor_size = float(input("Enter tumour size in cm: "))
grade = int(input("Enter tumour grade (1, 2 or 3): "))

node_positive = input(
    "Are lymph nodes positive? (yes/no): "
).strip().lower()

her2_positive = input(
    "Is HER2 positive? (yes/no): "
).strip().lower()

treatment = input(
    "Treatment (No_Treatment / Hormone / Chemotherapy / Combined): "
).strip()

current_stage = int(input("Enter current cancer stage (1-4): "))


# ---------------------------------------
# 3. Validate patient details
# ---------------------------------------

if age <= 0 or age > 120:
    raise ValueError("Age must be between 1 and 120.")

if tumor_size < 0:
    raise ValueError("Tumour size cannot be negative.")

if grade not in [1, 2, 3]:
    raise ValueError("Grade must be 1, 2 or 3.")

if node_positive not in ["yes", "no"]:
    raise ValueError("Enter yes or no for node positivity.")

if her2_positive not in ["yes", "no"]:
    raise ValueError("Enter yes or no for HER2 status.")

if treatment not in [
    "No_Treatment", "Hormone", "Chemotherapy", "Combined"
]:
    raise ValueError("Invalid treatment category.")

if current_stage not in [1, 2, 3, 4]:
    raise ValueError("Stage must be between 1 and 4.")

# Save patient input details for the PDF report
patient_profile = pd.DataFrame([{
    "age": age,
    "tumor_size_cm": tumor_size,
    "grade": grade,
    "node_positive": node_positive,
    "HER2_positive": her2_positive,
    "treatment": treatment,
    "current_stage": current_stage
}])

patient_profile.to_csv(
    "outputs/patient_profile.csv",
    index=False
)

# ---------------------------------------
# 4. Prepare patient covariates
# ---------------------------------------

patient_features = {
    "age_effect": (age - 50) / 20,
    "tumor_size_effect": (tumor_size - 2.5) / 2,
    "grade_effect": grade - 2,
    "node_positive": 1.0 if node_positive == "yes" else 0.0,
    "HER2_positive": 1.0 if her2_positive == "yes" else 0.0,
    "treatment_Hormone": 1.0 if treatment == "Hormone" else 0.0,
    "treatment_Chemotherapy": 1.0 if treatment == "Chemotherapy" else 0.0,
    "treatment_Combined": 1.0 if treatment == "Combined" else 0.0
}

X = np.array([
    patient_features[name] for name in feature_names
])

# Calculate the shared covariate multiplier.
linear_effect = np.dot(X, beta)
multiplier = np.exp(np.clip(linear_effect, -10, 10))

# Calculate patient-specific transition rates.
patient_rates = base_rates * multiplier


# ---------------------------------------
# 5. Construct patient-specific Q matrix
# ---------------------------------------

q12, q13, q23, q24, q34 = patient_rates

Q = np.array([
    [-(q12 + q13), q12, q13, 0.0],
    [0.0, -(q23 + q24), q23, q24],
    [0.0, 0.0, -q34, q34],
    [0.0, 0.0, 0.0, 0.0]
])


# ---------------------------------------
# 6. Calculate stage probabilities
# ---------------------------------------

stage_names = [
    "Stage I",
    "Stage II",
    "Stage III",
    "Stage IV"
]

start_index = current_stage - 1

time_points = [12, 24, 36, 60]

results = []

print("\n========== STAGE PROGRESSION PROBABILITIES ==========")
print("Current stage:", stage_names[start_index])

for months in time_points:

    P = expm(Q * months)

    probabilities = P[start_index, :]

    # Remove tiny numerical rounding errors.
    probabilities = np.clip(probabilities, 0, 1)
    probabilities = probabilities / probabilities.sum()

    row = {"Months": months}

    for i, stage in enumerate(stage_names):
        row[stage] = probabilities[i] * 100

    results.append(row)

    print(f"\nAfter {months} months:")

    for i, stage in enumerate(stage_names):
        print(f"{stage}: {probabilities[i] * 100:.2f}%")


# ---------------------------------------
# 7. Calculate transition rates
# ---------------------------------------

transition_names = [
    "Stage I -> Stage II",
    "Stage I -> Stage III",
    "Stage II -> Stage III",
    "Stage II -> Stage IV",
    "Stage III -> Stage IV"
]

print("\n========== PATIENT-SPECIFIC RATES ==========")

for name, rate in zip(transition_names, patient_rates):
    print(f"{name}: {rate:.6f} per month")


# ---------------------------------------
# 8. Estimate waiting time in current stage
# ---------------------------------------

exit_rate = -Q[start_index, start_index]

print("\n========== CURRENT STAGE WAITING TIME ==========")

if exit_rate > 0:

    mean_wait = 1 / exit_rate
    median_wait = np.log(2) / exit_rate

    print(f"Mean waiting time: {mean_wait:.2f} months")
    print(f"Median waiting time: {median_wait:.2f} months")

    print("\nProbability of leaving current stage:")

    for months in time_points:
        probability_leave = 1 - np.exp(-exit_rate * months)
        print(
            f"Within {months} months: "
            f"{probability_leave * 100:.2f}%"
        )

else:
    print("Stage IV is absorbing in this model.")
    mean_wait = np.inf
    median_wait = np.inf


# ---------------------------------------
# 9. Save patient prediction report data
# ---------------------------------------

probability_df = pd.DataFrame(results)

probability_df.to_csv(
    "outputs/patient_stage_probabilities.csv",
    index=False
)

rate_output = pd.DataFrame({
    "transition": transition_names,
    "patient_rate_per_month": patient_rates
})

rate_output.to_csv(
    "outputs/patient_transition_rates.csv",
    index=False
)

print("\nPrediction files saved in outputs/")
print("Patient-specific analysis completed.")
