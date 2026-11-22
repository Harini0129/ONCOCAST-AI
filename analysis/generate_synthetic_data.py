"""
ONCOCAST AI
Synthetic longitudinal breast cancer dataset generator.

IMPORTANT:
All generated records are synthetic research simulations.
The parameter values are illustrative assumptions, not clinical estimates.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SEED = 42
N_PATIENTS = 5000
FOLLOW_UP_MONTHS = 60

rng = np.random.default_rng(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Four simplified progression states
STAGES = [1, 2, 3, 4]

# Synthetic baseline transition rates per month.
# These are simulation assumptions, NOT clinical rates.
BASE_RATES = {
    "1_to_2": 0.012,
    "1_to_3": 0.002,
    "2_to_3": 0.015,
    "2_to_4": 0.001,
    "3_to_4": 0.020,
}

# --------------------------------------------------
# 2. GENERATE BASELINE PATIENT INFORMATION
# --------------------------------------------------

def generate_patients():
    records = []

    for i in range(N_PATIENTS):
        age = int(rng.integers(25, 81))
        tumor_size = round(float(rng.uniform(0.5, 7.0)), 2)
        grade = int(rng.choice([1, 2, 3], p=[0.20, 0.45, 0.35]))
        node_positive = int(rng.binomial(1, 0.40))
        positive_nodes = (
            int(rng.integers(1, 10)) if node_positive else 0
        )

        er = int(rng.binomial(1, 0.70))
        pr = int(rng.binomial(1, 0.60))
        her2 = int(rng.binomial(1, 0.20))

        # Initial simplified stage distribution.
        initial_stage = int(
            rng.choice([1, 2, 3, 4], p=[0.30, 0.40, 0.23, 0.07])
        )

        # Treatment is a simulated baseline characteristic.
        treatment = rng.choice(
        ["No_Treatment", "Hormone", "Chemotherapy", "Combined"],
        p=[0.10, 0.25, 0.30, 0.35]
        )

        records.append({
            "patient_id": f"ONC{i + 1:05d}",
            "age": age,
            "tumor_size_cm": tumor_size,
            "grade": grade,
            "node_positive": node_positive,
            "positive_nodes": positive_nodes,
            "ER_positive": er,
            "PR_positive": pr,
            "HER2_positive": her2,
            "treatment": treatment,
            "initial_stage": initial_stage,
            "follow_up_limit_months": FOLLOW_UP_MONTHS
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# 3. COVARIATE-DEPENDENT TRANSITION RATES
# --------------------------------------------------

def calculate_rates(patient):
    """
    Synthetic log-linear rate model.

    Covariates change the simulated transition intensity.
    Treatment coefficients are illustrative and must not
    be interpreted as causal or clinical treatment effects.
    """

    age_effect = (patient["age"] - 50) / 20
    size_effect = (patient["tumor_size_cm"] - 2.5) / 2
    grade_effect = patient["grade"] - 2
    node_effect = patient["node_positive"]
    her2_effect = patient["HER2_positive"]

    treatment_effect = {
    "No_Treatment": 0.0,
    "Hormone": -0.15,
    "Chemotherapy": -0.20,
    "Combined": -0.30
    } [patient["treatment"]]

    common_effect = (
        0.08 * age_effect
        + 0.20 * size_effect
        + 0.25 * grade_effect
        + 0.18 * node_effect
        + 0.10 * her2_effect
        + treatment_effect
    )

    rates = {
        "1_to_2": BASE_RATES["1_to_2"] * np.exp(common_effect),
        "1_to_3": BASE_RATES["1_to_3"] * np.exp(common_effect),
        "2_to_3": BASE_RATES["2_to_3"] * np.exp(common_effect),
        "2_to_4": BASE_RATES["2_to_4"] * np.exp(common_effect),
        "3_to_4": BASE_RATES["3_to_4"] * np.exp(common_effect),
    }

    return rates


# --------------------------------------------------
# 4. SIMULATE LATENT STAGE TRANSITIONS
# --------------------------------------------------

def simulate_patient_history(patient):
    """
    Simulates continuous-time progression.

    Stage IV is absorbing in this simplified model.
    The latent event times are kept separate from
    the observed assessment data.
    """

    rates = calculate_rates(patient)

    current_stage = int(patient["initial_stage"])
    current_time = 0.0
    latent_events = []

    while current_stage < 4:
        possible_transitions = {
            1: [("1_to_2", 2), ("1_to_3", 3)],
            2: [("2_to_3", 3), ("2_to_4", 4)],
            3: [("3_to_4", 4)]
        }

        options = possible_transitions[current_stage]

        option_rates = np.array([
            rates[name] for name, _ in options
        ])

        total_rate = option_rates.sum()

        # Waiting time in the current stage.
        waiting_time = rng.exponential(1 / total_rate)
        event_time = current_time + waiting_time

        if event_time > FOLLOW_UP_MONTHS:
            break

        # Select the next state based on competing rates.
        probabilities = option_rates / total_rate
        selected_index = rng.choice(
            len(options), p=probabilities
        )

        transition_name, next_stage = options[selected_index]

        latent_events.append({
            "patient_id": patient["patient_id"],
            "event_time_months": event_time,
            "from_stage": current_stage,
            "to_stage": next_stage,
            "transition_name": transition_name
        })

        current_stage = next_stage
        current_time = event_time

    return latent_events


# --------------------------------------------------
# 5. GENERATE IRREGULAR ASSESSMENTS
# --------------------------------------------------

def generate_assessments(patient, latent_events):
    """
    Generates irregular assessments approximately
    every 3–6 months, with exactly one final visit
    at month 60.
    """

    assessment_times = [0.0]

    while assessment_times[-1] < FOLLOW_UP_MONTHS:
        gap = float(rng.uniform(3.0, 6.0))

        next_time = min(
            assessment_times[-1] + gap,
            float(FOLLOW_UP_MONTHS)
        )

        next_time = round(next_time, 3)

        if next_time <= assessment_times[-1]:
            break

        assessment_times.append(next_time)

    records = []

    for visit_number, visit_time in enumerate(
        assessment_times, start=1
    ):
        stage = int(patient["initial_stage"])

        for event in latent_events:
            if event["event_time_months"] <= visit_time:
                stage = event["to_stage"]
            else:
                break

        records.append({
            "patient_id": patient["patient_id"],
            "assessment_number": visit_number,
            "assessment_month": visit_time,
            "observed_stage": stage,
            "tumor_size_cm": patient["tumor_size_cm"],
            "grade": patient["grade"],
            "node_positive": patient["node_positive"],
            "positive_nodes": patient["positive_nodes"],
            "ER_positive": patient["ER_positive"],
            "PR_positive": patient["PR_positive"],
            "HER2_positive": patient["HER2_positive"],
            "treatment": patient["treatment"]
        })

    return records
# --------------------------------------------------
# 6. CREATE CONSECUTIVE OBSERVATION INTERVALS
# --------------------------------------------------

def create_intervals(assessments):
    intervals = []

    for patient_id, group in assessments.groupby("patient_id"):
        group = group.sort_values("assessment_month")
        rows = group.to_dict("records")

        for i in range(len(rows) - 1):
            start = rows[i]
            end = rows[i + 1]

            duration = (
                end["assessment_month"]
                - start["assessment_month"]
            )

            start_stage = int(start["observed_stage"])
            end_stage = int(end["observed_stage"])

            intervals.append({
                "patient_id": patient_id,
                "start_month": start["assessment_month"],
                "end_month": end["assessment_month"],
                "duration_months": round(duration, 3),
                "start_stage": start_stage,
                "end_stage": end_stage,
                "transition": int(start_stage != end_stage),
                "age": int(
                    patients_lookup[patient_id]["age"]
                ),
                "tumor_size_cm": start["tumor_size_cm"],
                "grade": start["grade"],
                "node_positive": start["node_positive"],
                "positive_nodes": start["positive_nodes"],
                "ER_positive": start["ER_positive"],
                "PR_positive": start["PR_positive"],
                "HER2_positive": start["HER2_positive"],
                "treatment": start["treatment"]
            })

    return intervals


# --------------------------------------------------
# 7. MAIN GENERATION PROCESS
# --------------------------------------------------

print("Starting ONCOCAST AI synthetic data generation...")

patients_df = generate_patients()

patients_lookup = (
    patients_df.set_index("patient_id").to_dict("index")
)

all_assessments = []
all_latent_events = []

for _, patient in patients_df.iterrows():
    events = simulate_patient_history(patient)

    all_latent_events.extend(events)

    assessments = generate_assessments(patient, events)
    all_assessments.extend(assessments)

assessments_df = pd.DataFrame(all_assessments)

intervals_df = pd.DataFrame(
    create_intervals(assessments_df)
)

truth_df = pd.DataFrame(all_latent_events)

# --------------------------------------------------
# 8. SAVE DATASETS
# --------------------------------------------------

patients_df.to_csv(
    DATA_DIR / "patients.csv", index=False
)

assessments_df.to_csv(
    DATA_DIR / "assessments.csv", index=False
)

intervals_df.to_csv(
    DATA_DIR / "transition_intervals.csv", index=False
)

# This file contains hidden simulator truth.
# NEVER use it as a model training input.
truth_df.to_csv(
    DATA_DIR / "simulation_truth_DO_NOT_TRAIN.csv",
    index=False
)

# --------------------------------------------------
# 9. GENERATION SUMMARY
# --------------------------------------------------

print("\nDATA GENERATION COMPLETED")
print("----------------------------------")
print("Patients:", len(patients_df))
print("Assessments:", len(assessments_df))
print("Observation intervals:", len(intervals_df))
print("Latent transitions:", len(truth_df))
print("\nInitial stage distribution:")
print(patients_df["initial_stage"].value_counts().sort_index())

print("\nObserved stage distribution:")
print(assessments_df["observed_stage"].value_counts().sort_index())

print("\nFiles saved in:", DATA_DIR)
print("----------------------------------")
print("IMPORTANT: These are synthetic research data.")
print("Simulation truth must not be used for model training.")
