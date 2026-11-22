
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

patients = pd.read_csv(DATA / "patients.csv")
assessments = pd.read_csv(DATA / "assessments.csv")
intervals = pd.read_csv(DATA / "transition_intervals.csv")

truth_path = DATA / "simulation_truth_DO_NOT_TRAIN.csv"

print("\n========== ONCOCAST AI DATA VALIDATION ==========\n")

errors = []
warnings = []

def check(condition, success_message, error_message):
    if condition:
        print("[PASS]", success_message)
    else:
        print("[FAIL]", error_message)
        errors.append(error_message)


# ---------------------------------------------
# 2. BASIC DATASET CHECKS
# ---------------------------------------------

check(
    len(patients) == 5000,
    "Exactly 5,000 patients found.",
    f"Expected 5,000 patients, found {len(patients)}."
)

check(
    patients["patient_id"].is_unique,
    "Patient IDs are unique.",
    "Duplicate patient IDs found."
)

check(
    not patients.isnull().any().any(),
    "No missing values in patient records.",
    "Missing values found in patient records."
)

check(
    not assessments.isnull().any().any(),
    "No missing values in assessment records.",
    "Missing values found in assessment records."
)

check(
    not intervals.isnull().any().any(),
    "No missing values in interval records.",
    "Missing values found in interval records."
)


# ---------------------------------------------
# 3. PATIENT AND ASSESSMENT CONSISTENCY
# ---------------------------------------------

patient_ids = set(patients["patient_id"])
assessment_ids = set(assessments["patient_id"])
interval_ids = set(intervals["patient_id"])

check(
    assessment_ids.issubset(patient_ids),
    "All assessments belong to known patients.",
    "Some assessments refer to unknown patients."
)

check(
    interval_ids.issubset(patient_ids),
    "All intervals belong to known patients.",
    "Some intervals refer to unknown patients."
)

assessment_counts = assessments.groupby("patient_id").size()

check(
    len(assessment_counts) == 5000,
    "Every patient has assessment records.",
    "Some patients have no assessment records."
)

check(
    (assessment_counts >= 2).all(),
    "Every patient has at least two assessments.",
    "Some patients have fewer than two assessments."
)


# ---------------------------------------------
# 4. CHECK STAGE VALUES AND PROGRESSION
# ---------------------------------------------

valid_stages = {1, 2, 3, 4}

check(
    set(assessments["observed_stage"].unique()).issubset(valid_stages),
    "Assessment stages contain only stages 1–4.",
    "Invalid stage values found in assessments."
)

check(
    set(intervals["start_stage"].unique()).issubset(valid_stages)
    and set(intervals["end_stage"].unique()).issubset(valid_stages),
    "Interval stages contain only stages 1–4.",
    "Invalid stage values found in intervals."
)

stage_order_ok = True

for patient_id, group in assessments.groupby("patient_id"):
    group = group.sort_values("assessment_month")
    stages = group["observed_stage"].to_numpy()

    if np.any(np.diff(stages) < 0):
        stage_order_ok = False
        break

check(
    stage_order_ok,
    "No backward stage transitions observed.",
    "Backward stage transition found."
)

stage_iv_ok = True

for patient_id, group in assessments.groupby("patient_id"):
    group = group.sort_values("assessment_month")
    stages = group["observed_stage"].to_numpy()

    if 4 in stages:
        first_stage_iv = np.where(stages == 4)[0][0]

        if not np.all(stages[first_stage_iv:] == 4):
            stage_iv_ok = False
            break

check(
    stage_iv_ok,
    "Stage IV remains absorbing.",
    "A patient moved out of Stage IV."
)


# ---------------------------------------------
# 5. CHECK ASSESSMENT DATES
# ---------------------------------------------

time_ok = True

for patient_id, group in assessments.groupby("patient_id"):
    times = group.sort_values("assessment_month")[
        "assessment_month"
    ].to_numpy()

    if np.any(np.diff(times) <= 0):
        time_ok = False
        break

check(
    time_ok,
    "Assessment dates are strictly increasing.",
    "Duplicate or decreasing assessment dates found."
)

check(
    assessments["assessment_month"].between(0, 60).all(),
    "All assessment times are within 0–60 months.",
    "Assessment time outside the follow-up period."
)


# ---------------------------------------------
# 6. CHECK INTERVAL CALCULATIONS
# ---------------------------------------------

calculated_duration = (
    intervals["end_month"] - intervals["start_month"]
)

duration_ok = np.allclose(
    calculated_duration,
    intervals["duration_months"],
    atol=0.001
)

check(
    duration_ok,
    "Interval durations are correctly calculated.",
    "Incorrect interval durations found."
)

transition_ok = (
    intervals["transition"]
    == (
        intervals["start_stage"]
        != intervals["end_stage"]
    ).astype(int)
).all()

check(
    transition_ok,
    "Transition indicators match stage changes.",
    "Transition indicator mismatches found."
)

check(
    (intervals["duration_months"] > 0).all(),
    "All interval durations are positive.",
    "Zero or negative interval duration found."
)


# ---------------------------------------------
# 7. CHECK CONSECUTIVE ASSESSMENT MATCHING
# ---------------------------------------------

assessment_lookup = {}

for patient_id, group in assessments.groupby("patient_id"):
    group = group.sort_values("assessment_month")

    assessment_lookup[patient_id] = group.to_dict("records")

interval_matching_ok = True

for patient_id, group in intervals.groupby("patient_id"):
    expected = assessment_lookup[patient_id]

    group = group.sort_values("start_month")

    if len(group) != len(expected) - 1:
        interval_matching_ok = False
        break

    for i, interval in enumerate(group.to_dict("records")):
        start = expected[i]
        end = expected[i + 1]

        if not (
            np.isclose(interval["start_month"], start["assessment_month"])
            and np.isclose(interval["end_month"], end["assessment_month"])
            and interval["start_stage"] == start["observed_stage"]
            and interval["end_stage"] == end["observed_stage"]
        ):
            interval_matching_ok = False
            break

    if not interval_matching_ok:
        break

check(
    interval_matching_ok,
    "Intervals correctly match consecutive assessments.",
    "Some intervals do not match consecutive assessments."
)


# ---------------------------------------------
# 8. CHECK INITIAL STAGE
# ---------------------------------------------

initial_stage_ok = True

patient_lookup = patients.set_index("patient_id")["initial_stage"]

for patient_id, group in assessments.groupby("patient_id"):
    first_record = group.sort_values("assessment_month").iloc[0]

    if (
        first_record["assessment_month"] != 0
        or first_record["observed_stage"] != patient_lookup[patient_id]
    ):
        initial_stage_ok = False
        break

check(
    initial_stage_ok,
    "Initial assessment matches baseline stage.",
    "Initial assessment does not match baseline stage."
)


# ---------------------------------------------
# 9. TRANSITION SUMMARY
# ---------------------------------------------

print("\n========== TRANSITION SUMMARY ==========\n")

print("Observed interval transitions:")
print(
    intervals.groupby(
        ["start_stage", "end_stage"]
    ).size().to_string()
)

print("\nTransition indicator counts:")
print(intervals["transition"].value_counts().to_string())

print("\nAssessment stage distribution:")
print(
    assessments["observed_stage"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nAverage interval duration:",
      round(intervals["duration_months"].mean(), 3), "months")


# ---------------------------------------------
# 10. HIDDEN TRUTH FILE CHECK
# ---------------------------------------------

if truth_path.exists():
    truth = pd.read_csv(truth_path)

    check(
        truth["patient_id"].isin(patient_ids).all(),
        "Hidden truth contains known patient IDs.",
        "Unknown patient IDs found in hidden truth."
    )

    print(
        "\nHidden simulator transitions:",
        len(truth)
    )

    print(
        "Reminder: Do not include hidden truth in model training."
    )
else:
    warnings.append("Hidden simulation truth file was not found.")


# ---------------------------------------------
# 11. FINAL RESULT
# ---------------------------------------------

print("\n========== FINAL VALIDATION RESULT ==========\n")

if errors:
    print("VALIDATION FAILED.")
    print("Issues found:", len(errors))

    for error in errors:
        print("-", error)
else:
    print("ALL AUTOMATED CHECKS PASSED.")

if warnings:
    print("\nWARNINGS:")
    for warning in warnings:
        print("-", warning)

print("\nThis validates internal consistency,")
print("not clinical validity or realism.")
