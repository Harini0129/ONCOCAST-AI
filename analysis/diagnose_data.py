
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

patients = pd.read_csv(DATA / "patients.csv")
assessments = pd.read_csv(DATA / "assessments.csv")
intervals = pd.read_csv(DATA / "transition_intervals.csv")

print("\n========== MISSING VALUES ==========")

for name, df in [
    ("PATIENTS", patients),
    ("ASSESSMENTS", assessments),
    ("INTERVALS", intervals)
]:
    print(f"\n{name}")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.sum() else "No missing values")

print("\n========== DUPLICATE ASSESSMENT TIMES ==========")

duplicates = assessments[
    assessments.duplicated(
        subset=["patient_id", "assessment_month"],
        keep=False
    )
].sort_values(["patient_id", "assessment_month"])

print("Duplicate rows:", len(duplicates))
print(duplicates.head(20).to_string(index=False))

print("\n========== ZERO-DURATION INTERVALS ==========")

zero = intervals[
    intervals["duration_months"] <= 0
]

print("Affected intervals:", len(zero))
print(zero.head(20).to_string(index=False))

print("\n========== AFFECTED PATIENTS ==========")

bad_ids = set(duplicates["patient_id"]) | set(zero["patient_id"])

print("Number of affected patients:", len(bad_ids))

for patient_id in list(bad_ids)[:5]:
    print(f"\nPatient: {patient_id}")
    print(
        assessments[
            assessments["patient_id"] == patient_id
        ].sort_values("assessment_month").to_string(index=False)
    )

print("\n========== DIAGNOSTIC COMPLETE ==========")
