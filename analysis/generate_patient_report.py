
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

# ---------------------------------------
# 1. File locations
# ---------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

profile_file = output_dir / "patient_profile.csv"
probability_file = output_dir / "patient_stage_probabilities.csv"
rates_file = output_dir / "patient_transition_rates.csv"

chart_file = output_dir / "patient_stage_probability_chart.png"
pdf_file = output_dir / "ONCOCAST_AI_Patient_Report.pdf"

required_files = [
    profile_file,
    probability_file,
    rates_file
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found. Run patient_prediction.py first."
        )

# ---------------------------------------
# 2. Read patient results
# ---------------------------------------

profile = pd.read_csv(profile_file).iloc[0]
probabilities = pd.read_csv(probability_file)
rates = pd.read_csv(rates_file)

stage_names = [
    "Stage I",
    "Stage II",
    "Stage III",
    "Stage IV"
]

# ---------------------------------------
# 3. Create chart if not already available
# ---------------------------------------

if not chart_file.exists():

    plt.figure(figsize=(9, 5))

    for stage in stage_names:
        plt.plot(
            probabilities["Months"],
            probabilities[stage],
            marker="o",
            linewidth=2,
            label=stage
        )

    plt.title("Stage Occupancy Probability Over Time")
    plt.xlabel("Time (Months)")
    plt.ylabel("Probability (%)")
    plt.xticks(probabilities["Months"])
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_file, dpi=250)
    plt.close()

# ---------------------------------------
# 4. Prepare PDF document
# ---------------------------------------

doc = SimpleDocTemplate(
    str(pdf_file),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=18 * mm,
    bottomMargin=18 * mm
)

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontSize=22,
    leading=28,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#123B52"),
    spaceAfter=8
))

styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading2"],
    fontSize=13,
    textColor=colors.HexColor("#087E8B"),
    spaceBefore=14,
    spaceAfter=8
))

styles.add(ParagraphStyle(
    name="SmallNote",
    parent=styles["BodyText"],
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#555555")
))

story = []

# ---------------------------------------
# 5. Report title
# ---------------------------------------

story.append(Paragraph(
    "ONCOCAST AI",
    styles["ReportTitle"]
))

story.append(Paragraph(
    "Patient-Specific Breast Cancer Progression Research Report",
    styles["Heading2"]
))

story.append(Spacer(1, 10))

story.append(Paragraph(
    "Research prototype report based on a fitted "
    "continuous-time Markov model.",
    styles["BodyText"]
))

# ---------------------------------------
# 6. Patient profile
# ---------------------------------------

story.append(Paragraph(
    "1. Patient Profile",
    styles["SectionHeading"]
))

profile_data = [
    ["Parameter", "Recorded Value"],
    ["Age", f"{profile['age']} years"],
    ["Tumour size", f"{profile['tumor_size_cm']} cm"],
    ["Tumour grade", str(profile["grade"])],
    ["Lymph node status", str(profile["node_positive"]).title()],
    ["HER2 status", str(profile["HER2_positive"]).title()],
    ["Treatment category", str(profile["treatment"])],
    ["Current stage", f"Stage {int(profile['current_stage'])}"]
]

profile_table = Table(
    profile_data,
    colWidths=[65 * mm, 95 * mm]
)

profile_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ("PADDING", (0, 0), (-1, -1), 7),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
]))

story.append(profile_table)

# ---------------------------------------
# 7. Stage probability results
# ---------------------------------------

story.append(Paragraph(
    "2. Estimated Stage Occupancy Probabilities",
    styles["SectionHeading"]
))

story.append(Paragraph(
    "The following values represent the model-estimated probability "
    "of occupying each stage at the specified future time, "
    "conditional on the entered patient characteristics and current stage.",
    styles["BodyText"]
))

probability_data = [[
    "Months",
    "Stage I",
    "Stage II",
    "Stage III",
    "Stage IV"
]]

for _, row in probabilities.iterrows():
    probability_data.append([
        str(int(row["Months"])),
        *[f"{row[stage]:.2f}%" for stage in stage_names]
    ])

probability_table = Table(
    probability_data,
    colWidths=[25 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm]
)

probability_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("PADDING", (0, 0), (-1, -1), 7)
]))

story.append(probability_table)

# ---------------------------------------
# 8. Progression chart
# ---------------------------------------

story.append(Paragraph(
    "3. Stage Progression Visualization",
    styles["SectionHeading"]
))

if chart_file.exists():
    chart = Image(str(chart_file), width=165 * mm, height=92 * mm)
    story.append(chart)

# ---------------------------------------
# 9. Patient-specific transition rates
# ---------------------------------------

story.append(Paragraph(
    "4. Estimated Transition Rates",
    styles["SectionHeading"]
))

rate_data = [
    ["Transition", "Estimated rate per month"]
]

for _, row in rates.iterrows():
    rate_data.append([
        str(row["transition"]),
        f"{row['patient_rate_per_month']:.6f}"
    ])

rate_table = Table(
    rate_data,
    colWidths=[95 * mm, 65 * mm]
)

rate_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ("PADDING", (0, 0), (-1, -1), 7)
]))

story.append(rate_table)

# ---------------------------------------
# 10. Waiting-time explanation
# ---------------------------------------

story.append(Paragraph(
    "5. Current Stage Waiting-Time Interpretation",
    styles["SectionHeading"]
))

current_stage = int(profile["current_stage"])

exit_rates = {
    1: rates.loc[
        rates["transition"].isin([
            "Stage I -> Stage II",
            "Stage I -> Stage III"
        ]),
        "patient_rate_per_month"
    ].sum(),
    2: rates.loc[
        rates["transition"].isin([
            "Stage II -> Stage III",
            "Stage II -> Stage IV"
        ]),
        "patient_rate_per_month"
    ].sum(),
    3: rates.loc[
        rates["transition"] == "Stage III -> Stage IV",
        "patient_rate_per_month"
    ].sum(),
    4: 0.0
}

exit_rate = exit_rates[current_stage]

if exit_rate > 0:
    mean_wait = 1 / exit_rate
    median_wait = 0.69314718056 / exit_rate

    waiting_text = (
        f"Mean model-estimated waiting time: {mean_wait:.2f} months.<br/>"
        f"Median model-estimated waiting time: {median_wait:.2f} months."
    )
else:
    waiting_text = (
        "Stage IV is treated as an absorbing state in this simplified model. "
        "A waiting-time estimate for leaving Stage IV is therefore not calculated."
    )

story.append(Paragraph(waiting_text, styles["BodyText"]))

# ---------------------------------------
# 11. Scientific limitations
# ---------------------------------------

story.append(Paragraph(
    "6. Important Scientific Limitations",
    styles["SectionHeading"]
))

limitations = """
<ul>
<li>This report is generated using a synthetic research dataset.</li>
<li>The model uses a simplified four-stage, forward-only progression structure.</li>
<li>Stage IV is assumed to be an absorbing state.</li>
<li>Estimated probabilities depend on the model assumptions and fitted parameters.</li>
<li>These estimates are not clinically validated predictions.</li>
<li>The report must not be used to diagnose cancer, determine treatment,
or replace advice from qualified oncology professionals.</li>
<li>Real-world validation using suitable longitudinal clinical data is
required before clinical interpretation.</li>
</ul>
"""

story.append(Paragraph(
    limitations,
    styles["BodyText"]
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "ONCOCAST AI | Academic Research Prototype",
    styles["SmallNote"]
))

# ---------------------------------------
# 12. Generate PDF
# ---------------------------------------

doc.build(story)

print("\n===================================")
print("ONCOCAST AI PDF REPORT GENERATED")
print("===================================")
print("Saved at:", pdf_file)
