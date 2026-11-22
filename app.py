import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.linalg import expm, solve
from scipy.optimize import brentq
from pathlib import Path
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ONCOCAST AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. DASHBOARD THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #081321;
    color: #E6F1F7;
}
[data-testid="stSidebar"] {
    background-color: #101F30;
}
h1, h2, h3, h4, p, label {
    color: #E6F1F7;
}
div[data-testid="stMetric"] {
    background-color: #122438;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #24435A;
}
.stButton > button {
    background-color: #087E8B;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
.stButton > button:hover {
    background-color: #0A9BA8;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. FILE LOCATIONS AND MODEL PARAMETERS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

RATES_FILE = OUTPUT_DIR / "ctmc_v2_rates.csv"
EFFECTS_FILE = OUTPUT_DIR / "ctmc_v2_effects.csv"

STAGE_NAMES = ["Stage I", "Stage II", "Stage III", "Stage IV"]

TRANSITION_NAMES = [
    "Stage I -> Stage II",
    "Stage I -> Stage III",
    "Stage II -> Stage III",
    "Stage II -> Stage IV",
    "Stage III -> Stage IV"
]

TIME_POINTS = [12, 24, 36, 60]


# =====================================================
# 4. LOAD MODEL PARAMETERS
# =====================================================

@st.cache_data
def load_model_parameters():

    if not RATES_FILE.exists():
        raise FileNotFoundError(
            "outputs/ctmc_v2_rates.csv is missing."
        )

    if not EFFECTS_FILE.exists():
        raise FileNotFoundError(
            "outputs/ctmc_v2_effects.csv is missing."
        )

    rates_df = pd.read_csv(RATES_FILE)
    effects_df = pd.read_csv(EFFECTS_FILE)

    if rates_df["transition"].tolist() != TRANSITION_NAMES:
        raise ValueError("Unexpected transition order in rates file.")

    expected_features = [
        "age_effect",
        "tumor_size_effect",
        "grade_effect",
        "node_positive",
        "HER2_positive",
        "treatment_Hormone",
        "treatment_Chemotherapy",
        "treatment_Combined"
    ]

    if effects_df["covariate"].tolist() != expected_features:
        raise ValueError("Unexpected covariate order in effects file.")

    base_rates = rates_df[
        "baseline_rate_per_month"
    ].to_numpy(float)

    beta = effects_df["coefficient"].to_numpy(float)

    if (
        len(base_rates) != 5
        or len(beta) != 8
        or not np.all(np.isfinite(base_rates))
        or not np.all(np.isfinite(beta))
        or np.any(base_rates <= 0)
    ):
        raise ValueError("Invalid model parameter values.")

    return base_rates, beta, effects_df


# =====================================================
# 5. PATIENT-SPECIFIC CTMC CALCULATION
# =====================================================

def calculate_patient_prediction(
    age,
    tumor_size,
    grade,
    node_positive,
    her2_positive,
    treatment,
    current_stage,
    base_rates,
    beta
):

    treatment_features = [
        1.0 if treatment == "Hormone" else 0.0,
        1.0 if treatment == "Chemotherapy" else 0.0,
        1.0 if treatment == "Combined" else 0.0
    ]

    features = np.array([
        (age - 50) / 20,
        (tumor_size - 2.5) / 2,
        grade - 2,
        1.0 if node_positive else 0.0,
        1.0 if her2_positive else 0.0,
        *treatment_features
    ])

    linear_effect = np.dot(features, beta)
    multiplier = np.exp(np.clip(linear_effect, -10, 10))
    patient_rates = base_rates * multiplier

    q12, q13, q23, q24, q34 = patient_rates

    Q = np.array([
        [-(q12 + q13), q12, q13, 0.0],
        [0.0, -(q23 + q24), q23, q24],
        [0.0, 0.0, -q34, q34],
        [0.0, 0.0, 0.0, 0.0]
    ])

    start_index = current_stage - 1
    results = []

    for months in TIME_POINTS:

        P = expm(Q * months)

        probabilities = np.clip(
            P[start_index, :],
            0,
            1
        )

        probabilities = probabilities / probabilities.sum()

        row = {"Months": months}

        for i, stage in enumerate(STAGE_NAMES):
            row[stage] = probabilities[i] * 100

        results.append(row)

    probability_df = pd.DataFrame(results)

    exit_rate = -Q[start_index, start_index]

    if exit_rate > 0:
        mean_wait = 1 / exit_rate
        median_wait = np.log(2) / exit_rate
    else:
        mean_wait = np.inf
        median_wait = np.inf

    return (
        Q,
        patient_rates,
        probability_df,
        mean_wait,
        median_wait
    )


# =====================================================
# 6. FIRST-PASSAGE PROGRESSION ANALYSIS
# =====================================================

def calculate_first_passage(Q, current_stage, horizon):

    current_index = current_stage - 1
    results = []

    for target_index in range(current_index + 1, 4):

        target_stage = target_index + 1

        pre_target_indices = list(
            range(current_index, target_index)
        )

        T = Q[np.ix_(
            pre_target_indices,
            pre_target_indices
        )]

        r = Q[pre_target_indices, target_index]

        alpha = np.zeros(len(pre_target_indices))
        alpha[0] = 1.0

        try:

            h = solve(-T, r)

            eventual_probability = float(alpha @ h)

            eventual_probability = float(
                np.clip(eventual_probability, 0, 1)
            )

            if eventual_probability <= 1e-12:

                results.append({
                    "Destination": f"Stage {target_stage}",
                    "Probability within horizon (%)": 0.0,
                    "Probability eventually (%)": 0.0,
                    "Mean first-passage time (months)": np.nan,
                    "Median first-passage time (months)": np.nan
                })

                continue

            conditional_moment_numerator = float(
                alpha @ solve(-T, h)
            )

            conditional_mean = (
                conditional_moment_numerator /
                eventual_probability
            )

            def first_passage_cdf(t):

                remaining = float(
                    alpha @ expm(T * t) @ h
                )

                return float(np.clip(
                    eventual_probability - remaining,
                    0,
                    eventual_probability
                ))

            horizon_probability = first_passage_cdf(horizon)

            median_target = eventual_probability / 2
            upper = max(1.0, conditional_mean * 2)

            while first_passage_cdf(upper) < median_target:

                upper *= 2

                if upper > 1e9:
                    raise ValueError(
                        "Could not calculate first-passage median."
                    )

            conditional_median = brentq(
                lambda t: (
                    first_passage_cdf(t) - median_target
                ),
                0,
                upper
            )

            results.append({
                "Destination": f"Stage {target_stage}",
                "Probability within horizon (%)":
                    horizon_probability * 100,
                "Probability eventually (%)":
                    eventual_probability * 100,
                "Mean first-passage time (months)":
                    conditional_mean,
                "Median first-passage time (months)":
                    conditional_median
            })

        except (np.linalg.LinAlgError, ValueError):

            results.append({
                "Destination": f"Stage {target_stage}",
                "Probability within horizon (%)": np.nan,
                "Probability eventually (%)": np.nan,
                "Mean first-passage time (months)": np.nan,
                "Median first-passage time (months)": np.nan
            })

    return pd.DataFrame(results)
# =====================================================
# 7. PDF REPORT GENERATION
# =====================================================

def create_pdf_report(
    profile,
    rates_df,
    probability_df,
    mean_wait,
    median_wait,
    first_passage_df,
    horizon
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ONCOCAST AI", styles["Title"]))
    story.append(Paragraph(
        "Patient-Specific Progression Research Report",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph("1. Patient Profile", styles["Heading2"]))

    profile_data = [
        ["Parameter", "Value"],
        ["Age", f"{profile['age']} years"],
        ["Tumour size", f"{profile['tumor_size_cm']} cm"],
        ["Grade", str(profile["grade"])],
        ["Lymph node positive", str(profile["node_positive"])],
        ["HER2 positive", str(profile["HER2_positive"])],
        ["Treatment", profile["treatment"]],
        ["Current stage", STAGE_NAMES[profile["current_stage"] - 1]]
    ]

    story.append(Table(profile_data, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7)
    ])))
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "2. Estimated Stage Occupancy Probabilities",
        styles["Heading2"]
    ))
    story.append(Paragraph(
        "These values show the estimated probability of occupying "
        "each stage at the specified future time points.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    probability_data = [[
        "Months", "Stage I", "Stage II", "Stage III", "Stage IV"
    ]]

    for _, row in probability_df.iterrows():
        probability_data.append([
            str(int(row["Months"])),
            *[f"{row[s]:.2f}%" for s in STAGE_NAMES]
        ])

    story.append(Table(probability_data, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ])))
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "3. Patient-Specific Transition Rates",
        styles["Heading2"]
    ))
    story.append(Paragraph(
        "A transition rate describes the modelled rate of movement "
        "from one stage to another. These are rates per month.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    rate_data = [["Transition", "Rate per month"]]

    for _, row in rates_df.iterrows():
        rate_data.append([
            row["transition"],
            f"{row['patient_rate_per_month']:.6f}"
        ])

    story.append(Table(rate_data, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6)
    ])))
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "4. Current Stage Waiting Time",
        styles["Heading2"]
    ))

    if np.isfinite(mean_wait):

        waiting_data = [
            ["Measure", "Estimated duration"],
            ["Mean time to leave current stage", f"{mean_wait:.2f} months"],
            ["Median time to leave current stage", f"{median_wait:.2f} months"]
        ]

        story.append(Table(waiting_data, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])))

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "These are the mean and median of the modelled time until "
            "the patient leaves the current stage through any permitted "
            "outgoing transition. They are not estimates of the time "
            "until Stage IV and do not specify an exact progression date.",
            styles["BodyText"]
        ))

    else:
        story.append(Paragraph(
            "Stage IV is absorbing in this simplified model, so no "
            "outgoing-stage waiting time is calculated.",
            styles["BodyText"]
        ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "5. First-Passage Progression Analysis",
        styles["Heading2"]
    ))
    story.append(Paragraph(
        f"This section estimates the probability of reaching each "
        f"future stage for the first time within {horizon} months. "
        "Mean and median first-passage times are conditional on "
        "eventually reaching the relevant destination.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    if first_passage_df.empty:

        story.append(Paragraph(
            "There are no later stages reachable from the current stage "
            "in the simplified transition structure.",
            styles["BodyText"]
        ))

    else:

        fp_data = [[
            "Destination",
            f"By {horizon} mo.",
            "Eventually",
            "Mean time",
            "Median time"
        ]]

        for _, row in first_passage_df.iterrows():

            fp_data.append([
                row["Destination"],
                (
                    f"{row['Probability within horizon (%)']:.2f}%"
                    if pd.notna(row["Probability within horizon (%)"])
                    else "N/A"
                ),
                (
                    f"{row['Probability eventually (%)']:.2f}%"
                    if pd.notna(row["Probability eventually (%)"])
                    else "N/A"
                ),
                (
                    f"{row['Mean first-passage time (months)']:.2f}"
                    if pd.notna(row["Mean first-passage time (months)"])
                    else "N/A"
                ),
                (
                    f"{row['Median first-passage time (months)']:.2f}"
                    if pd.notna(row["Median first-passage time (months)"])
                    else "N/A"
                )
            ])

        fp_table = Table(fp_data, repeatRows=1)

        fp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B52")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))

        story.append(fp_table)

        valid = first_passage_df.dropna(
            subset=["Probability within horizon (%)"]
        )

        if not valid.empty:

            best = valid.loc[
                valid["Probability within horizon (%)"].idxmax()
            ]

            story.append(Spacer(1, 8))

            story.append(Paragraph(
                f"The destination with the highest individual "
                f"first-passage probability within {horizon} months is "
                f"{best['Destination']} "
                f"({best['Probability within horizon (%)']:.2f}%). "
                "This is the largest individual destination probability; "
                "it does not imply that the destination is more likely "
                "than all other outcomes combined.",
                styles["BodyText"]
            ))

    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "First-passage probability is different from stage occupancy "
        "probability. Occupancy describes the stage at a particular time; "
        "first-passage probability describes whether a stage has been "
        "reached at least once by that time. A patient may bypass an "
        "intermediate stage in this model.",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "6. Scientific Limitations",
        styles["Heading2"]
    ))

    story.append(Paragraph(
        "This report uses a synthetic research dataset and a simplified "
        "four-state forward-only continuous-time Markov model. Stage IV "
        "is treated as an absorbing state. The model uses a shared "
        "covariate multiplier for all five transition rates. Treatment "
        "effects are simulated associations, not causal estimates. "
        "The estimates are not clinically validated predictions and "
        "must not be used for diagnosis or treatment decisions. "
        "Real-world longitudinal clinical validation is required.",
        styles["BodyText"]
    ))

    doc.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# =====================================================
# 8. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🧬 ONCOCAST AI")
st.sidebar.caption("Breast Cancer Progression Research")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Patient Analysis",
        "Analysis Results",
        "Research Methodology"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Research prototype | CTMC framework")


# =====================================================
# 9. LOAD MODEL
# =====================================================

try:

    BASE_RATES, BETA, EFFECTS_DF = load_model_parameters()
    model_available = True

except Exception as error:

    model_available = False
    model_error = str(error)
# =====================================================
# 10. HOME PAGE
# =====================================================

if page == "Home":

    st.title("🧬 ONCOCAST AI")
    st.subheader(
        "Understanding Progression. Estimating What Comes Next."
    )

    st.markdown("""
    ONCOCAST AI is a breast cancer progression research prototype
    based on a covariate-dependent Continuous-Time Markov Chain (CTMC).

    It explores how patient characteristics may influence modelled
    stage-transition rates and future stage probabilities.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Disease States", "4")

    with col2:
        st.metric("Modelled Transitions", "5")

    with col3:
        st.metric("Prediction Horizons", "12–60 months")

    st.divider()

    st.subheader("How ONCOCAST AI Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 01")
        st.markdown("**Patient Profile**")
        st.write(
            "Enter the patient's available characteristics "
            "and current disease stage."
        )

    with step2:
        st.markdown("### 02")
        st.markdown("**Stochastic Model**")
        st.write(
            "The CTMC estimates patient-specific transition "
            "rates and stage probabilities."
        )

    with step3:
        st.markdown("### 03")
        st.markdown("**Progression Report**")
        st.write(
            "Explore future stage occupancy, first-passage "
            "probabilities and conditional transition times."
        )

    st.divider()

    st.warning(
        "Research prototype only. Outputs are based on synthetic "
        "data and are not clinical predictions or medical advice."
    )


# =====================================================
# 11. PATIENT ANALYSIS PAGE
# =====================================================

elif page == "Patient Analysis":

    st.title("Patient Analysis")
    st.write(
        "Enter the patient profile to generate a model-based "
        "progression analysis."
    )

    if not model_available:

        st.error(
            f"Model files could not be loaded: {model_error}"
        )

    else:

        with st.form("patient_form"):

            st.subheader("Patient Information")

            col1, col2 = st.columns(2)

            with col1:

                age = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=90,
                    value=50
                )

                tumor_size = st.number_input(
                    "Tumour size (cm)",
                    min_value=0.1,
                    max_value=20.0,
                    value=2.5,
                    step=0.1
                )

                grade = st.selectbox(
                    "Tumour grade",
                    [1, 2, 3],
                    index=1
                )

                current_stage = st.selectbox(
                    "Current cancer stage",
                    [1, 2, 3, 4],
                    format_func=lambda x: STAGE_NAMES[x - 1]
                )

            with col2:

                node_positive = st.selectbox(
                    "Lymph node status",
                    ["No", "Yes"]
                )

                her2_positive = st.selectbox(
                    "HER2 status",
                    ["No", "Yes"]
                )

                treatment = st.selectbox(
                    "Treatment category",
                    [
                        "No_Treatment",
                        "Hormone",
                        "Chemotherapy",
                        "Combined"
                    ]
                )

                horizon = st.selectbox(
                    "First-passage analysis horizon",
                    [12, 24, 36, 60],
                    index=3,
                    format_func=lambda x: f"{x} months"
                )

            submitted = st.form_submit_button(
                "Generate Patient Analysis",
                use_container_width=True
            )

        if submitted:

            try:

                (
                    Q,
                    patient_rates,
                    probability_df,
                    mean_wait,
                    median_wait
                ) = calculate_patient_prediction(
                    age=age,
                    tumor_size=tumor_size,
                    grade=grade,
                    node_positive=(node_positive == "Yes"),
                    her2_positive=(her2_positive == "Yes"),
                    treatment=treatment.replace("_", " "),
                    current_stage=current_stage,
                    base_rates=BASE_RATES,
                    beta=BETA
                )

                # Store the original treatment category expected
                # by the model feature construction.
                treatment_for_model = treatment

                first_passage_df = calculate_first_passage(
                    Q,
                    current_stage,
                    horizon
                )

                rates_df = pd.DataFrame({
                    "transition": TRANSITION_NAMES,
                    "patient_rate_per_month": patient_rates
                })

                profile = {
                    "age": age,
                    "tumor_size_cm": tumor_size,
                    "grade": grade,
                    "node_positive": node_positive,
                    "HER2_positive": her2_positive,
                    "treatment": treatment_for_model,
                    "current_stage": current_stage
                }

                st.session_state["patient_result"] = {
                    "profile": profile,
                    "Q": Q,
                    "rates_df": rates_df,
                    "probability_df": probability_df,
                    "mean_wait": mean_wait,
                    "median_wait": median_wait,
                    "first_passage_df": first_passage_df,
                    "horizon": horizon
                }

                st.success(
                    "Patient analysis generated successfully."
                )

                st.info(
                    "Open the Analysis Results page from the sidebar "
                    "to view the complete progression analysis."
                )

            except Exception as error:

                st.error(
                    f"Unable to generate the analysis: {error}"
                )
# =====================================================
# 12. ANALYSIS RESULTS PAGE
# =====================================================

elif page == "Analysis Results":

    st.title("Patient Analysis Results")

    if "patient_result" not in st.session_state:

        st.info(
            "No patient analysis is available yet. "
            "Go to Patient Analysis and submit a patient profile first."
        )

    else:

        result = st.session_state["patient_result"]

        profile = result["profile"]
        Q = result["Q"]
        rates_df = result["rates_df"]
        probability_df = result["probability_df"]
        mean_wait = result["mean_wait"]
        median_wait = result["median_wait"]
        first_passage_df = result["first_passage_df"]
        horizon = result["horizon"]

        st.subheader("Patient Profile")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Age", f"{profile['age']} years")
        col2.metric("Tumour Size", f"{profile['tumor_size_cm']} cm")
        col3.metric("Grade", profile["grade"])
        col4.metric(
            "Current Stage",
            STAGE_NAMES[profile["current_stage"] - 1]
        )

        st.divider()

        # ---------------------------------------------
        # A. STAGE OCCUPANCY PROBABILITIES
        # ---------------------------------------------

        st.subheader("1. Future Stage Probabilities")

        st.write(
            "These probabilities estimate the stage occupied "
            "at each selected future time point."
        )

        display_df = probability_df.copy()

        for stage in STAGE_NAMES:
            display_df[stage] = display_df[stage].map(
                lambda x: f"{x:.2f}%"
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        fig = go.Figure()

        for stage in STAGE_NAMES:

            fig.add_trace(go.Scatter(
                x=probability_df["Months"],
                y=probability_df[stage],
                mode="lines+markers",
                name=stage
            ))

        fig.update_layout(
            title="Stage Occupancy Probability Over Time",
            xaxis_title="Time (months)",
            yaxis_title="Probability (%)",
            template="plotly_dark",
            hovermode="x unified",
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Occupancy probability describes the modelled stage "
            "at a particular time. It is not the same as the "
            "probability of having reached that stage at any time "
            "before that point."
        )

        st.divider()

        # ---------------------------------------------
        # B. TRANSITION RATES
        # ---------------------------------------------

        st.subheader("2. Patient-Specific Transition Rates")

        st.write(
            "These rates describe the modelled rate of each "
            "permitted transition, measured per month."
        )

        st.dataframe(
            rates_df.style.format({
                "patient_rate_per_month": "{:.6f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        fig_rates = go.Figure()

        fig_rates.add_trace(go.Bar(
            x=rates_df["transition"],
            y=rates_df["patient_rate_per_month"],
            text=[
                f"{x:.5f}" for x in
                rates_df["patient_rate_per_month"]
            ],
            textposition="outside"
        ))

        fig_rates.update_layout(
            title="Patient-Specific Transition Rates",
            xaxis_title="Transition",
            yaxis_title="Rate per month",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig_rates,
            use_container_width=True
        )

        st.divider()

        # ---------------------------------------------
        # C. WAITING TIME IN CURRENT STAGE
        # ---------------------------------------------

        st.subheader("3. Time Until Leaving Current Stage")

        if np.isfinite(mean_wait):

            col1, col2 = st.columns(2)

            col1.metric(
                "Mean Waiting Time",
                f"{mean_wait:.2f} months"
            )

            col2.metric(
                "Median Waiting Time",
                f"{median_wait:.2f} months"
            )

            st.info(
                "This estimates the time until leaving the current "
                "stage through any permitted outgoing transition. "
                "It is not the time until reaching Stage IV."
            )

        else:

            st.info(
                "Stage IV is absorbing in this model. "
                "There is no outgoing transition from Stage IV."
            )

        st.divider()

        # ---------------------------------------------
        # D. FIRST-PASSAGE ANALYSIS
        # ---------------------------------------------

        st.subheader("4. First-Passage Progression Analysis")

        st.write(
            f"Probability of reaching each future stage for "
            f"the first time within {horizon} months."
        )

        if first_passage_df.empty:

            st.info(
                "There are no future stages reachable from "
                "the selected current stage in this model."
            )

        else:

            result_display = first_passage_df.copy()

            for column in [
                "Probability within horizon (%)",
                "Probability eventually (%)"
            ]:

                result_display[column] = (
                    result_display[column].map(
                        lambda x: (
                            f"{x:.2f}%"
                            if pd.notna(x)
                            else "N/A"
                        )
                    )
                )

            for column in [
                "Mean first-passage time (months)",
                "Median first-passage time (months)"
            ]:

                result_display[column] = (
                    result_display[column].map(
                        lambda x: (
                            f"{x:.2f} months"
                            if pd.notna(x)
                            else "N/A"
                        )
                    )
                )

            st.dataframe(
                result_display,
                use_container_width=True,
                hide_index=True
            )

            valid = first_passage_df.dropna(
                subset=["Probability within horizon (%)"]
            )

            if not valid.empty:

                best = valid.loc[
                    valid["Probability within horizon (%)"].idxmax()
                ]

                st.success(
                    f"Highest individual first-passage probability "
                    f"within {horizon} months: "
                    f"{best['Destination']} "
                    f"({best['Probability within horizon (%)']:.2f}%)."
                )

            st.caption(
                "First-passage probability measures whether a stage "
                "has been reached at least once by the selected time. "
                "Mean and median first-passage times are conditional "
                "on eventually reaching that destination."
            )

        st.divider()

        # ---------------------------------------------
        # E. PDF REPORT
        # ---------------------------------------------

        st.subheader("5. Download Patient Report")

        pdf_bytes = create_pdf_report(
            profile,
            rates_df,
            probability_df,
            mean_wait,
            median_wait,
            first_passage_df,
            horizon
        )

        st.download_button(
            label="Download Patient Report (PDF)",
            data=pdf_bytes,
            file_name="ONCOCAST_AI_Patient_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =====================================================
# 13. RESEARCH METHODOLOGY PAGE
# =====================================================

elif page == "Research Methodology":

    st.title("Research Methodology")

    st.subheader("Model Overview")

    st.write("""
    ONCOCAST AI uses a covariate-dependent Continuous-Time
    Markov Chain (CTMC) to represent breast cancer stage
    progression across four simplified disease states.
    """)

    st.latex(
        r"P(t)=e^{Qt}"
    )

    st.subheader("Four Disease States")

    for stage in STAGE_NAMES:

        st.write(f"• {stage}")

    st.subheader("Permitted Transitions")

    transition_data = pd.DataFrame({
        "Transition": TRANSITION_NAMES,
        "Description": [
            "Stage I to Stage II",
            "Stage I to Stage III",
            "Stage II to Stage III",
            "Stage II to Stage IV",
            "Stage III to Stage IV"
        ]
    })

    st.dataframe(
        transition_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Covariates Used")

    st.write("""
    - Age
    - Tumour size
    - Tumour grade
    - Lymph node positivity
    - HER2 status
    - Treatment category
    """)

    st.subheader("Covariate-Dependent Transition Rates")

    st.latex(
        r"q_{ij}(x)=q_{ij}^{(0)}\exp(\beta^T x)"
    )

    st.write("""
    The baseline transition rate is modified by a shared
    covariate-dependent multiplier. This follows the structure
    used to generate the synthetic research dataset.
    """)

    st.subheader("First-Passage Analysis")

    st.write("""
    First-passage analysis estimates the probability and time
    associated with reaching a destination stage for the first
    time. It is different from estimating the stage occupied
    at a particular future time.
    """)

    st.subheader("Important Limitations")

    st.warning("""
    1. The current dataset is synthetic.
    2. The four disease states are simplified stage categories.
    3. Stage IV is assumed to be an absorbing state.
    4. Treatment effects are simulated associations, not causal effects.
    5. Real-world clinical validation is not yet established.
    6. The dashboard is a research prototype, not a diagnostic tool.
    7. Outputs must not be used to make patient treatment decisions.
    """)

    st.subheader("Research Objective")

    st.write("""
    The objective is to study how patient characteristics may
    influence breast cancer stage-transition rates and to estimate
    future stage occupancy, first-passage probabilities and
    conditional first-passage times using a stochastic framework.
    """)