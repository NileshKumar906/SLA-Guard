
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from sla_guard.predict import predict_incident, get_feature_importance


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="SLA-Guard",
    page_icon="🛡️",
    layout="wide"
)


# ---------- Styling ----------
st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.2);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def load_metrics():

    path = REPORTS / "metrics.json"

    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ablation():

    path = REPORTS / "ablation_results.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


def format_metric(value):

    if value is None:
        return "N/A"

    return f"{value:.3f}"


# ---------- Sidebar ----------
st.sidebar.title("🛡️ SLA-Guard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Risk Prediction",
        "Model Insights",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Early Incident SLA Risk Intelligence"
)


# ---------- Data ----------
metrics = load_metrics()
ablation = load_ablation()


# ---------- Overview ----------
if page == "Overview":

    st.title("🛡️ SLA-Guard")
    st.subheader("Early Incident SLA Risk Intelligence")

    st.markdown(
        """
        SLA-Guard uses machine learning to estimate the risk
        of an IT incident failing to meet its SLA.

        The system analyzes incident information and produces
        a breach-risk probability to support prioritization.
        """
    )

    if metrics:

        selected_model = metrics.get(
            "selected_model",
            "Unknown"
        )

        result = metrics.get(
            "results",
            {}
        ).get(selected_model, {})

        test = result.get("test", {})

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Selected Model",
            selected_model
        )

        c2.metric(
            "Test PR-AUC",
            format_metric(test.get("pr_auc"))
        )

        c3.metric(
            "Test Recall",
            format_metric(test.get("recall"))
        )

        c4.metric(
            "Test F1",
            format_metric(test.get("f1"))
        )

        st.markdown("---")

        st.subheader("Model Performance")

        performance = pd.DataFrame({
            "Metric": [
                "PR-AUC",
                "ROC-AUC",
                "Precision",
                "Recall",
                "F1",
                "Balanced Accuracy"
            ],
            "Value": [
                test.get("pr_auc"),
                test.get("roc_auc"),
                test.get("precision"),
                test.get("recall"),
                test.get("f1"),
                test.get("balanced_accuracy")
            ]
        })

        performance = performance.dropna()

        fig = px.bar(
            performance,
            x="Metric",
            y="Value",
            title="Selected Model — Test Metrics",
            range_y=[0, 1]
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.warning(
            "Training metrics were not found. "
            "Run the training pipeline first."
        )

    st.markdown("---")

    st.subheader("Project Workflow")

    st.markdown("""
    1. Incident data preprocessing
    2. Incident-level aggregation
    3. Feature engineering
    4. Chronological model evaluation
    5. SLA breach-risk prediction
    6. Risk-informed incident prioritization
    """)

    st.info(
        "This project uses historical incident event-log data. "
        "Its earliest-event feature snapshot is a proxy for "
        "prediction-time availability, not proof of actual "
        "real-time availability."
    )


# ---------- Prediction ----------
elif page == "Risk Prediction":

    st.title("🔍 Incident Risk Prediction")

    st.write(
        "Enter incident information to estimate its "
        "probability of an SLA breach."
    )

    with st.form("incident_form"):

        col1, col2 = st.columns(2)

        with col1:

            contact_type = st.selectbox(
                "Contact Type",
                ["Phone", "Email", "Self-service", "Other"]
            )

            location = st.text_input(
                "Location",
                "Location 1"
            )

            category = st.text_input(
                "Category",
                "Software"
            )

            subcategory = st.text_input(
                "Subcategory",
                "Internal Application"
            )

            symptom = st.text_input(
                "Symptom",
                "Symptom 1"
            )

            assignment_group = st.text_input(
                "Assignment Group",
                "Group 1"
            )

        with col2:

            impact = st.selectbox(
                "Impact",
                ["1", "2", "3"]
            )

            urgency = st.selectbox(
                "Urgency",
                ["1", "2", "3"]
            )

            priority = st.selectbox(
                "Priority",
                ["1", "2", "3", "4", "5"]
            )

            knowledge = st.selectbox(
                "Knowledge Available",
                ["false", "true"]
            )

            opened_month = st.selectbox(
                "Opened Month",
                list(range(1, 13)),
                index=4
            )

            opened_weekday = st.selectbox(
                "Weekday (0=Monday)",
                list(range(7)),
                index=2
            )

            opened_hour = st.slider(
                "Hour",
                0,
                23,
                10
            )

        submitted = st.form_submit_button(
            "Predict SLA Breach Risk",
            width="stretch"
        )

    if submitted:

        input_data = {
            "contact_type": contact_type,
            "location": location,
            "category": category,
            "subcategory": subcategory,
            "u_symptom": symptom,
            "impact": impact,
            "urgency": urgency,
            "priority": priority,
            "assignment_group": assignment_group,
            "knowledge": knowledge,
            "opened_month": opened_month,
            "opened_weekday": opened_weekday,
            "opened_hour": opened_hour
        }

        try:

            result = predict_incident(input_data)

            probability = result[
                "breach_probability"
            ]

            st.markdown("---")

            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Breach Probability",
                f"{probability:.1%}"
            )

            col2.metric(
                "Risk Category",
                result["risk_level"]
            )

            col3.metric(
                "Decision",
                result["prediction"]
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if result["risk_level"] == "HIGH":

                st.error(
                    "High estimated breach risk. "
                    "Consider reviewing this incident."
                )

            elif result["risk_level"] == "MEDIUM":

                st.warning(
                    "Moderate estimated breach risk. "
                    "Consider monitoring this incident."
                )

            else:

                st.success(
                    "Lower estimated breach risk. "
                    "Continue normal monitoring."
                )

            st.caption(
                f"Decision threshold: "
                f"{result['threshold']:.2f}"
            )

            st.caption(
                "Risk bands are provisional and have not "
                "been operationally validated."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ---------- Model Insights ----------
elif page == "Model Insights":

    st.title("📊 Model Insights")

    st.write(
        "Explore the influence of features in the "
        "selected Logistic Regression model."
    )

    try:

        importance = get_feature_importance()

        if "coefficient" in importance.columns:

            importance = importance.head(20)

            fig = px.bar(
                importance,
                x="coefficient",
                y="feature",
                orientation="h",
                title="Top Feature Coefficients"
            )

            fig.update_layout(
                yaxis={
                    "categoryorder": "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            st.dataframe(
                importance,
                width="stretch"
            )

            st.info(
                "Positive Logistic Regression coefficients "
                "increase the model's log-odds of breach, "
                "while negative coefficients decrease them, "
                "relative to the encoded feature reference "
                "structure. These are associations, not "
                "causal explanations."
            )

        else:

            st.warning(
                "Coefficient-based explanation "
                "is unavailable."
            )

    except Exception as e:

        st.error(
            f"Could not load model insights: {e}"
        )


# ---------- Evaluation ----------
elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")

    if metrics:

        results = metrics.get(
            "results",
            {}
        )

        rows = []

        for model_name, result in results.items():

            test = result.get(
                "test",
                {}
            )

            rows.append({
                "Model": model_name,
                "PR-AUC": test.get("pr_auc"),
                "ROC-AUC": test.get("roc_auc"),
                "Precision": test.get("precision"),
                "Recall": test.get("recall"),
                "F1": test.get("f1"),
                "Balanced Accuracy": test.get(
                    "balanced_accuracy"
                ),
                "Brier": test.get("brier")
            })

        comparison = pd.DataFrame(rows)

        st.subheader("Model Comparison")

        st.dataframe(
            comparison,
            width="stretch"
        )

        if not comparison.empty:

            fig = px.bar(
                comparison,
                x="Model",
                y="PR-AUC",
                title="Test PR-AUC Comparison",
                range_y=[0, 1]
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

    else:

        st.warning(
            "Metrics file not found."
        )

    if ablation is not None:

        st.markdown("---")

        st.subheader("Feature Ablation Analysis")

        st.dataframe(
            ablation,
            width="stretch"
        )

        fig = px.bar(
            ablation,
            x="experiment",
            y="test_pr_auc",
            title="Feature Ablation — Test PR-AUC"
        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.markdown("---")

    st.caption(
        "The test set is held out for evaluation. "
        "Differences in test results across exploratory "
        "ablation runs should not be used to retrospectively "
        "select the final model."
    )
