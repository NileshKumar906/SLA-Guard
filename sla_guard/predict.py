import joblib
import pandas as pd
from pathlib import Path

from .config import ARTIFACTS


MODEL_PATH = ARTIFACTS / "model.joblib"


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Train the model first."
        )

    artifact = joblib.load(MODEL_PATH)

    return artifact


def risk_level(probability):

    if probability >= 0.70:
        return "HIGH"

    elif probability >= 0.40:
        return "MEDIUM"

    return "LOW"


def predict_incident(input_data):

    artifact = load_model()

    model = artifact["model"]

    feature_columns = artifact.get(
        "feature_columns",
        list(input_data.keys())
    )

    # Ensure all expected features exist
    input_df = pd.DataFrame([input_data])

    for column in feature_columns:

        if column not in input_df.columns:
            input_df[column] = None

    input_df = input_df[feature_columns]

    probability = model.predict_proba(input_df)[0][1]

    level = risk_level(float(probability))

    return {
        "breach_probability": float(probability),
        "risk_level": level,
        "prediction": (
            "Potential SLA breach"
            if probability >= artifact.get("threshold", 0.5)
            else "Likely to meet SLA"
        ),
        "threshold": artifact.get("threshold", 0.5),
        "model_name": artifact.get("model_name", "Unknown"),
    }


def get_feature_importance():

    artifact = load_model()

    model = artifact["model"]

    # Find final estimator in pipeline, if present
    estimator = model

    if hasattr(model, "named_steps"):

        steps = list(model.named_steps.values())

        estimator = steps[-1]

    # Tree-based models
    if hasattr(estimator, "feature_importances_"):

        importances = estimator.feature_importances_

        try:

            feature_names = model[:-1].get_feature_names_out()

        except Exception:

            feature_names = artifact.get(
                "feature_columns",
                []
            )

        if len(feature_names) == len(importances):

            result = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            })

            return result.sort_values(
                "importance",
                ascending=False
            )

    # Logistic Regression
    if hasattr(estimator, "coef_"):

        coefficients = estimator.coef_[0]

        try:

            feature_names = model[:-1].get_feature_names_out()

        except Exception:

            feature_names = artifact.get(
                "feature_columns",
                []
            )

        if len(feature_names) == len(coefficients):

            result = pd.DataFrame({
                "feature": feature_names,
                "coefficient": coefficients,
                "importance": abs(coefficients)
            })

            return result.sort_values(
                "importance",
                ascending=False
            )

    return pd.DataFrame({
        "message": [
            "Feature importance extraction is not supported "
            "for this model configuration."
        ]
    })


def main():

    print("\nSLA-Guard Prediction Demo\n")

    sample = {
        "contact_type": "Phone",
        "location": "Location 1",
        "category": "Software",
        "subcategory": "Internal Application",
        "u_symptom": "Symptom 1",
        "impact": "2",
        "urgency": "2",
        "priority": "3",
        "assignment_group": "Group 1",
        "knowledge": "false",
        "opened_month": 5,
        "opened_weekday": 2,
        "opened_hour": 10,
    }

    try:

        prediction = predict_incident(sample)

        print("Prediction:")
        print(
            f"Breach probability: "
            f"{prediction['breach_probability']:.2%}"
        )

        print("Risk level:", prediction["risk_level"])
        print("Prediction:", prediction["prediction"])
        print("Model:", prediction["model_name"])
        print("Threshold:", prediction["threshold"])

        print("\nFeature Importance:")

        importance = get_feature_importance()

        print(
            importance.head(15).to_string(
                index=False
            )
        )

    except Exception as error:

        print("\nPrediction failed:")
        print(error)


if __name__ == "__main__":
    main()