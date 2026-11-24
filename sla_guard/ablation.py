import json

import pandas as pd

from .config import REPORTS
from .data import load
from .features import group_incidents, make_X
from .train import run


FEATURE_GROUPS = {
    "metadata": [
        "category",
        "subcategory",
        "contact_type",
    ],
    "priority": [
        "impact",
        "urgency",
        "priority",
    ],
    "assignment": [
        "assignment_group",
    ],
    "symptom": [
        "u_symptom",
    ],
    "temporal": [
        "opened_month",
        "opened_weekday",
        "opened_hour",
    ],
}


def run_ablation():

    print("Loading dataset...")

    df, _ = load()

    incidents, target, ident, opened = group_incidents(df)

    X, y, _ = make_X(
        incidents,
        target,
        ident,
        opened
    )

    dates = (
        pd.to_datetime(
            incidents[opened],
            errors="coerce",
            utc=True
        )
        if opened
        else None
    )

    all_features = list(X.columns)

    configurations = {}

    # Full feature model
    configurations["Full model"] = all_features

    # Individual feature groups
    for group, features in FEATURE_GROUPS.items():

        configurations[f"{group.title()} only"] = [
            f for f in features if f in all_features
        ]

    # Remove one group at a time
    for group, features in FEATURE_GROUPS.items():

        configurations[f"Full minus {group}"] = [
            f for f in all_features if f not in features
        ]

    results = {}
    rows = []

    for name, features in configurations.items():

        print(f"\n{'=' * 60}")
        print(f"Experiment: {name}")
        print(f"Features: {len(features)}")
        print(f"{'=' * 60}")

        if not features:
            print("Skipping empty feature set.")
            continue

        X_subset = X[features].copy()

        experiment = run(
            X_subset,
            y,
            dates
        )

        selected_model = experiment["model_name"]

        selected_results = experiment["results"][
            selected_model
        ]

        validation = selected_results["validation"]
        test = selected_results["test"]

        results[name] = {
            "features": features,
            "feature_count": len(features),
            "selected_model": selected_model,
            "threshold": experiment["threshold"],
            "validation": validation,
            "test": test,
            "split": experiment["split_type"],
        }

        rows.append({
            "experiment": name,
            "feature_count": len(features),
            "selected_model": selected_model,
            "threshold": experiment["threshold"],
            "validation_pr_auc": validation["pr_auc"],
            "test_pr_auc": test["pr_auc"],
            "test_roc_auc": test["roc_auc"],
            "test_precision": test["precision"],
            "test_recall": test["recall"],
            "test_f1": test["f1"],
            "test_balanced_accuracy": test["balanced_accuracy"],
            "test_brier": test["brier"],
        })

        print("Selected model:", selected_model)
        print("Validation PR-AUC:", validation["pr_auc"])
        print("Test PR-AUC:", test["pr_auc"])
        print("Test Recall:", test["recall"])
        print("Test F1:", test["f1"])

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save complete experiment details
    with open(
        REPORTS / "ablation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            default=str
        )

    # Save concise comparison table
    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        by="validation_pr_auc",
        ascending=False
    )

    comparison.to_csv(
        REPORTS / "ablation_results.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("ABLATION EXPERIMENT COMPLETED")
    print("=" * 60)

    print(
        comparison[
            [
                "experiment",
                "selected_model",
                "validation_pr_auc",
                "test_pr_auc",
                "test_recall",
                "test_f1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print("reports/ablation_results.json")
    print("reports/ablation_results.csv")


if __name__ == "__main__":
    run_ablation()