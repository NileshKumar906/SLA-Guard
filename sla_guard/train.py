
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.model_selection import train_test_split

from .evaluate import metrics, threshold_f1
from .config import SEED


def preprocess(X):

    categorical_columns = list(X.columns)

    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2
            )
        ),
    ])

    return ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])


def split(X, y, dates):

    if (
        dates is not None
        and dates.notna().sum() >= max(10, int(0.5 * len(dates)))
    ):

        indices = np.argsort(
            dates.fillna(pd.Timestamp.min)
            .astype("int64")
            .values
        )

        n = len(indices)

        train_end = int(0.70 * n)
        validation_end = int(0.85 * n)

        train_indices = indices[:train_end]
        validation_indices = indices[train_end:validation_end]
        test_indices = indices[validation_end:]

        return (
            train_indices,
            validation_indices,
            test_indices,
            "chronological"
        )

    # Fallback if dates are unavailable.
    a, b = train_test_split(
        np.arange(len(y)),
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    validation_indices, test_indices = train_test_split(
        b,
        test_size=0.50,
        random_state=SEED,
        stratify=np.asarray(y)[b]
    )

    return (
        a,
        validation_indices,
        test_indices,
        "stratified_fallback"
    )


def run(X, y, dates=None):

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    train_indices, validation_indices, test_indices, split_type = split(
        X,
        y,
        dates
    )

    specifications = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED
        )
    }

    trained_models = {}
    results = {}

    for name, estimator in specifications.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("prep", preprocess(X)),
            ("model", estimator)
        ])

        pipeline.fit(
            X.iloc[train_indices],
            y.iloc[train_indices]
        )

        # Validation predictions
        validation_probabilities = pipeline.predict_proba(
            X.iloc[validation_indices]
        )[:, 1]

        # Threshold chosen ONLY on validation
        threshold = threshold_f1(
            y.iloc[validation_indices],
            validation_probabilities
        )

        validation_metrics = metrics(
            y.iloc[validation_indices],
            validation_probabilities,
            threshold
        )

        # Test predictions
        test_probabilities = pipeline.predict_proba(
            X.iloc[test_indices]
        )[:, 1]

        test_metrics = metrics(
            y.iloc[test_indices],
            test_probabilities,
            threshold
        )

        results[name] = {
            "validation": validation_metrics,
            "test": test_metrics
        }

        trained_models[name] = pipeline

        print("Validation PR-AUC:",
              validation_metrics["pr_auc"])

        print("Test PR-AUC:",
              test_metrics["pr_auc"])

        print("Test Recall:",
              test_metrics["recall"])

        print("Test F1:",
              test_metrics["f1"])

        print("Threshold:",
              threshold)

    # Select model based on validation PR-AUC.
    valid_models = [
        name for name in results
        if results[name]["validation"]["pr_auc"] is not None
    ]

    if not valid_models:
        raise RuntimeError(
            "No model could be selected using validation PR-AUC."
        )

    best_model_name = max(
        valid_models,
        key=lambda name: results[name]["validation"]["pr_auc"]
    )

    best_model = trained_models[best_model_name]

    best_threshold = results[
        best_model_name
    ]["validation"]["threshold"]

    # Simple baseline:
    # constant probability equal to training breach rate.
    train_breach_rate = float(
        y.iloc[train_indices].mean()
    )

    test_y = y.iloc[test_indices]

    baseline_probabilities = np.full(
        len(test_indices),
        train_breach_rate
    )

    baseline_threshold = 0.5

    baseline_metrics = metrics(
        test_y,
        baseline_probabilities,
        baseline_threshold
    )

    return {
        "model": best_model,
        "models": trained_models,
        "model_name": best_model_name,
        "results": results,
        "split_type": split_type,
        "threshold": best_threshold,
        "feature_columns": list(X.columns),
        "baseline": {
            "name": "Constant Training Prevalence",
            "training_breach_rate": train_breach_rate,
            "test": baseline_metrics
        }
    }
