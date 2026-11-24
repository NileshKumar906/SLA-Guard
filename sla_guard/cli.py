
import pandas as pd
import argparse
import json
import joblib

from .config import RAW, PROCESSED, ARTIFACTS, REPORTS
from .data import download, load, audit
from .features import group_incidents, make_X
from .train import run


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8"
    )


def do_audit():

    df, path = load()

    result = audit(df)
    result["source_file"] = str(path)

    write(
        REPORTS / "dataset_audit.json",
        result
    )

    print(
        f"Audit saved; {result['rows']} rows, "
        f"{result['columns']} columns"
    )


def do_train():

    df, path = load()

    incidents, target, ident, opened = group_incidents(df)

    X, y, selected = make_X(
        incidents,
        target,
        ident,
        opened
    )

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    incidents.to_csv(
        PROCESSED / "incidents.csv",
        index=False
    )

    X.assign(_target=y).to_csv(
        PROCESSED / "features.csv",
        index=False
    )

    write(
        REPORTS / "feature_audit.json",
        {
            "target": target,
            "incident_id": ident,
            "opening_time": opened,
            "features": list(X.columns),
            "selected_columns": selected,
            "feature_count": X.shape[1],
            "incident_count": len(X),
            "caveat": (
                "The earliest event row is used as a proxy "
                "for prediction-time features. The event log "
                "does not establish actual point-in-time "
                "feature availability."
            )
        }
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

    result = run(X, y, dates)

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        result,
        ARTIFACTS / "model.joblib"
    )

    write(
        REPORTS / "metrics.json",
        {
            "selected_model": result["model_name"],
            "split": result["split_type"],
            "results": result["results"],
            "threshold": result["threshold"],
            "baseline": result["baseline"]
        }
    )

    print("\nSelected model:", result["model_name"])
    print("Threshold:", result["threshold"])
    print("Metrics and artifact saved.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["download", "audit", "train"]
    )

    args = parser.parse_args()

    if args.command == "download":

        print(
            json.dumps(
                download(),
                indent=2
            )
        )

    elif args.command == "audit":

        do_audit()

    else:

        do_train()


if __name__ == "__main__":
    main()
