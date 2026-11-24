import pandas as pd
import numpy as np

LEAK = {
    "resolved_at", "closed_at", "resolution_time", "close_code",
    "close_notes", "resolved_by", "closed_by", "resolution_description",
    "sys_updated_at", "sys_updated_on", "sys_mod_count",
    "reassignment_count", "resolved", "incident_state", "state",
    "made_sla", "sla", "sla_met", "breached_sla",
    "actual_resolution_time", "elapsed_time", "duration",
    "time_to_resolve"
}

IDS = {
    "sys_id", "number", "incident_id", "id",
    "caller_id", "assigned_to", "opened_by"
}


def norm_target(s):
    mapping = {
        "true": 1, "1": 1, "yes": 1, "y": 1,
        "false": 0, "0": 0, "no": 0, "n": 0,
        "met": 1, "breached": 0, "not met": 0,
        "breach": 0
    }

    values = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    numeric_values = pd.to_numeric(s, errors="coerce")
    values = values.fillna(numeric_values)

    if values.isna().any():
        raise ValueError(
            "Target contains missing or unrecognized values."
        )

    if not set(values.unique()).issubset({0, 1}):
        raise ValueError("Target is not binary/recognizable.")

    return values.astype(int)


def find(df, names):
    column_map = {str(c).lower(): c for c in df.columns}

    for name in names:
        if name.lower() in column_map:
            return column_map[name.lower()]

    return None


def group_incidents(df):

    target = find(
        df,
        ["made_sla", "sla_met", "sla", "breached_sla"]
    )

    ident = find(
        df,
        ["number", "incident_id", "sys_id"]
    )

    opened = find(
        df,
        ["opened_at", "opened", "created_at", "sys_created_on"]
    )

    mod_count = find(df, ["sys_mod_count"])
    updated = find(df, ["sys_updated_on", "sys_updated_at"])

    if target is None:
        raise ValueError("Target column not found.")

    if ident is None:
        raise ValueError("Incident identifier not found.")

    d = df.copy()

    d["_original_target"] = norm_target(d[target])

    if opened is not None:
        d[opened] = pd.to_datetime(
            d[opened],
            errors="coerce",
            utc=True,
            dayfirst=True
        )

    if updated is not None:
        d[updated] = pd.to_datetime(
            d[updated],
            errors="coerce",
            utc=True,
            dayfirst=True
        )

    d["_original_row_order"] = np.arange(len(d))

    if mod_count is not None:
        d["_event_order"] = pd.to_numeric(
            d[mod_count],
            errors="coerce"
        )

    elif updated is not None:
        d["_event_order"] = d[updated]

    else:
        d["_event_order"] = d["_original_row_order"]

    rows = []

    for key, group in d.groupby(
        ident,
        dropna=False,
        sort=False
    ):

        group = group.sort_values(
            ["_event_order", "_original_row_order"],
            kind="stable",
            na_position="last"
        )

        first_row = group.iloc[0].copy()

        last_target = int(
            group.iloc[-1]["_original_target"]
        )

        first_row["_incident_key"] = str(key)

        # 1 = breach, 0 = SLA met
        first_row["_y"] = int(1 - last_target)

        rows.append(first_row)

    if not rows:
        raise ValueError("No incidents were generated.")

    incidents = pd.DataFrame(rows).reset_index(drop=True)

    incidents.drop(
        columns=[
            "_original_target",
            "_original_row_order",
            "_event_order"
        ],
        inplace=True,
        errors="ignore"
    )

    return incidents, target, ident, opened


def make_X(inc, target, ident, opened):

    y = inc["_y"].astype(int)

    allowed = {
        "contact_type",
        "location",
        "category",
        "subcategory",
        "u_symptom",
        "impact",
        "urgency",
        "priority",
        "assignment_group",
        "knowledge"
    }

    cols = []

    for column in inc.columns:

        lower_column = column.lower()

        if column in {
            "_y",
            "_incident_key",
            target,
            ident,
            opened
        }:
            continue

        if lower_column in LEAK:
            continue

        if lower_column in IDS:
            continue

        if lower_column in allowed:
            cols.append(column)

    X = inc[cols].copy()

    # Normalize placeholder values.
    placeholders = [
        "?",
        "",
        "none",
        "null",
        "nan",
        "unknown"
    ]

    for column in X.columns:

        if X[column].dtype == "object":

            X[column] = (
                X[column]
                .astype(str)
                .str.strip()
            )

            X[column] = X[column].replace(
                placeholders,
                np.nan
            )

    # Temporal features
    if opened is not None and opened in inc.columns:

        dt = pd.to_datetime(
            inc[opened],
            errors="coerce",
            utc=True,
            dayfirst=True
        )

        X["opened_month"] = (
            dt.dt.month.astype("Int64").astype(str)
        )

        X["opened_weekday"] = (
            dt.dt.dayofweek.astype("Int64").astype(str)
        )

        X["opened_hour"] = (
            dt.dt.hour.astype("Int64").astype(str)
        )

    if X.empty:
        raise ValueError("No eligible features were found.")

    return X, y, cols