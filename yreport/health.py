# yreport/health.py
import pandas as pd

from .diagnostics import (
    categorical_drift_readiness,
    datetime_diagnostics,
    missing_pattern_clusters,
    temporal_leakage_detection,
)
from .recommend import generate_recommendations, numeric_diagnostics
from .report import DataHealthReport
from .types import detect_column_types

# Weights for the three sub-scores that make up the final health score
WEIGHTS = {"missing": 0.5, "duplicates": 0.3, "cardinality": 0.2}


def data_health_report(
    df: pd.DataFrame,
    drop_cols=None,
    categorical_cols=None,
    numeric_cols=None,
    ignore_cols=None,
) -> DataHealthReport:
    """
    Analyse a DataFrame and return a DataHealthReport.

    Parameters
    ----------
    df              : Input DataFrame.
    drop_cols       : Columns the user explicitly wants dropped from recommendations.
    categorical_cols: Columns to force-treat as categorical (overrides auto-detection).
    numeric_cols    : Columns to force-treat as numeric (overrides auto-detection).
    ignore_cols     : Columns to exclude entirely from analysis.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Normalise all col-list arguments to sets
    drop_cols        = set(drop_cols or [])
    categorical_cols = set(categorical_cols or [])
    numeric_cols     = set(numeric_cols or [])
    ignore_cols      = set(ignore_cols or [])

    # Drop ignored columns from the working copy
    df = df.drop(columns=ignore_cols, errors="ignore")

    # --- Column type detection ---
    column_types = detect_column_types(df)
    column_types["numeric"]     = set(column_types["numeric"])
    column_types["categorical"] = set(column_types["categorical"])
    column_types["datetime"]    = set(column_types["datetime"])

    # Apply user overrides — categorical takes priority over auto-numeric
    if categorical_cols:
        column_types["categorical"].update(categorical_cols)
        column_types["numeric"].difference_update(categorical_cols)

    if numeric_cols:
        column_types["numeric"].update(numeric_cols)
        column_types["categorical"].difference_update(numeric_cols)

    # Remove any ignored columns that crept into type sets
    for key in column_types:
        column_types[key].difference_update(ignore_cols)

    # Convert sets back to sorted lists for stable output
    for key in column_types:
        column_types[key] = sorted(column_types[key])

    rows, cols = df.shape

    # --- Sub-scores (each in [0, 1]) ---

    # Missing: penalises the average fraction of missing values
    missing_ratio = df.isnull().mean().mean()
    missing_score = 1 - missing_ratio

    # Duplicates: penalises the fraction of duplicate rows
    duplicate_ratio = df.duplicated().mean()
    duplicate_score = 1 - duplicate_ratio

    # Existing recommendations & numeric diagnostics
    recommendations = generate_recommendations(df, drop_cols, column_types)
    numeric         = numeric_diagnostics(df, column_types["numeric"])

    # --- v0.1.4 Deep Diagnostics ---

    # #10 — Datetime column health (gaps, freq, timezone, future dates)
    dt_diagnostics = datetime_diagnostics(df, column_types["datetime"])

    # #12 — Categorical drift readiness (cardinality, rare cats, entropy)
    drift_readiness = categorical_drift_readiness(df, column_types["categorical"])

    # #13 — Missing pattern clustering (MCAR / MAR / MNAR inference)
    missing_patterns = missing_pattern_clusters(df)

    # #14 — Temporal leakage detection (future dates, index correlation, duplicates)
    leakage_report = temporal_leakage_detection(df, column_types["datetime"])

    # --- Cardinality sub-score ---
    # Use a separate variable so user-passed drop_cols is NOT overwritten
    auto_drop_cols = {
        col
        for col, info in recommendations["missing"].items()
        if info["action"] == "drop"
    }
    all_drop_cols = drop_cols | auto_drop_cols  # merged; original drop_cols preserved

    categorical_col_list = column_types["categorical"]
    high_card_cols = [
        col
        for col in categorical_col_list
        if df[col].nunique() > 50 and col not in all_drop_cols
    ]

    cardinality_ratio = len(high_card_cols) / max(len(categorical_col_list), 1)
    cardinality_score = 1 - cardinality_ratio

    # --- Final weighted score, clamped to [0, 100] ---
    raw_score = (
        missing_score  * WEIGHTS["missing"]
        + duplicate_score * WEIGHTS["duplicates"]
        + cardinality_score * WEIGHTS["cardinality"]
    ) * 100

    final_score = max(0.0, min(100.0, raw_score))  # safety clamp

    return DataHealthReport(
        health_score=round(final_score, 2),
        shape={"rows": rows, "columns": cols},
        column_types=column_types,
        missing_percentage=(df.isnull().mean() * 100).round(2).to_dict(),
        duplicate_rows=int(df.duplicated().sum()),
        warnings={
            "high_missing":    [
                col for col, pct in (df.isnull().mean() * 100).items() if pct > 30
            ],
            "high_cardinality": high_card_cols,
        },
        recommendations=recommendations,
        numeric=numeric,
        # v0.1.4 deep diagnostics
        datetime_diagnostics=dt_diagnostics,
        drift_readiness=drift_readiness,
        missing_patterns=missing_patterns,
        leakage_report=leakage_report,
    )