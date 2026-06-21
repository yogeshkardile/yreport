# yreport/recommend.py
import numpy as np
import pandas as pd
from scipy.stats import skew


def generate_recommendations(df: pd.DataFrame, drop_cols: set, column_types: dict) -> dict:
    """
    Produce actionable recommendations for missing-value handling
    and categorical encoding.

    Parameters
    ----------
    df           : The DataFrame being analysed.
    drop_cols    : User-defined columns to force-drop.
    column_types : Detected/overridden column type mapping.

    Returns
    -------
    dict with keys "encoding" and "missing", each mapping
    column names to recommendation dicts.
    """
    recommendations: dict = {"encoding": {}, "missing": {}}

    missing_pct = (df.isnull().mean() * 100).round(2).to_dict()

    # --- Missing value rules ---
    for col, pct in missing_pct.items():
        if pct > 60:
            recommendations["missing"][col] = {
                "action": "drop",
                "message": f"{pct}% missing values",
                "confidence": "HIGH",
            }
        elif pct > 5:
            recommendations["missing"][col] = {
                "action": "impute",
                "message": f"{pct}% missing values",
                "confidence": "MEDIUM",
            }

    # Merge auto-detected drops with user-defined drops
    # (do NOT reassign drop_cols — use a local merged set)
    auto_drop_cols = {
        col
        for col, info in recommendations["missing"].items()
        if info["action"] == "drop"
    }
    all_drop_cols = drop_cols | auto_drop_cols

    # Force user-defined drop columns into the missing recommendations
    for col in drop_cols:
        recommendations["missing"][col] = {
            "action": "drop",
            "message": "Dropped by user configuration",
            "confidence": "HIGH",
        }

    # --- Encoding recommendations (categorical columns only) ---
    for col in column_types["categorical"]:
        if col in all_drop_cols:
            continue  # skip columns that will be dropped

        high_card = df[col].nunique() > 50

        recommendations["encoding"][col] = {
            "action": "required",
            "message": (
                "Categorical encoding required (high cardinality)"
                if high_card
                else "Categorical encoding required"
            ),
            "confidence": "HIGH" if high_card else "MEDIUM",
        }

    return recommendations


def numeric_diagnostics(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Compute skewness and IQR-based outlier percentage for each
    numeric column, and suggest a transform where appropriate.

    Parameters
    ----------
    df           : The DataFrame being analysed.
    numeric_cols : List of column names to treat as numeric.

    Returns
    -------
    dict mapping column name → diagnostic info dict.
    """
    diagnostics: dict = {}

    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            continue

        # Guard: skip if the column is not actually numeric
        # (can happen when the user manually overrides column types)
        if not pd.api.types.is_numeric_dtype(series):
            continue

        col_skew = skew(series)

        # IQR-based outlier detection
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        outlier_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        outlier_pct = outlier_mask.mean() * 100

        diagnostics[col] = {
            "skewness": round(col_skew, 2),
            "outlier_percentage": round(outlier_pct, 2),
            "recommendation": (
                "consider log/robust transform"
                if abs(col_skew) > 1 or outlier_pct > 5
                else "no transform needed"
            ),
            "confidence": "HIGH" if abs(col_skew) > 1 else "MEDIUM",
        }

    return diagnostics
