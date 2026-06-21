# yreport/diagnostics.py
"""
Deep diagnostic functions for v0.1.4.

Covers four new analysis areas:
  - datetime_diagnostics         (#10) — gaps, frequency, timezone issues
  - categorical_drift_readiness  (#12) — flags columns likely to drift
  - missing_pattern_clusters     (#13) — groups columns by missing pattern (MCAR/MAR)
  - temporal_leakage_detection   (#14) — flags datetime columns risking train/test leakage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 10 — Datetime Column Diagnostics


def datetime_diagnostics(df: pd.DataFrame, datetime_cols: list) -> dict:
    """
    Analyse datetime columns for common data quality problems.

    Checks performed per column:
      - Null percentage
      - Inferred frequency (daily, monthly, irregular, …)
      - Gap detection: max consecutive gap vs. median gap
      - Timezone awareness
      - Future-date contamination (dates beyond today)
      - Monotonicity (is the column sorted?)

    Parameters
    ----------
    df            : Input DataFrame.
    datetime_cols : List of column names detected/overridden as datetime.

    Returns
    -------
    dict mapping column name → diagnostic info dict.
    """
    diagnostics: dict = {}

    for col in datetime_cols:
        if col not in df.columns:
            continue

        # Attempt coercion to datetime if not already
        series = pd.to_datetime(df[col], errors="coerce").dropna().sort_values()

        if series.empty:
            diagnostics[col] = {
                "null_percentage": 100.0,
                "issues": ["column is entirely null or unparseable"],
                "recommendation": "drop or investigate column",
                "confidence": "HIGH",
            }
            continue

        null_pct = round(df[col].isnull().mean() * 100, 2)
        issues: list[str] = []
        warnings: list[str] = []

        # --- Frequency inference ---
        inferred_freq = pd.infer_freq(series)
        freq_label = inferred_freq if inferred_freq else "irregular"

        # --- Gap analysis ---
        if len(series) > 1:
            deltas = series.diff().dropna()
            median_gap = deltas.median()
            max_gap = deltas.max()

            # A gap more than 5× the median is flagged as anomalous
            if max_gap > median_gap * 5:
                issues.append(
                    f"large gap detected: max gap {max_gap}, median gap {median_gap}"
                )
        else:
            median_gap = None
            max_gap = None

        # --- Timezone awareness ---
        tz_aware = series.dt.tz is not None
        if not tz_aware:
            warnings.append("no timezone info — may cause issues in multi-region data")

        # --- Future-date contamination ---
        now = pd.Timestamp.now(tz=series.dt.tz)
        future_count = int((series > now).sum())
        if future_count > 0:
            issues.append(
                f"{future_count} future dates detected — possible data leakage or entry error"
            )

        # --- Monotonicity ---
        is_monotonic = series.is_monotonic_increasing
        if not is_monotonic:
            warnings.append(
                "column is not sorted — consider sorting before time-series modelling"
            )

        all_issues = issues + warnings
        diagnostics[col] = {
            "null_percentage": null_pct,
            "inferred_frequency": freq_label,
            "max_gap": str(max_gap) if max_gap is not None else "N/A",
            "median_gap": str(median_gap) if median_gap is not None else "N/A",
            "timezone_aware": tz_aware,
            "future_dates": future_count,
            "is_monotonic": is_monotonic,
            "issues": all_issues if all_issues else ["no issues detected"],
            "recommendation": (
                "review issues before modelling" if issues else "column looks healthy"
            ),
            "confidence": "HIGH" if issues else "LOW",
        }

    return diagnostics


# 12 — Categorical Drift Readiness Checks


def categorical_drift_readiness(df: pd.DataFrame, categorical_cols: list) -> dict:
    """
    Flag categorical columns that are likely to drift in production.

    A column is considered drift-prone when it has:
      - High cardinality (> 50 unique values) — unseen categories at inference
      - Rare categories (any category with frequency < 1%) — fragile one-hot encodings
      - High entropy — uniform distribution with no dominant category
      - Single dominant category (> 95%) — near-constant, low signal

    Parameters
    ----------
    df               : Input DataFrame.
    categorical_cols : List of column names treated as categorical.

    Returns
    -------
    dict mapping column name → drift readiness info dict.
    """
    results: dict = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        n_unique = series.nunique()
        value_counts = series.value_counts(normalize=True)  # relative frequencies
        top_freq = float(value_counts.iloc[0])  # most common category share

        # Rare categories: share < 1%
        rare_cats = value_counts[value_counts < 0.01].index.tolist()

        # Shannon entropy (higher = more uniform distribution)
        probs = value_counts.values
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

        drift_risks: list[str] = []

        if n_unique > 50:
            drift_risks.append(
                f"high cardinality ({n_unique} unique values) — unseen categories likely at inference"
            )
        if rare_cats:
            drift_risks.append(
                f"{len(rare_cats)} rare categories (<1% frequency) — vulnerable to distribution shift"
            )
        if top_freq > 0.95:
            drift_risks.append(
                f"near-constant column ({top_freq:.1%} in one category) — low signal, may not generalise"
            )
        if entropy > 4.0 and n_unique <= 50:
            drift_risks.append(
                f"high entropy ({entropy:.2f}) — very uniform distribution, encoding may be unstable"
            )

        results[col] = {
            "n_unique": n_unique,
            "top_category_frequency": round(top_freq * 100, 2),
            "rare_category_count": len(rare_cats),
            "entropy": round(entropy, 3),
            "drift_risks": (
                drift_risks if drift_risks else ["no significant drift risks detected"]
            ),
            "recommendation": (
                "high drift risk — monitor in production"
                if drift_risks
                else "low drift risk"
            ),
            "confidence": (
                "HIGH"
                if len(drift_risks) >= 2
                else ("MEDIUM" if drift_risks else "LOW")
            ),
        }

    return results


# 13 — Missing Pattern Clustering


def missing_pattern_clusters(df: pd.DataFrame) -> dict:
    """
    Group columns by their missing-value pattern to hint at the
    missing-data mechanism (MCAR, MAR, or MNAR).

    Strategy:
      1. Build a binary missingness matrix (1 = missing, 0 = present).
      2. Identify columns that share identical missing patterns → cluster them.
      3. Infer likely mechanism based on cluster characteristics.

    Missing mechanism heuristics used:
      - MCAR (Missing Completely At Random): missingness is uncorrelated
        with any other column — indicated by a unique/isolated pattern.
      - MAR  (Missing At Random): multiple columns share the exact same
        missing rows → likely driven by a common upstream condition.
      - MNAR (Missing Not At Random): a column is nearly always missing
        when another specific column has a value → structural relationship.

    Parameters
    ----------
    df : Input DataFrame (all columns considered).

    Returns
    -------
    dict with keys:
      "clusters"         — list of column groups sharing a missing pattern
      "column_mechanism" — per-column MCAR/MAR/MNAR hint
      "summary"          — human-readable overview
    """
    # Columns that actually have missing values
    missing_cols = [col for col in df.columns if df[col].isnull().any()]

    if not missing_cols:
        return {
            "clusters": [],
            "column_mechanism": {},
            "summary": "No missing values — pattern analysis not applicable.",
        }

    # Binary missingness matrix: rows = observations, cols = features
    miss_matrix = df[missing_cols].isnull().astype(int)

    # Group columns by their identical missing pattern (as a tuple)
    pattern_map: dict[tuple, list[str]] = {}
    for col in missing_cols:
        pattern = tuple(miss_matrix[col].values)
        pattern_map.setdefault(pattern, []).append(col)

    clusters = list(pattern_map.values())

    # Per-column mechanism inference
    column_mechanism: dict[str, dict] = {}
    for pattern, cols in pattern_map.items():
        miss_rate = sum(pattern) / len(pattern)

        if len(cols) > 1:
            # Multiple columns share the exact same missing rows → MAR
            mechanism = "MAR"
            explanation = (
                f"{len(cols)} columns share identical missing rows — "
                "likely driven by a common condition (MAR)"
            )
        elif miss_rate > 0.6:
            # Extremely high missingness in isolation → MNAR suspected
            mechanism = "MNAR"
            explanation = (
                f"{miss_rate:.1%} missing — high isolated missingness "
                "suggests structural/self-selection reason (MNAR)"
            )
        else:
            # Isolated low-to-moderate missingness → assume MCAR
            mechanism = "MCAR"
            explanation = (
                "isolated missing pattern with moderate rate — "
                "consistent with random missingness (MCAR)"
            )

        for col in cols:
            column_mechanism[col] = {
                "mechanism": mechanism,
                "missing_rate": round(miss_rate * 100, 2),
                "cluster_size": len(cols),
                "cluster_members": cols,
                "explanation": explanation,
            }

    # Summary string
    mcar_cols = [c for c, v in column_mechanism.items() if v["mechanism"] == "MCAR"]
    mar_cols = [c for c, v in column_mechanism.items() if v["mechanism"] == "MAR"]
    mnar_cols = [c for c, v in column_mechanism.items() if v["mechanism"] == "MNAR"]

    summary_parts = []
    if mcar_cols:
        summary_parts.append(f"MCAR: {mcar_cols} — safe to impute or drop")
    if mar_cols:
        summary_parts.append(f"MAR: {mar_cols} — impute using related columns")
    if mnar_cols:
        summary_parts.append(
            f"MNAR: {mnar_cols} — consider indicator variable or domain fix"
        )

    return {
        "clusters": clusters,
        "column_mechanism": column_mechanism,
        "summary": "; ".join(summary_parts) if summary_parts else "No patterns found.",
    }


# 14 — Temporal Leakage Detection


def temporal_leakage_detection(df: pd.DataFrame, datetime_cols: list) -> dict:
    """
    Detect datetime columns that could cause train/test leakage.

    Leakage scenarios detected:
      1. Future-date contamination — values beyond today's date.
      2. Target-period overlap — column values overlap the likely
         prediction horizon (last 10% of the date range).
      3. High correlation with row index — dates that are almost
         perfectly aligned with row order suggest the column encodes
         sequence information that leaks ordering into features.
      4. Near-duplicate datetime columns — two columns with very
         similar distributions risk encoding the same temporal signal twice.

    Parameters
    ----------
    df            : Input DataFrame.
    datetime_cols : List of column names detected/overridden as datetime.

    Returns
    -------
    dict mapping column name → leakage risk info dict.
    """
    results: dict = {}
    parsed: dict[str, pd.Series] = {}

    # Pre-parse all datetime columns once
    for col in datetime_cols:
        if col not in df.columns:
            continue
        s = pd.to_datetime(df[col], errors="coerce").dropna()
        if not s.empty:
            parsed[col] = s

    for col, series in parsed.items():
        leakage_risks: list[str] = []

        now = pd.Timestamp.now(tz=series.dt.tz)
        future_count = int((series > now).sum())
        if future_count > 0:
            leakage_risks.append(
                f"{future_count} future-dated rows — feature contains post-prediction information"
            )

        # Target-period overlap: does the column touch the last 10% of its own range?
        date_min, date_max = series.min(), series.max()
        date_range = date_max - date_min
        if date_range.total_seconds() > 0:
            cutoff = date_max - date_range * 0.10
            overlap_count = int((series >= cutoff).sum())
            overlap_pct = round(overlap_count / len(series) * 100, 1)
            if overlap_pct > 5:
                leakage_risks.append(
                    f"{overlap_pct}% of values fall in the last 10% of the date range "
                    "— likely target-period overlap if used as a feature"
                )

        # Index correlation: convert datetime to ordinal, correlate with row index
        ordinal = series.reset_index(drop=True).map(pd.Timestamp.toordinal)
        index_series = pd.Series(range(len(ordinal)))
        if len(ordinal) > 2:
            corr = float(ordinal.corr(index_series))
            if abs(corr) > 0.95:
                leakage_risks.append(
                    f"high index correlation ({corr:.3f}) — column nearly encodes "
                    "row order, which leaks sequence information"
                )

        # Near-duplicate datetime columns
        for other_col, other_series in parsed.items():
            if other_col == col:
                continue
            # Align on index, compare ordinal representations
            shared_idx = series.index.intersection(other_series.index)
            if len(shared_idx) < 10:
                continue
            ord_a = series.loc[shared_idx].map(pd.Timestamp.toordinal)
            ord_b = other_series.loc[shared_idx].map(pd.Timestamp.toordinal)
            try:
                sim_corr = float(ord_a.corr(ord_b))
            except Exception:
                sim_corr = 0.0
            if sim_corr > 0.98:
                leakage_risks.append(
                    f"near-duplicate of '{other_col}' (correlation {sim_corr:.3f}) "
                    "— redundant temporal signal, consider dropping one"
                )

        results[col] = {
            "future_dates": future_count,
            "leakage_risks": (
                leakage_risks if leakage_risks else ["no leakage risks detected"]
            ),
            "recommendation": (
                "high leakage risk — review before train/test split"
                if leakage_risks
                else "no leakage detected"
            ),
            "confidence": (
                "HIGH"
                if len(leakage_risks) >= 2
                else ("MEDIUM" if leakage_risks else "LOW")
            ),
        }

    return results
