# yreport

<p align="center">
  <a href="https://pypi.org/project/yreport/">
    <img src="https://img.shields.io/pypi/v/yreport.svg">
  </a>
  <a href="https://pepy.tech/projects/yreport">
    <img src="https://static.pepy.tech/personalized-badge/yreport?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads">
  </a>
  <img src="https://img.shields.io/github/license/yogeshkardile/yreport">
  <a href="https://github.com/yogeshkardile/yreport/actions/workflows/ci.yml">
    <img src="https://github.com/yogeshkardile/yreport/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  </a>
  <a href="https://github.com/yogeshkardile/yreport">
    <img src="https://img.shields.io/badge/coverage-94%25-brightgreen.svg" alt="Coverage 94%">
  </a>
</p>

---

yreport is a lightweight, pipeline-ready data validation and deep diagnostics library for tabular ML datasets. It analyses data quality, detects potential issues, and provides honest, actionable diagnostics without making unsafe assumptions.

Unlike heavy EDA tools, yreport is designed to be pipeline-friendly, explainable, configurable, and production-aware.

---

## Why yreport?

Most EDA libraries generate large HTML reports, make aggressive assumptions (e.g. one-hot everything), and are hard to integrate into ML pipelines.

**yreport focuses on decisions, not decoration.**

It helps answer:

- Is this dataset usable?
- Which columns are problematic?
- What should be fixed first?
- Where should I be careful before modelling?
- Are my datetime columns healthy and leakage-free?
- Which categorical columns will drift in production?

---

## Features

- Weighted Data Health Score (0–100)
- Automatic column type detection
- Missing value diagnostics with confidence levels
- High-cardinality categorical detection
- Numeric skewness and outlier analysis
- Honest categorical handling (no forced one-hot / ordinal)
- User override support
- Non-contradictory recommendations
- JSON and Markdown export
- scikit-learn Pipeline integration
- Lightweight and fast
- **v0.1.4 — Deep Diagnostics:**
  - Datetime column diagnostics (gaps, frequency, timezone, future dates)
  - Categorical drift readiness checks
  - Missing pattern clustering (MCAR / MAR / MNAR inference)
  - Temporal leakage detection

---

## Installation

### Install from PyPI

```bash
pip install yreport
```

### Install from source (recommended)

```bash
git clone https://github.com/yogeshkardile/yreport.git
cd yreport
pip install -e .
```

---

## Core Concept

yreport does not modify your data.

It:
- Inspects datasets
- Reports potential issues
- Suggests actions with confidence

It does not:
- Apply transformations
- Guess encoding methods
- Perform feature engineering

This makes it safe and transparent.

---

## Quick Start

```python
import pandas as pd
from yreport import data_health_report

df = pd.read_csv("data.csv")

report = data_health_report(df)
report.summary()
```

**Example console output:**

```
Data Health Score: 87.95/100
Rows: 891 | Columns: 12
Duplicate Rows: 0

Numeric Columns    : ['Age', 'Fare', 'SibSp']
Categorical Columns: ['Embarked', 'Sex']
DateTime Columns   : ['booking_date']

Missing Percentage:
  - Cabin: 77.1%
  - Age: 19.87%

Warnings:
  - high_missing: ['Cabin']
  - high_cardinality: ['Name', 'Ticket']

Datetime Diagnostics:
  - booking_date: freq=D, issues=['no issues detected']

Categorical Drift Readiness:
  - Embarked: low drift risk (confidence=LOW)

Missing Pattern Clusters:
  MCAR: ['Age'] — safe to impute or drop; MNAR: ['Cabin'] — consider indicator variable

Temporal Leakage Detection:
  - booking_date: no leakage detected (confidence=LOW)
```

---

## What the Report Includes

### 1. Data Health Score

A weighted score (0–100) based on:
- Missing values (weight: 0.5)
- Duplicate rows (weight: 0.3)
- High-cardinality features (weight: 0.2)

### 2. Column Type Detection

Automatically detects:
- Numeric columns
- Categorical columns
- Datetime columns

### 3. Missing Value Diagnostics

- Missing percentage per column
- Drop or impute recommendations
- Confidence levels: `HIGH` / `MEDIUM`

### 4. Categorical Diagnostics

- Flags categorical columns that require encoding
- Detects high-cardinality features (> 50 unique values)
- Does not assume one-hot or ordinal encoding

### 5. Numeric Diagnostics

For numeric columns:
- Skewness
- Outlier percentage (IQR method)
- Transform suggestions (log / robust)

### 6. Datetime Diagnostics *(v0.1.4)*

For each datetime column:
- Inferred frequency (daily, monthly, irregular, …)
- Gap detection — flags gaps more than 5× the median gap
- Timezone awareness check
- Future-date contamination count
- Monotonicity check

### 7. Categorical Drift Readiness *(v0.1.4)*

Flags columns likely to behave differently at inference:
- High cardinality (> 50 unique values) — unseen categories at inference time
- Rare categories (< 1% frequency) — vulnerable to distribution shift
- Near-constant columns (> 95% one category) — low signal
- High-entropy distributions — encoding instability risk

### 8. Missing Pattern Clustering *(v0.1.4)*

Groups columns by their shared missing-value pattern and infers the likely mechanism:

| Mechanism | Meaning | Suggested Action |
|-----------|---------|-----------------|
| MCAR | Missing completely at random — isolated pattern | Safe to impute or drop |
| MAR  | Missing at random — shared pattern with other columns | Impute using related columns |
| MNAR | Missing not at random — structural reason suspected | Add indicator variable or apply domain fix |

### 9. Temporal Leakage Detection *(v0.1.4)*

Scans datetime columns for train/test leakage risks:
- Future-dated rows (post-prediction information in features)
- Target-period overlap (values in the last 10% of the date range)
- High index correlation (> 0.95) — column encodes row ordering
- Near-duplicate datetime columns (correlation > 0.98)

---

## User Overrides

Automatic detection is never perfect. yreport allows explicit user control.

```python
data_health_report(
    df,
    ignore_cols=[...],
    drop_cols=[...],
    categorical_cols=[...],
    numeric_cols=[...]
)
```

| Override | Purpose |
|----------|---------|
| `ignore_cols` | Completely ignore columns from all analysis |
| `drop_cols` | Force drop columns in recommendations |
| `categorical_cols` | Force categorical treatment |
| `numeric_cols` | Force numeric treatment |

**Rules:**
- User intent always overrides automation
- A column belongs to only one semantic type
- Ignored or dropped columns are excluded everywhere

---

## Exporting Reports

**JSON export (machine-readable):**

```python
report.to_json("report.json")
# or get the dict directly
data = report.to_json()
```

The JSON output includes all deep diagnostic fields:

```json
{
  "health_score": 87.95,
  "shape": {"rows": 891, "columns": 12},
  "numeric_diagnostics": { ... },
  "datetime_diagnostics": { ... },
  "drift_readiness": { ... },
  "missing_patterns": { ... },
  "leakage_report": { ... }
}
```

**Markdown export (human-readable):**

```python
report.to_markdown("report.md")
```

The Markdown report includes all sections including deep diagnostics, with a formatted table for missing pattern clusters.

---

## scikit-learn Pipeline Integration

yreport provides a no-op sklearn inspector that lets you observe data during training without interfering with the model or the pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from yreport import YReportInspector

pipe = Pipeline([
    ("inspect", YReportInspector(
        categorical_cols=["Pclass"],
        ignore_cols=["Name"]
    )),
    ("model", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)

# Access full report after fit
pipe.named_steps["inspect"].report_.summary()
pipe.named_steps["inspect"].report_.to_markdown("train_report.md")

# Access deep diagnostics directly
drift = pipe.named_steps["inspect"].report_.drift_readiness
leakage = pipe.named_steps["inspect"].report_.leakage_report
```

- Model trains normally
- Data remains unchanged
- Full report (including deep diagnostics) is available after `fit()`

---

## Report Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `health_score` | float | Weighted score 0–100 |
| `shape` | dict | `{rows, columns}` |
| `column_types` | dict | `{numeric, categorical, datetime}` lists |
| `missing_percentage` | dict | Per-column missing % |
| `duplicate_rows` | int | Count of duplicate rows |
| `warnings` | dict | `high_missing`, `high_cardinality` lists |
| `recommendations` | dict | `encoding` and `missing` action dicts |
| `numeric_diagnostics` | dict | Skewness, outliers, transform advice |
| `datetime_diagnostics` | dict | Frequency, gaps, timezone, future dates *(v0.1.4)* |
| `drift_readiness` | dict | Per-column drift risk flags *(v0.1.4)* |
| `missing_patterns` | dict | MCAR/MAR/MNAR cluster analysis *(v0.1.4)* |
| `leakage_report` | dict | Temporal leakage risks per datetime column *(v0.1.4)* |

---

## Testing

Run tests from the project root:

```bash
pytest
```

Includes:
- sklearn pipeline compatibility test
- Core API regression protection
- Deep diagnostics coverage

---

## Design Philosophy

| Principle | Approach |
|-----------|---------|
| Correctness > Automation | Reports issues; does not silently fix them |
| Transparency > Guessing | All recommendations include confidence levels |
| Diagnostics > Decoration | Output is structured and machine-readable |
| User intent > Heuristics | Overrides always take precedence |

yreport will never silently apply transformations.

---

## What yreport is NOT

- An AutoML tool
- A feature engineering pipeline
- A visualization-heavy EDA library
- An encoding decision engine

This is intentional. yreport is a diagnostics layer, not a transformation layer.

---

## License

This project is licensed under the terms specified in the [LICENSE](https://github.com/yogeshkardile/yreport/blob/main/LICENSE) file.