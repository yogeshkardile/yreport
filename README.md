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

yreport is a lightweight, pipeline-ready data validation and health diagnostics library for tabular ML datasets. It analyzes data quality, detects potential issues, and provides honest, actionable diagnostics without making unsafe assumptions.

Unlike heavy EDA tools, yreport is designed to be pipeline-friendly, explainable, configurable, and production-aware.

---

## Why yreport?

Most EDA libraries generate large HTML reports, make aggressive assumptions (e.g. one-hot everything), and are hard to integrate into ML pipelines.

**yreport focuses on decisions, not decoration.**

It helps answer:

- Is this dataset usable?
- Which columns are problematic?
- What should be fixed first?
- Where should I be careful before modeling?

---

## Features

- Weighted Data Health Score (0-100)
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
Rows: 891 | No_Columns: 12

Warnings:
- high_missing: ['Cabin']
- high_cardinality: ['Name', 'Ticket']
```

---

## What the Report Includes

### 1. Data Health Score

A weighted score based on:
- Missing values
- Duplicate rows
- High-cardinality features

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
- Detects high-cardinality features
- Does not assume one-hot or ordinal encoding

### 5. Numeric Diagnostics

For numeric columns:
- Skewness
- Outlier percentage (IQR method)
- Transform suggestions (log / robust)

---

## User Overrides

Automatic detection is never perfect. yreport allows explicit user control.

```python
data_health_report(
    df,
    ignore_cols=[...],
    drop_cols=[...],
    categorical_cols=[...],
    numerical_cols=[...]
)
```

| Override | Purpose |
| --- | --- |
| `ignore_cols` | Completely ignore columns |
| `drop_cols` | Force drop columns |
| `categorical_cols` | Force categorical treatment |
| `numerical_cols` | Force numeric treatment |

**Rules:**
- User intent always overrides automation
- A column belongs to only one semantic type
- Ignored or dropped columns are excluded everywhere

---

## Exporting Reports

**JSON export (machine-readable):**

```python
report.to_json("report.json")
# or
data = report.to_json()
```

**Markdown export (human-readable):**

```python
report.to_markdown("report.md")
```

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

pipe.named_steps["inspect"].report_.summary()
pipe.named_steps["inspect"].report_.to_markdown("train_report.md")
```

- Model trains normally
- Data remains unchanged
- Report is available after `fit()`

---

## Testing

Run tests from the project root:

```bash
pytest
```

Includes:
- sklearn pipeline compatibility test
- Core API regression protection

---

## Design Philosophy

| Principle | Approach |
| --- | --- |
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
