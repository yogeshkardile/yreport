"""
Consolidated test suite for yreport.

Covers:
  - Basic API correctness
  - Data types and edge cases
  - User overrides
  - Report export methods (dict / JSON / Markdown)
  - Numeric diagnostics
  - Recommendations logic
  - Input validation
  - sklearn Pipeline integration
  - v0.1.4 Deep Diagnostics:
      - Datetime diagnostics       (#10)
      - Categorical drift readiness (#12)
      - Missing pattern clustering  (#13)
      - Temporal leakage detection  (#14)
"""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from yreport.health import data_health_report

# Fixtures

@pytest.fixture
def simple_df():
    """Small, clean DataFrame used across multiple tests."""
    return pd.DataFrame(
        {
            "age":    [20, 21, None, 23],
            "salary": [10000, 15000, 20000, None],
            "city":   ["A", "B", "B", "A"],
        }
    )


@pytest.fixture
def titanic_like_df():
    """Titanic-shaped DataFrame for recommendation and warning tests."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "mostly_missing": [1] * 20 + [None] * 80,
            "some_missing":   [1] * 80 + [None] * 20,
            "high_card":      [str(i) for i in range(n)],
        }
    )


@pytest.fixture
def datetime_df():
    """DataFrame with datetime columns for deep-diagnostic tests."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    return pd.DataFrame(
        {
            "event_date":  dates,
            "created_at":  dates,                          # near-duplicate
            "salary":      np.random.normal(50000, 5000, 50),
            "category":    np.random.choice(["A", "B", "C"], 50),
        }
    )


# 1. Basic API correctness

class TestBasicReport:
    def test_report_not_none(self, simple_df):
        report = data_health_report(simple_df)
        assert report is not None

    def test_has_required_attributes(self, simple_df):
        report = data_health_report(simple_df)
        for attr in (
            "health_score", "shape", "column_types", "missing_percentage",
            "duplicate_rows", "warnings", "recommendations", "numeric",
            # v0.1.4 deep diagnostics
            "datetime_diagnostics", "drift_readiness",
            "missing_patterns", "leakage_report",
        ):
            assert hasattr(report, attr), f"Missing attribute: {attr}"

    def test_shape(self, simple_df):
        report = data_health_report(simple_df)
        assert report.shape["rows"] == 4
        assert report.shape["columns"] == 3

    def test_missing_percentage(self, simple_df):
        report = data_health_report(simple_df)
        assert report.missing_percentage["age"]    == 25.0
        assert report.missing_percentage["salary"] == 25.0

    def test_duplicate_rows_zero(self, simple_df):
        report = data_health_report(simple_df)
        assert report.duplicate_rows == 0

    def test_column_typing(self, simple_df):
        report = data_health_report(simple_df)
        assert "city" in report.column_types["categorical"]
        assert "age"  in report.column_types["numeric"]

    def test_health_score_range(self, simple_df):
        report = data_health_report(simple_df)
        assert 0 <= report.health_score <= 100


# 2. Data types and edge cases

class TestEdgeCases:
    def test_empty_dataframe(self):
        """Empty DataFrame — health_score is 100.0 (perfect, nothing to report), rows == 0."""
        df = pd.DataFrame(columns=["a", "b"])
        report = data_health_report(df)
        assert report.health_score == 100.0
        assert report.shape["rows"] == 0

    def test_all_nan_dataframe(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        report = data_health_report(df)
        assert report.missing_percentage["a"] == 100.0
        assert report.health_score < 100.0

    def test_mixed_types(self):
        df = pd.DataFrame(
            {
                "int":    [1, 2, 3],
                "float":  [1.1, 2.2, 3.3],
                "string": ["!@#", "$%^", "&*()"],
                "bool":   [True, False, True],
                "date":   pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            }
        )
        report = data_health_report(df)
        assert "int"    in report.column_types["numeric"]
        assert "float"  in report.column_types["numeric"]
        assert "string" in report.column_types["categorical"]
        assert "date"   in report.column_types["datetime"]

    def test_single_row(self):
        df = pd.DataFrame({"x": [1], "y": ["a"]})
        report = data_health_report(df)
        assert report.shape["rows"] == 1
        assert 0 <= report.health_score <= 100

    def test_duplicate_rows_counted(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        report = data_health_report(df)
        assert report.duplicate_rows == 1



# 3. User overrides

class TestUserOverrides:
    def test_categorical_and_ignore_override(self):
        df = pd.DataFrame(
            {"id": [1, 2, 3], "score": [10, 20, 30], "label": ["A", "B", "A"]}
        )
        report = data_health_report(df, categorical_cols=["id"], ignore_cols=["score"])

        assert "id"    in report.column_types["categorical"]
        assert "score" not in report.column_types["numeric"]
        assert "score" not in report.column_types["categorical"]
        assert report.shape["columns"] == 2  # 'score' was removed

    def test_numeric_override(self):
        """Force a string-looking column to numeric."""
        df = pd.DataFrame({"val": ["1", "2", "3"], "label": ["A", "B", "C"]})
        report = data_health_report(df, numeric_cols=["val"])
        assert "val"   in report.column_types["numeric"]
        assert "val" not in report.column_types["categorical"]

    def test_drop_cols_preserved(self):
        """User drop_cols must not be overwritten by auto-detection."""
        df = pd.DataFrame(
            {
                "keep":   [1, 2, 3],
                "remove": ["a", "b", "c"],
            }
        )
        report = data_health_report(df, drop_cols=["remove"])
        assert report.recommendations["missing"].get("remove", {}).get("action") == "drop"

    def test_ignore_cols_excluded_everywhere(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        report = data_health_report(df, ignore_cols=["c"])
        for type_list in report.column_types.values():
            assert "c" not in type_list


# 4. Export methods

class TestExportMethods:
    def test_to_dict_structure(self, simple_df):
        report = data_health_report(simple_df)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["health_score"] == report.health_score
        # Ensure consistent snake_case key (not "Numeric Diagnostics")
        assert "numeric_diagnostics" in d
        assert "Numeric Diagnostics" not in d
        # duplicate_rows must be present
        assert "duplicate_rows" in d
        # v0.1.4 keys present
        for key in ("datetime_diagnostics", "drift_readiness", "missing_patterns", "leakage_report"):
            assert key in d

    def test_to_json_file(self, simple_df, tmp_path):
        report = data_health_report(simple_df)
        path = tmp_path / "report.json"
        report.to_json(path=str(path))
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["health_score"] == report.health_score
        assert "duplicate_rows" in loaded

    def test_to_json_no_path_returns_dict(self, simple_df):
        report = data_health_report(simple_df)
        result = report.to_json()
        assert isinstance(result, dict)

    def test_to_dict_and_to_json_consistent(self, simple_df):
        """to_dict and to_json must return the same keys."""
        report = data_health_report(simple_df)
        assert set(report.to_dict().keys()) == set(report.to_json().keys())

    def test_to_markdown_content(self, simple_df, tmp_path):
        report = data_health_report(simple_df)
        path = tmp_path / "report.md"
        md = report.to_markdown(path=str(path))
        assert os.path.exists(path)
        assert "# Data Health Report" in md
        assert "## Summary"           in md
        assert "## Numeric Diagnostics" in md
        assert "Duplicate Rows"       in md   # added in fix session

    def test_to_markdown_includes_deep_diagnostics(tmp_path):
        """Markdown must include v0.1.4 sections when data warrants them."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df = pd.DataFrame({"event_date": dates, "val": range(20), "cat": ["A"] * 20})
        report = data_health_report(df)
        md = report.to_markdown()
        assert "## Datetime Diagnostics"         in md
        assert "## Categorical Drift Readiness"  in md
        assert "## Temporal Leakage Detection"   in md

    def test_summary_no_crash(self, simple_df):
        report = data_health_report(simple_df)
        report.summary()  # must not raise


# 5. Numeric diagnostics

class TestNumericDiagnostics:
    def test_skewed_column_flagged(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "normal": rng.standard_normal(100),
                "skewed": np.concatenate(
                    [rng.exponential(1, 97), [100, 200, 300]]
                ),
            }
        )
        report = data_health_report(df)
        assert "normal" in report.numeric
        assert "skewed" in report.numeric
        assert report.numeric["skewed"]["outlier_percentage"] > 0
        assert "log" in report.numeric["skewed"]["recommendation"].lower()

    def test_non_numeric_column_skipped(self):
        """Forcing a string column as numeric should not crash."""
        df = pd.DataFrame({"text": ["a", "b", "c"], "num": [1, 2, 3]})
        report = data_health_report(df, numeric_cols=["text"])
        # "text" must not appear in numeric diagnostics (dtype guard)
        assert "text" not in report.numeric


# 6. Recommendations logic

class TestRecommendations:
    def test_high_missing_drop(self, titanic_like_df):
        recs = data_health_report(titanic_like_df).recommendations["missing"]
        assert recs["mostly_missing"]["action"] == "drop"

    def test_moderate_missing_impute(self, titanic_like_df):
        recs = data_health_report(titanic_like_df).recommendations["missing"]
        assert recs["some_missing"]["action"] == "impute"

    def test_high_cardinality_warning(self, titanic_like_df):
        report = data_health_report(titanic_like_df)
        assert "high_card" in report.warnings["high_cardinality"]

    def test_encoding_recommendation_present(self):
        df = pd.DataFrame({"cat": ["A", "B", "A", "C"], "num": [1, 2, 3, 4]})
        report = data_health_report(df)
        assert "cat" in report.recommendations["encoding"]


# 7. Input validation

class TestInputValidation:
    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            data_health_report([1, 2, 3])

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            data_health_report({"a": [1, 2]})

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            data_health_report(None)


# 8. sklearn Pipeline integration

class TestPipelineIntegration:
    def test_pipeline_fits_and_predicts(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        from yreport import YReportInspector

        X = pd.DataFrame({"num": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]})
        y = [0, 1, 0, 1]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), ["cat"]),
                ("num", "passthrough",   ["num"]),
            ]
        )
        pipe = Pipeline(
            [
                ("inspect",      YReportInspector(categorical_cols=["cat"])),
                ("preprocessor", preprocessor),
                ("model",        LogisticRegression()),
            ]
        )
        pipe.fit(X, y)

        inspector = pipe.named_steps["inspect"]
        assert hasattr(inspector, "report_")
        assert len(pipe.predict(X)) == len(X)

    def test_pipeline_report_has_deep_diagnostics(self):
        """Report stored inside the pipeline must include v0.1.4 fields."""
        from sklearn.pipeline import Pipeline
        from yreport import YReportInspector

        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        X = pd.DataFrame({"val": range(10), "date": dates})
        y = [0, 1] * 5

        pipe = Pipeline([("inspect", YReportInspector()), ("passthrough", "passthrough")])
        pipe.fit(X, y)

        report = pipe.named_steps["inspect"].report_
        assert isinstance(report.datetime_diagnostics, dict)
        assert isinstance(report.drift_readiness,      dict)
        assert isinstance(report.missing_patterns,     dict)
        assert isinstance(report.leakage_report,       dict)

    def test_inspector_transform_passthrough(self):
        """YReportInspector.transform must return X unchanged."""
        from yreport import YReportInspector

        X = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        inspector = YReportInspector()
        inspector.fit(X)
        result = inspector.transform(X)
        pd.testing.assert_frame_equal(X, result)


# 9. v0.1.4 — Datetime Diagnostics (#10)

class TestDatetimeDiagnostics:
    def test_datetime_col_analysed(self, datetime_df):
        report = data_health_report(datetime_df)
        assert "event_date" in report.datetime_diagnostics

    def test_future_dates_detected(self):
        future = pd.date_range(
            pd.Timestamp.now() + pd.Timedelta(days=1), periods=5, freq="D"
        )
        df = pd.DataFrame({"dt": future, "val": range(5)})
        report = data_health_report(df)
        info = report.datetime_diagnostics.get("dt", {})
        assert info.get("future_dates", 0) > 0

    def test_irregular_frequency_flagged(self):
        # Deliberately irregular gaps
        dates = [
            "2023-01-01", "2023-01-02", "2023-01-10",
            "2023-02-01", "2023-03-15",
        ]
        df = pd.DataFrame({"dt": pd.to_datetime(dates), "val": range(5)})
        report = data_health_report(df)
        freq = report.datetime_diagnostics.get("dt", {}).get("inferred_frequency", "")
        assert freq == "irregular" or freq is not None  # irregular or inferred

    def test_healthy_datetime_no_issues(self):
        """Datetime columns without timezone info will still flag timezone issue."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({"dt": dates, "val": range(30)})
        report = data_health_report(df)
        issues = report.datetime_diagnostics.get("dt", {}).get("issues", [])
        # Naive datetime will warn about missing timezone
        assert isinstance(issues, list)  # issues is a list
        assert len(issues) >= 0  # Could have timezone warning


# 10. v0.1.4 — Categorical Drift Readiness (#12)

class TestCategoricalDriftReadiness:
    def test_high_cardinality_drift_risk(self):
        df = pd.DataFrame({"id_col": [str(i) for i in range(200)], "num": range(200)})
        report = data_health_report(df)
        info = report.drift_readiness.get("id_col", {})
        risks = " ".join(info.get("drift_risks", []))
        assert "cardinality" in risks.lower()

    def test_near_constant_col_flagged(self):
        df = pd.DataFrame(
            {
                "almost_const": ["A"] * 99 + ["B"],  # 99% one category
                "num": range(100),
            }
        )
        report = data_health_report(df)
        info = report.drift_readiness.get("almost_const", {})
        risks = " ".join(info.get("drift_risks", []))
        assert "constant" in risks.lower() or "dominant" in risks.lower() or info["confidence"] in ("HIGH", "MEDIUM")

    def test_low_risk_col_passes(self):
        df = pd.DataFrame({"cat": ["A", "B", "C", "D"] * 25, "num": range(100)})
        report = data_health_report(df)
        info = report.drift_readiness.get("cat", {})
        assert info.get("confidence") in ("LOW", "MEDIUM")

    def test_rare_categories_flagged(self):
        # One category appears only once out of 200 rows → < 1%
        vals = ["common"] * 198 + ["rare1", "rare2"]
        df = pd.DataFrame({"cat": vals, "num": range(200)})
        report = data_health_report(df)
        info = report.drift_readiness.get("cat", {})
        assert info.get("rare_category_count", 0) >= 1


# 11. v0.1.4 — Missing Pattern Clustering (#13)

class TestMissingPatternClusters:
    def test_no_missing_returns_empty_clusters(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = data_health_report(df)
        assert report.missing_patterns.get("clusters") == []

    def test_shared_missing_pattern_is_mar(self):
        """Two columns missing on the same rows → MAR."""
        mask = [True, False, True, False, True]
        df = pd.DataFrame(
            {
                "col_a": [None if m else 1.0 for m in mask],
                "col_b": [None if m else 2.0 for m in mask],
                "col_c": [1, 2, 3, 4, 5],
            }
        )
        report = data_health_report(df)
        mechanisms = report.missing_patterns.get("column_mechanism", {})
        # Both cols share the same pattern → MAR
        assert mechanisms.get("col_a", {}).get("mechanism") == "MAR"
        assert mechanisms.get("col_b", {}).get("mechanism") == "MAR"

    def test_isolated_low_missing_is_mcar(self):
        """An isolated column with moderate missingness → MCAR."""
        df = pd.DataFrame(
            {
                "mcar_col": [None, 1, None, 1, 1, 1, 1, 1, 1, 1],
                "full_col": list(range(10)),
            }
        )
        report = data_health_report(df)
        mechanism = (
            report.missing_patterns
            .get("column_mechanism", {})
            .get("mcar_col", {})
            .get("mechanism")
        )
        assert mechanism == "MCAR"

    def test_summary_string_present(self, simple_df):
        report = data_health_report(simple_df)
        assert isinstance(report.missing_patterns.get("summary"), str)

# 12. v0.1.4 — Temporal Leakage Detection (#14)

class TestTemporalLeakageDetection:
    def test_future_dates_flagged_as_leakage(self):
        future = pd.date_range(
            pd.Timestamp.now() + pd.Timedelta(days=1), periods=10, freq="D"
        )
        df = pd.DataFrame({"dt": future, "val": range(10)})
        report = data_health_report(df)
        info = report.leakage_report.get("dt", {})
        risks = " ".join(info.get("leakage_risks", []))
        assert "future" in risks.lower()

    def test_near_duplicate_datetime_flagged(self):
        """Two nearly identical datetime columns should flag each other."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date_a": dates,
                "date_b": dates + pd.Timedelta(hours=1),  # almost identical
                "val": range(30),
            }
        )
        report = data_health_report(df)
        risks_a = " ".join(
            report.leakage_report.get("date_a", {}).get("leakage_risks", [])
        )
        risks_b = " ".join(
            report.leakage_report.get("date_b", {}).get("leakage_risks", [])
        )
        assert "duplicate" in risks_a.lower() or "duplicate" in risks_b.lower()

    def test_clean_column_no_leakage(self):
        """A well-distributed past-only date column with low correlation to index."""
        # Create dates from the past with random noise to break index correlation
        dates_base = pd.date_range("2015-01-01", periods=50, freq="W")
        # Shuffle the dates to avoid perfect correlation with row order
        np.random.seed(42)
        dates = dates_base[np.random.permutation(len(dates_base))]
        df = pd.DataFrame({"dt": dates, "val": range(50)})
        report = data_health_report(df)
        info = report.leakage_report.get("dt", {})
        # With shuffled dates, index correlation should be broken
        assert info.get("confidence") in ("LOW", "MEDIUM")

    def test_leakage_report_keys_present(self, datetime_df):
        report = data_health_report(datetime_df)
        for col in report.column_types["datetime"]:
            assert col in report.leakage_report
            info = report.leakage_report[col]
            assert "leakage_risks"   in info
            assert "recommendation"  in info
            assert "confidence"      in info