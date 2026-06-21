# yreport/report.py


class DataHealthReport:
    """
    Stores the results of a data health analysis and provides
    multiple output formats: summary print, dict, JSON, and Markdown.

    v0.1.4 adds four deep-diagnostic fields:
      datetime_diagnostics — per-column datetime health (#10)
      drift_readiness      — categorical drift risk flags (#12)
      missing_patterns     — MCAR/MAR/MNAR clustering (#13)
      leakage_report       — temporal leakage risks (#14)
    """

    def __init__(
        self,
        health_score: int,
        shape: dict,
        column_types: dict,
        missing_percentage: dict,
        duplicate_rows: int,
        warnings: dict,
        recommendations: dict,
        numeric: dict,
        # v0.1.4 deep diagnostics
        datetime_diagnostics: dict = None,
        drift_readiness: dict = None,
        missing_patterns: dict = None,
        leakage_report: dict = None,
    ):
        self.health_score = health_score
        self.shape = shape
        self.column_types = column_types
        self.missing_percentage = missing_percentage
        self.duplicate_rows = duplicate_rows
        self.warnings = warnings
        self.recommendations = recommendations
        self.numeric = numeric
        # Deep diagnostics (default to empty dicts if not provided)
        self.datetime_diagnostics = datetime_diagnostics or {}
        self.drift_readiness = drift_readiness or {}
        self.missing_patterns = missing_patterns or {}
        self.leakage_report = leakage_report or {}

    # summary()

    def summary(self) -> None:
        """Print a human-readable summary of the full data health report."""
        print(f"Data Health Score: {self.health_score}/100")
        print(f"Rows: {self.shape['rows']} | Columns: {self.shape['columns']}")
        print(f"Duplicate Rows: {self.duplicate_rows}")
        print()
        print(f"Numeric Columns    : {self.column_types['numeric']}")
        print(f"Categorical Columns: {self.column_types['categorical']}")
        print(f"DateTime Columns   : {self.column_types['datetime']}")
        print()

        # Missing — only print if at least one column has missing values
        missing_cols = {
            col: pct for col, pct in self.missing_percentage.items() if pct > 0
        }
        if missing_cols:
            print("Missing Percentage:")
            for col, pct in missing_cols.items():
                print(f"  - {col}: {pct}%")
        else:
            print("Missing Percentage: None")

        # Warnings — only print keys that have non-empty values
        active_warnings = {k: v for k, v in self.warnings.items() if v}
        if active_warnings:
            print("\nWarnings:")
            for k, v in active_warnings.items():
                print(f"  - {k}: {v}")
        else:
            print("\nNo major data issues detected")

        print("\nRecommendations:")
        for k, v in self.recommendations.items():
            print(f"  - {k}: {v}")

        print("\nNumeric Diagnostics:")
        for col, info in self.numeric.items():
            print(
                f"  - {col}: skew={info['skewness']}, "
                f"outliers={info['outlier_percentage']}%, "
                f"{info['recommendation']}"
            )

        # --- v0.1.4 deep diagnostics ---

        if self.datetime_diagnostics:
            print("\nDatetime Diagnostics:")
            for col, info in self.datetime_diagnostics.items():
                print(
                    f"  - {col}: freq={info.get('inferred_frequency', 'N/A')}, "
                    f"issues={info.get('issues', [])}"
                )

        if self.drift_readiness:
            print("\nCategorical Drift Readiness:")
            for col, info in self.drift_readiness.items():
                print(
                    f"  - {col}: {info['recommendation']} "
                    f"(confidence={info['confidence']})"
                )

        if self.missing_patterns:
            print("\nMissing Pattern Clusters:")
            print(f"  {self.missing_patterns.get('summary', '')}")

        if self.leakage_report:
            print("\nTemporal Leakage Detection:")
            for col, info in self.leakage_report.items():
                print(
                    f"  - {col}: {info['recommendation']} "
                    f"(confidence={info['confidence']})"
                )

    # to_dict()

    def to_dict(self) -> dict:
        """
        Return the full report as a Python dictionary.
        All keys use consistent snake_case naming.
        """
        return {
            "health_score": self.health_score,
            "shape": self.shape,
            "column_types": self.column_types,
            "missing_percentage": self.missing_percentage,
            "duplicate_rows": self.duplicate_rows,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "numeric_diagnostics": self.numeric,
            # v0.1.4
            "datetime_diagnostics": self.datetime_diagnostics,
            "drift_readiness": self.drift_readiness,
            "missing_patterns": self.missing_patterns,
            "leakage_report": self.leakage_report,
        }

    # to_json()

    def to_json(self, path: str | None = None) -> dict:
        """
        Serialise the report to JSON.
        Optionally writes to a file when `path` is provided.
        Returns the serialised dict regardless.
        """
        import json

        data = self.to_dict()  # single source of truth — reuse to_dict

        if path:
            with open(path, "w") as f:
                json.dump(
                    data, f, indent=4, default=str
                )  # default=str handles Timedelta etc.

        return data

    # to_markdown()

    def to_markdown(self, path: str | None = None) -> str:
        """
        Render the report as a Markdown string.
        Optionally writes to a file when `path` is provided.
        Returns the Markdown string regardless.
        """
        lines = []

        # --- Title ---
        lines.append("# Data Health Report\n")

        # --- Summary ---
        lines.append("## Summary")
        lines.append(f"- **Health Score:** {self.health_score}/100")
        lines.append(f"- **Rows:** {self.shape['rows']}")
        lines.append(f"- **Columns:** {self.shape['columns']}")
        lines.append(f"- **Duplicate Rows:** {self.duplicate_rows}")
        lines.append("")

        # --- Column Types ---
        lines.append("## Column Types")
        for k, v in self.column_types.items():
            lines.append(f"- **{k.capitalize()}**: {v}")
        lines.append("")

        # --- Missing Percentage ---
        lines.append("## Missing Percentage")
        missing_cols = {
            col: pct for col, pct in self.missing_percentage.items() if pct > 0
        }
        if missing_cols:
            for col, pct in missing_cols.items():
                lines.append(f"- **{col}**: {pct}%")
        else:
            lines.append("- No missing values")
        lines.append("")

        # --- Warnings ---
        lines.append("## Warnings")
        active_warnings = {k: v for k, v in self.warnings.items() if v}
        if active_warnings:
            for k, v in active_warnings.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- No major warnings")
        lines.append("")

        # --- Recommendations ---
        lines.append("## Recommendations")
        if "encoding" in self.recommendations:
            lines.append("### Encoding")
            for col, info in self.recommendations["encoding"].items():
                lines.append(
                    f"- **{col}**: {info['message']} (Confidence: {info['confidence']})"
                )
        if "missing" in self.recommendations:
            lines.append("\n### Missing Values")
            for col, info in self.recommendations["missing"].items():
                lines.append(
                    f"- **{col}**: {info['action']} "
                    f"({info['message']}, Confidence: {info['confidence']})"
                )
        lines.append("")

        # --- Numeric Diagnostics ---
        lines.append("## Numeric Diagnostics")
        for col, info in self.numeric.items():
            lines.append(
                f"- **{col}**: skew={info['skewness']}, "
                f"outliers={info['outlier_percentage']}%, "
                f"{info['recommendation']}"
            )
        lines.append("")

        # --- v0.1.4: Datetime Diagnostics (#10) ---
        if self.datetime_diagnostics:
            lines.append("## Datetime Diagnostics")
            for col, info in self.datetime_diagnostics.items():
                lines.append(f"### {col}")
                lines.append(f"- **Null %:** {info.get('null_percentage', 'N/A')}%")
                lines.append(
                    f"- **Inferred Frequency:** {info.get('inferred_frequency', 'N/A')}"
                )
                lines.append(
                    f"- **Timezone Aware:** {info.get('timezone_aware', 'N/A')}"
                )
                lines.append(f"- **Future Dates:** {info.get('future_dates', 0)}")
                lines.append(f"- **Monotonic:** {info.get('is_monotonic', 'N/A')}")
                for issue in info.get("issues", []):
                    lines.append(f"-  {issue}")
                lines.append(f"- **Recommendation:** {info.get('recommendation', '')}")
            lines.append("")

        # --- v0.1.4: Categorical Drift Readiness (#12) ---
        if self.drift_readiness:
            lines.append("## Categorical Drift Readiness")
            for col, info in self.drift_readiness.items():
                lines.append(f"### {col}")
                lines.append(f"- **Unique Values:** {info['n_unique']}")
                lines.append(
                    f"- **Top Category Frequency:** {info['top_category_frequency']}%"
                )
                lines.append(f"- **Rare Categories:** {info['rare_category_count']}")
                lines.append(f"- **Entropy:** {info['entropy']}")
                for risk in info.get("drift_risks", []):
                    lines.append(f"-  {risk}")
                lines.append(
                    f"- **Recommendation:** {info['recommendation']} "
                    f"(Confidence: {info['confidence']})"
                )
            lines.append("")

        # --- v0.1.4: Missing Pattern Clusters (#13) ---
        if self.missing_patterns:
            lines.append("## Missing Pattern Clusters")
            lines.append(f"**Summary:** {self.missing_patterns.get('summary', '')}")
            lines.append("")
            col_mechanism = self.missing_patterns.get("column_mechanism", {})
            if col_mechanism:
                lines.append("| Column | Mechanism | Missing Rate | Explanation |")
                lines.append("|--------|-----------|-------------|-------------|")
                for col, info in col_mechanism.items():
                    lines.append(
                        f"| {col} | {info['mechanism']} | "
                        f"{info['missing_rate']}% | {info['explanation']} |"
                    )
            lines.append("")

        # --- v0.1.4: Temporal Leakage Detection (#14) ---
        if self.leakage_report:
            lines.append("## Temporal Leakage Detection")
            for col, info in self.leakage_report.items():
                lines.append(f"### {col}")
                lines.append(f"- **Future Dates:** {info.get('future_dates', 0)}")
                for risk in info.get("leakage_risks", []):
                    lines.append(f"-  {risk}")
                lines.append(
                    f"- **Recommendation:** {info['recommendation']} "
                    f"(Confidence: {info['confidence']})"
                )
            lines.append("")

        markdown_text = "\n".join(lines)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

        return markdown_text
