from sklearn.base import BaseEstimator, TransformerMixin

from yreport import data_health_report


class YReportInspector(BaseEstimator, TransformerMixin):
    """
    A sklearn-compatible inspector that generates a YReport during `fit`
    and passes data through unchanged during `transform`.

    Designed to sit at the start of a Pipeline so you can inspect the
    raw data health before any preprocessing steps run.

    Parameters
    ----------
    drop_cols       : Columns to force-drop in recommendations.
    categorical_cols: Columns to force-treat as categorical.
    numerical_cols  : Columns to force-treat as numeric.
    ignore_cols     : Columns to exclude entirely from analysis.

    Attributes
    ----------
    report_ : DataHealthReport
        Populated after `fit` is called. Access via
        ``pipeline.named_steps['inspector'].report_``
    """

    def __init__(
        self,
        drop_cols=None,
        categorical_cols=None,
        numerical_cols=None,
        ignore_cols=None,
    ):
        self.drop_cols = drop_cols
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.ignore_cols = ignore_cols

    def fit(self, x, y=None):
        """
        Run the data health report on X and store as self.report_.
        Follows sklearn convention: X not modified, y ignored.
        """
        self.report_ = data_health_report(
            x,
            drop_cols=self.drop_cols,
            categorical_cols=self.categorical_cols,
            numeric_cols=self.numerical_cols,  # maps to health.py param name
            ignore_cols=self.ignore_cols,
        )
        return self

    def transform(self, x):
        """
        Pass X through unchanged.
        YReportInspector is a diagnostic tool, not a data mutator.
        """
        return x
