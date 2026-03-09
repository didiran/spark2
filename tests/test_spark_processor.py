"""
Tests for SparkProcessor.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import pytest

from src.processing.spark_processor import SparkProcessor


class TestSparkProcessor:
    """Tests for the Spark processor (pandas-based)."""

    def setup_method(self):
        self.processor = SparkProcessor()

    def test_clean_data_removes_duplicates(self):
        df = pd.DataFrame({"id": [1, 1, 2, 3], "value": [10, 10, 20, 30]})
        result = self.processor.clean_data(df, drop_duplicates=True, subset_for_duplicates=["id"])
        assert len(result) == 3

    def test_fill_nulls_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = self.processor.fill_nulls(df, strategy="median")
        assert not result["a"].isnull().any()
        assert result["a"].iloc[1] == 3.0

    def test_fill_nulls_mean(self):
        df = pd.DataFrame({"a": [2.0, np.nan, 4.0]})
        result = self.processor.fill_nulls(df, strategy="mean")
        assert not result["a"].isnull().any()
        assert result["a"].iloc[1] == pytest.approx(3.0)

    def test_clip_outliers_iqr(self):
        np.random.seed(42)
        values = np.random.normal(100, 10, 100).tolist() + [500.0, -200.0]
        df = pd.DataFrame({"val": values})
        result = self.processor.clip_outliers(df, columns=["val"], method="iqr", iqr_multiplier=1.5)
        assert result["val"].max() < 500.0
        assert result["val"].min() > -200.0

    def test_normalize_columns_standard(self):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.normal(50, 10, 100)})
        result = self.processor.normalize_columns(df, columns=["x"], method="standard")
        assert result["x"].mean() == pytest.approx(0.0, abs=0.1)
        assert result["x"].std() == pytest.approx(1.0, abs=0.1)

    def test_get_data_profile(self, sample_fraud_df):
        profile = self.processor.get_data_profile(sample_fraud_df)
        assert "shape" in profile
        assert profile["shape"]["rows"] == 200
        assert "dtypes" in profile
        assert "null_counts" in profile

    def test_aggregate(self):
        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "value": [10, 20, 30, 40],
            }
        )
        result = self.processor.aggregate(df, group_by=["group"], agg_dict={"value": ["sum", "mean"]})
        assert len(result) == 2
