"""
Tests for FeatureEngineer.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Tests for the feature engineering module."""

    def setup_method(self):
        self.engineer = FeatureEngineer()

    def test_add_temporal_features(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
                "value": range(50),
            }
        )
        result = self.engineer.add_temporal_features(
            df, timestamp_column="timestamp", features=["hour", "day_of_week", "is_weekend"]
        )
        assert "timestamp_hour" in result.columns
        assert "timestamp_day_of_week" in result.columns
        assert "timestamp_is_weekend" in result.columns
        assert result["timestamp_hour"].between(0, 23).all()

    def test_encode_categorical_label(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})
        result = self.engineer.encode_categorical(df, columns=["color"], method="label")
        assert "color" in result.columns
        assert result["color"].dtype in (np.int64, np.int32, int, np.float64)

    def test_encode_categorical_onehot(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
        result = self.engineer.encode_categorical(df, columns=["color"], method="onehot")
        onehot_cols = [c for c in result.columns if c.startswith("color_")]
        assert len(onehot_cols) >= 2

    def test_add_interaction_features(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = self.engineer.add_interaction_features(df, column_pairs=[("a", "b")])
        assert "a_x_b" in result.columns
        assert result["a_x_b"].tolist() == [4.0, 10.0, 18.0]

    def test_add_ratio_features(self):
        df = pd.DataFrame({"numerator": [10.0, 20.0, 30.0], "denominator": [2.0, 5.0, 0.0]})
        result = self.engineer.add_ratio_features(
            df, numerator_denominator_pairs=[("numerator", "denominator")]
        )
        assert "numerator_div_denominator" in result.columns
        assert result["numerator_div_denominator"].iloc[0] == pytest.approx(5.0)

    def test_scale_features_standard(self):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.normal(100, 20, 100)})
        result = self.engineer.scale_features(df, columns=["val"], method="standard")
        assert result["val"].mean() == pytest.approx(0.0, abs=0.1)

    def test_lineage_tracking(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        self.engineer.add_interaction_features(df, column_pairs=[("a", "b")])
        lineage = self.engineer.get_lineage()
        assert len(lineage) >= 1
        assert lineage[0]["operation"] == "add_interaction_features"

    def test_select_features_variance(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "high_var": np.random.normal(0, 10, 100),
                "low_var": np.ones(100),
                "target": np.random.choice([0, 1], 100),
            }
        )
        result = self.engineer.select_features(
            df, target_column="target", method="variance", threshold=0.01
        )
        assert "high_var" in result.columns
        assert "low_var" not in result.columns
