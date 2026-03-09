"""
Tests for StandaloneDataValidator.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.data_validator_standalone import RuleSeverity, StandaloneDataValidator


class TestStandaloneDataValidator:
    """Tests for the standalone data validator."""

    def test_expect_column_not_null_passes(self, sample_fraud_df):
        validator = StandaloneDataValidator()
        validator.expect_column_not_null("transaction_id", severity=RuleSeverity.CRITICAL)
        report = validator.validate(sample_fraud_df)
        assert report.is_valid
        assert report.passed_rules == 1

    def test_expect_column_not_null_fails_with_nulls(self):
        df = pd.DataFrame({"col_a": [1, None, 3, None, 5]})
        validator = StandaloneDataValidator()
        validator.expect_column_not_null("col_a", severity=RuleSeverity.WARNING)
        report = validator.validate(df)
        assert report.failed_rules == 1

    def test_expect_column_values_in_range(self, sample_fraud_df):
        validator = StandaloneDataValidator()
        validator.expect_column_values_in_range("amount", min_value=0.0, max_value=100000.0)
        report = validator.validate(sample_fraud_df)
        assert report.passed_rules == 1

    def test_expect_column_values_in_set(self, sample_fraud_df):
        validator = StandaloneDataValidator()
        validator.expect_column_values_in_set("card_type", ["visa", "mastercard", "amex", "discover"])
        report = validator.validate(sample_fraud_df)
        assert report.passed_rules == 1

    def test_expect_column_values_in_set_fails(self):
        df = pd.DataFrame({"status": ["active", "inactive", "unknown"]})
        validator = StandaloneDataValidator()
        validator.expect_column_values_in_set("status", ["active", "inactive"])
        report = validator.validate(df)
        assert report.failed_rules == 1

    def test_detect_outliers_iqr(self, sample_fraud_df):
        validator = StandaloneDataValidator()
        validator.detect_outliers_iqr("amount", max_outlier_fraction=0.5)
        report = validator.validate(sample_fraud_df)
        assert report.total_rules == 1

    def test_expect_row_count(self, sample_fraud_df):
        validator = StandaloneDataValidator()
        validator.expect_row_count(min_count=100, severity=RuleSeverity.CRITICAL)
        report = validator.validate(sample_fraud_df)
        assert report.passed_rules == 1

    def test_quality_score_is_percentage(self, sample_fraud_df):
        validator = StandaloneDataValidator()
        validator.expect_column_not_null("transaction_id")
        validator.expect_column_not_null("amount")
        validator.expect_row_count(min_count=50)
        report = validator.validate(sample_fraud_df)
        assert 0 <= report.quality_score <= 100

    def test_report_to_dict(self, sample_fraud_df):
        validator = StandaloneDataValidator()
        validator.expect_column_not_null("amount")
        report = validator.validate(sample_fraud_df)
        report_dict = report.to_dict()
        assert "quality_score" in report_dict
        assert "is_valid" in report_dict
        assert "results" in report_dict
