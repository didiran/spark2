"""
Data validation module for Spark-Kafka ML Training Pipeline.

Provides distributed data quality checks including schema enforcement,
null detection, outlier analysis, and Great Expectations-style rules.
"""

from src.validation.data_validator import DataValidator

__all__ = ["DataValidator"]
