"""
Data processing module for Spark-Kafka ML Training Pipeline.

Provides distributed data cleaning, transformations, and feature
engineering with pandas fallback for local development.
"""

from src.processing.spark_processor import SparkProcessor
from src.processing.feature_engineering import FeatureEngineer

__all__ = ["SparkProcessor", "FeatureEngineer"]
