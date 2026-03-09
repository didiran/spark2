"""
Feature store module for Spark-Kafka ML Training Pipeline.

Provides versioned feature management with metadata tracking,
feature registration, and retrieval capabilities.
"""

from src.store.feature_store import FeatureStore

__all__ = ["FeatureStore"]
