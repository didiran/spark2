"""
Model evaluation module for Spark-Kafka ML Training Pipeline.

Provides distributed evaluation on Spark with comprehensive
metrics aggregation and quality gates for deployment decisions.
"""

from src.evaluation.evaluator import PipelineEvaluator

__all__ = ["PipelineEvaluator"]
