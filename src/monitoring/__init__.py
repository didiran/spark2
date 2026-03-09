"""
Pipeline monitoring module for Spark-Kafka ML Training Pipeline.

Provides health tracking, stage duration monitoring, data volume
metrics, and alerting for pipeline operations.
"""

from src.monitoring.pipeline_monitor import PipelineMonitor

__all__ = ["PipelineMonitor"]
