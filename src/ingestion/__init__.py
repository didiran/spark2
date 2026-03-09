"""
Data ingestion module for Spark-Kafka ML Training Pipeline.

Provides streaming and batch data loading capabilities with support
for Kafka Structured Streaming and Delta Lake batch reads.
"""

from src.ingestion.kafka_consumer import KafkaStreamReader
from src.ingestion.batch_loader import BatchDataLoader

__all__ = ["KafkaStreamReader", "BatchDataLoader"]
