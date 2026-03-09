"""
Kafka Structured Streaming consumer for the ML Training Pipeline.

Implements real-time data ingestion from Kafka topics using Spark
Structured Streaming with schema enforcement, watermarking, and
exactly-once checkpointing guarantees.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.config.settings import KafkaConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KafkaStreamReader:
    """
    Spark Structured Streaming reader for Kafka topics.

    Supports schema registry integration, event-time watermarking,
    fault-tolerant checkpointing, and pluggable deserialization.

    Attributes:
        spark: Active SparkSession.
        config: Kafka configuration settings.
    """

    def __init__(self, spark: SparkSession, config: KafkaConfig):
        self.spark = spark
        self.config = config
        self._active_queries: Dict[str, StreamingQuery] = {}
        logger.info(
            f"KafkaStreamReader initialized | brokers={config.bootstrap_servers} "
            f"| topics={config.topics}"
        )

    @staticmethod
    def default_schema() -> StructType:
        """Return the default schema for ML training data events."""
        return StructType([
            StructField("event_id", StringType(), nullable=False),
            StructField("timestamp", TimestampType(), nullable=False),
            StructField("entity_id", StringType(), nullable=False),
            StructField("feature_values", ArrayType(DoubleType()), nullable=True),
            StructField("feature_names", ArrayType(StringType()), nullable=True),
            StructField("label", DoubleType(), nullable=True),
            StructField("metadata", StringType(), nullable=True),
        ])

    def read_stream(
        self,
        schema: Optional[StructType] = None,
        event_time_column: str = "timestamp",
        watermark_delay: Optional[str] = None,
    ) -> DataFrame:
        """
        Create a streaming DataFrame from configured Kafka topics.

        Args:
            schema: Schema for JSON value deserialization.
                    Falls back to default_schema() if not provided.
            event_time_column: Column name used for watermark.
            watermark_delay: Watermark threshold (e.g., "10 seconds").

        Returns:
            Streaming DataFrame with parsed and watermarked records.
        """
        value_schema = schema or self.default_schema()
        options = self.config.to_spark_options()

        logger.info(f"Creating Kafka read stream | options={list(options.keys())}")

        raw_stream = (
            self.spark
            .readStream
            .format("kafka")
            .options(**options)
            .load()
        )

        parsed_stream = self._deserialize(raw_stream, value_schema)

        delay = watermark_delay or self.config.watermark_delay
        if event_time_column in parsed_stream.columns:
            parsed_stream = parsed_stream.withWatermark(event_time_column, delay)
            logger.info(
                f"Watermark applied | column={event_time_column} | delay={delay}"
            )

        return parsed_stream

    def _deserialize(
        self,
        raw_df: DataFrame,
        schema: StructType,
    ) -> DataFrame:
        """
        Deserialize Kafka message values from JSON bytes.

        Extracts Kafka metadata fields (key, topic, partition, offset, timestamp)
        alongside the parsed value payload.

        Args:
            raw_df: Raw Kafka DataFrame with key/value bytes.
            schema: Target schema for the JSON value column.

        Returns:
            DataFrame with deserialized and flattened columns.
        """
        parsed = (
            raw_df
            .select(
                F.col("key").cast("string").alias("kafka_key"),
                F.from_json(
                    F.col("value").cast("string"), schema
                ).alias("data"),
                F.col("topic").alias("kafka_topic"),
                F.col("partition").alias("kafka_partition"),
                F.col("offset").alias("kafka_offset"),
                F.col("timestamp").alias("kafka_timestamp"),
            )
            .select(
                "kafka_key",
                "kafka_topic",
                "kafka_partition",
                "kafka_offset",
                "kafka_timestamp",
                "data.*",
            )
        )
        return parsed

    def read_stream_with_registry(
        self,
        event_time_column: str = "timestamp",
        watermark_delay: Optional[str] = None,
    ) -> DataFrame:
        """
        Read stream using schema from the Confluent Schema Registry.

        Fetches the latest schema version for the subscribed topic(s)
        and applies it during deserialization.

        Args:
            event_time_column: Column name for watermarking.
            watermark_delay: Watermark threshold override.

        Returns:
            Streaming DataFrame deserialized with the registry schema.
        """
        schema = self._fetch_schema_from_registry()
        return self.read_stream(
            schema=schema,
            event_time_column=event_time_column,
            watermark_delay=watermark_delay,
        )

    def _fetch_schema_from_registry(self) -> StructType:
        """
        Fetch and convert the latest Avro schema from Schema Registry.

        Returns:
            PySpark StructType derived from the registry schema.
        """
        import requests

        topic = self.config.topics[0]
        subject = f"{topic}-value"
        url = f"{self.config.schema_registry_url}/subjects/{subject}/versions/latest"

        logger.info(f"Fetching schema from registry | url={url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            schema_data = response.json()
            avro_schema = json.loads(schema_data["schema"])
            return self._avro_to_spark_schema(avro_schema)
        except Exception as e:
            logger.warning(
                f"Schema registry fetch failed, using default | error={e}"
            )
            return self.default_schema()

    @staticmethod
    def _avro_to_spark_schema(avro_schema: Dict[str, Any]) -> StructType:
        """
        Convert an Avro schema dict to a PySpark StructType.

        Handles common Avro types: string, int, long, float, double,
        boolean, and nullable unions.

        Args:
            avro_schema: Parsed Avro schema dictionary.

        Returns:
            Equivalent PySpark StructType.
        """
        type_map = {
            "string": StringType(),
            "int": IntegerType(),
            "long": LongType(),
            "float": FloatType(),
            "double": DoubleType(),
            "boolean": StringType(),
        }

        fields = []
        for avro_field in avro_schema.get("fields", []):
            field_name = avro_field["name"]
            avro_type = avro_field["type"]

            nullable = False
            if isinstance(avro_type, list):
                non_null_types = [t for t in avro_type if t != "null"]
                nullable = "null" in avro_type
                avro_type = non_null_types[0] if non_null_types else "string"

            if isinstance(avro_type, str):
                spark_type = type_map.get(avro_type, StringType())
            elif isinstance(avro_type, dict) and avro_type.get("type") == "array":
                item_type = type_map.get(
                    avro_type.get("items", "string"), StringType()
                )
                spark_type = ArrayType(item_type)
            else:
                spark_type = StringType()

            fields.append(StructField(field_name, spark_type, nullable=nullable))

        return StructType(fields)

    def start_streaming_query(
        self,
        df: DataFrame,
        query_name: str,
        output_path: Optional[str] = None,
        output_mode: str = "append",
        trigger_interval: str = "30 seconds",
        format_type: str = "delta",
        partition_columns: Optional[List[str]] = None,
        foreach_batch_func: Optional[Callable] = None,
    ) -> StreamingQuery:
        """
        Start a streaming write query with checkpointing.

        Args:
            df: Streaming DataFrame to write.
            query_name: Unique identifier for the query.
            output_path: Sink path for file-based outputs.
            output_mode: Spark output mode (append, update, complete).
            trigger_interval: Processing interval string.
            format_type: Output format (delta, parquet, console, memory).
            partition_columns: Columns for data partitioning.
            foreach_batch_func: Optional micro-batch processing function.

        Returns:
            Active StreamingQuery handle.
        """
        checkpoint_path = f"{self.config.checkpoint_location}/{query_name}"

        writer = (
            df.writeStream
            .queryName(query_name)
            .outputMode(output_mode)
            .option("checkpointLocation", checkpoint_path)
            .trigger(processingTime=trigger_interval)
        )

        if foreach_batch_func is not None:
            query = writer.foreachBatch(foreach_batch_func).start()
        elif format_type == "console":
            query = writer.format("console").start()
        elif format_type == "memory":
            query = writer.format("memory").start()
        else:
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            query = writer.format(format_type).start(output_path)

        self._active_queries[query_name] = query
        logger.info(
            f"Streaming query started | name={query_name} | format={format_type} "
            f"| mode={output_mode} | trigger={trigger_interval}"
        )
        return query

    def stop_query(self, query_name: str) -> None:
        """Stop a running streaming query by name."""
        query = self._active_queries.pop(query_name, None)
        if query and query.isActive:
            query.stop()
            logger.info(f"Streaming query stopped | name={query_name}")
        else:
            logger.warning(f"No active query found | name={query_name}")

    def stop_all(self) -> None:
        """Stop all active streaming queries."""
        for name in list(self._active_queries.keys()):
            self.stop_query(name)
        logger.info("All streaming queries stopped")

    def get_query_status(self, query_name: str) -> Optional[Dict[str, Any]]:
        """Get progress information for a running query."""
        query = self._active_queries.get(query_name)
        if query is None:
            return None
        return {
            "name": query.name,
            "id": str(query.id),
            "is_active": query.isActive,
            "last_progress": query.lastProgress,
            "status": query.status,
        }
