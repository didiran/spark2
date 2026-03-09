"""
Batch data loader for Delta Lake and other file-based sources.

Supports partitioned reads, incremental loading via timestamp
filtering, and schema evolution for the ML training pipeline.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from src.config.settings import StorageConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchDataLoader:
    """
    Delta Lake batch data loader with incremental load support.

    Reads partitioned Delta tables with optional time-range filtering,
    schema enforcement, and column pruning for efficient I/O.

    Attributes:
        spark: Active SparkSession.
        config: Storage configuration settings.
    """

    def __init__(self, spark: SparkSession, config: StorageConfig):
        self.spark = spark
        self.config = config
        self._load_history: List[Dict[str, Any]] = []
        logger.info(
            f"BatchDataLoader initialized | delta_base={config.delta_base_path}"
        )

    def read_delta(
        self,
        table_path: Optional[str] = None,
        table_name: Optional[str] = None,
        version: Optional[int] = None,
        timestamp_as_of: Optional[str] = None,
        columns: Optional[List[str]] = None,
        filter_condition: Optional[str] = None,
    ) -> DataFrame:
        """
        Read data from a Delta Lake table.

        Args:
            table_path: Full path to the Delta table directory.
            table_name: Table name resolved relative to delta_base_path.
            version: Specific Delta version for time-travel reads.
            timestamp_as_of: ISO timestamp for time-travel reads.
            columns: Column subset to select (column pruning).
            filter_condition: SQL WHERE clause for predicate pushdown.

        Returns:
            DataFrame containing the requested data.
        """
        path = table_path or f"{self.config.delta_base_path}/{table_name}"

        reader = self.spark.read.format("delta")

        if version is not None:
            reader = reader.option("versionAsOf", version)
            logger.info(f"Delta time-travel | version={version}")
        elif timestamp_as_of is not None:
            reader = reader.option("timestampAsOf", timestamp_as_of)
            logger.info(f"Delta time-travel | timestamp={timestamp_as_of}")

        df = reader.load(path)

        if columns:
            df = df.select(*columns)

        if filter_condition:
            df = df.filter(filter_condition)

        record_count = df.count()
        self._record_load(path, record_count, version=version)

        logger.info(f"Delta read complete | path={path} | records={record_count}")
        return df

    def read_partitioned(
        self,
        table_path: Optional[str] = None,
        table_name: Optional[str] = None,
        partition_filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Read a partitioned Delta table with partition pruning.

        Args:
            table_path: Full path to the Delta table directory.
            table_name: Table name resolved relative to delta_base_path.
            partition_filters: Dict of partition_column -> value(s) for pruning.
            columns: Column subset to select.

        Returns:
            DataFrame with partition-pruned data.
        """
        path = table_path or f"{self.config.delta_base_path}/{table_name}"

        df = self.spark.read.format("delta").load(path)

        if partition_filters:
            for col_name, value in partition_filters.items():
                if isinstance(value, list):
                    df = df.filter(F.col(col_name).isin(value))
                elif isinstance(value, tuple) and len(value) == 2:
                    df = df.filter(
                        (F.col(col_name) >= value[0]) & (F.col(col_name) <= value[1])
                    )
                else:
                    df = df.filter(F.col(col_name) == value)

        if columns:
            df = df.select(*columns)

        logger.info(
            f"Partitioned read | path={path} | filters={partition_filters}"
        )
        return df

    def incremental_load(
        self,
        table_path: Optional[str] = None,
        table_name: Optional[str] = None,
        timestamp_column: str = "event_timestamp",
        last_loaded_at: Optional[datetime] = None,
        lookback_hours: int = 24,
        columns: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Perform an incremental load based on a timestamp column.

        Loads only records newer than the last loaded timestamp,
        with an optional lookback window for late-arriving data.

        Args:
            table_path: Full path to the Delta table.
            table_name: Table name resolved relative to delta_base_path.
            timestamp_column: Column containing event timestamps.
            last_loaded_at: Cutoff datetime; loads records after this.
            lookback_hours: Hours to look back for late arrivals.
            columns: Column subset to select.

        Returns:
            DataFrame containing incrementally loaded records.
        """
        path = table_path or f"{self.config.delta_base_path}/{table_name}"

        if last_loaded_at is None:
            last_loaded_at = datetime.utcnow() - timedelta(hours=lookback_hours)

        adjusted_start = last_loaded_at - timedelta(hours=lookback_hours)

        df = self.spark.read.format("delta").load(path)

        df = df.filter(F.col(timestamp_column) >= F.lit(adjusted_start))

        if columns:
            df = df.select(*columns)

        record_count = df.count()
        self._record_load(
            path, record_count,
            incremental=True,
            since=str(adjusted_start),
        )

        logger.info(
            f"Incremental load | path={path} | since={adjusted_start} "
            f"| records={record_count}"
        )
        return df

    def read_jdbc(
        self,
        table: str,
        query: Optional[str] = None,
        num_partitions: int = 4,
        partition_column: Optional[str] = None,
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
    ) -> DataFrame:
        """
        Read data from PostgreSQL via JDBC.

        Args:
            table: Table name or subquery alias.
            query: Full SQL query (overrides table).
            num_partitions: Number of read partitions for parallelism.
            partition_column: Column for range-based partitioning.
            lower_bound: Lower bound for partition column.
            upper_bound: Upper bound for partition column.

        Returns:
            DataFrame from the PostgreSQL source.
        """
        jdbc_url = self.config.postgres_jdbc_url
        props = self.config.postgres_connection_props

        reader = self.spark.read.format("jdbc").options(
            url=jdbc_url,
            driver=props["driver"],
            user=props["user"],
            password=props["password"],
            numPartitions=str(num_partitions),
        )

        if query:
            reader = reader.option("query", query)
        else:
            reader = reader.option("dbtable", table)

        if partition_column and lower_bound is not None and upper_bound is not None:
            reader = reader.options(
                partitionColumn=partition_column,
                lowerBound=str(lower_bound),
                upperBound=str(upper_bound),
            )

        df = reader.load()
        logger.info(
            f"JDBC read | source={query or table} | partitions={num_partitions}"
        )
        return df

    def read_with_schema_enforcement(
        self,
        table_path: str,
        expected_schema: StructType,
        strict: bool = True,
    ) -> DataFrame:
        """
        Read Delta table with explicit schema enforcement.

        Args:
            table_path: Path to the Delta table.
            expected_schema: Expected StructType for validation.
            strict: If True, raises on schema mismatch; otherwise logs a warning.

        Returns:
            DataFrame matching the expected schema.

        Raises:
            ValueError: If strict mode is on and schema does not match.
        """
        df = self.spark.read.format("delta").load(table_path)
        actual_fields = set(df.schema.fieldNames())
        expected_fields = set(expected_schema.fieldNames())

        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields

        if missing:
            msg = f"Missing columns in source: {missing}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            for field in expected_schema.fields:
                if field.name in missing:
                    df = df.withColumn(field.name, F.lit(None).cast(field.dataType))

        if extra:
            logger.info(f"Extra columns in source (ignored): {extra}")
            df = df.select([f.name for f in expected_schema.fields if f.name in actual_fields | missing])

        return df

    def get_table_metadata(self, table_path: str) -> Dict[str, Any]:
        """
        Retrieve Delta table metadata including version and partition info.

        Args:
            table_path: Path to the Delta table.

        Returns:
            Dictionary with version, partition columns, and row count.
        """
        from delta.tables import DeltaTable

        dt = DeltaTable.forPath(self.spark, table_path)
        history = dt.history(1).collect()
        detail = dt.detail().collect()

        metadata = {
            "path": table_path,
            "current_version": history[0]["version"] if history else None,
            "last_operation": history[0]["operation"] if history else None,
            "last_updated": str(history[0]["timestamp"]) if history else None,
            "partition_columns": detail[0]["partitionColumns"] if detail else [],
            "num_files": detail[0]["numFiles"] if detail else 0,
            "size_bytes": detail[0]["sizeInBytes"] if detail else 0,
        }

        logger.info(f"Table metadata | {metadata}")
        return metadata

    def _record_load(self, path: str, record_count: int, **kwargs) -> None:
        """Record load operation in history for auditing."""
        entry = {
            "path": path,
            "record_count": record_count,
            "loaded_at": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._load_history.append(entry)

    @property
    def load_history(self) -> List[Dict[str, Any]]:
        """Return the recorded load history."""
        return list(self._load_history)
