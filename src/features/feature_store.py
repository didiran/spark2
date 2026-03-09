"""
Delta Lake-based feature store for the ML Training Pipeline.

Provides versioned feature table management, point-in-time correct
joins for training data assembly, and feature metadata tracking.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config.settings import StorageConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeltaFeatureStore:
    """
    Offline feature store backed by Delta Lake.

    Manages versioned feature tables with support for point-in-time
    correct joins, incremental updates, and feature metadata tracking.

    Attributes:
        spark: Active SparkSession.
        config: Storage configuration settings.
        base_path: Root path for all feature tables.
    """

    def __init__(self, spark: SparkSession, config: StorageConfig):
        self.spark = spark
        self.config = config
        self.base_path = config.feature_store_path
        self._feature_registry: Dict[str, Dict[str, Any]] = {}
        logger.info(f"DeltaFeatureStore initialized | base_path={self.base_path}")

    def _table_path(self, feature_group: str) -> str:
        """Resolve the Delta table path for a feature group."""
        return f"{self.base_path}/{feature_group}"

    def register_feature_group(
        self,
        name: str,
        entity_key: str,
        timestamp_column: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register a new feature group in the store metadata.

        Args:
            name: Unique feature group identifier.
            entity_key: Column name used as the entity join key.
            timestamp_column: Column name used for event timestamps.
            description: Human-readable description of the feature group.
            tags: Optional key-value tags for organization.
        """
        self._feature_registry[name] = {
            "entity_key": entity_key,
            "timestamp_column": timestamp_column,
            "description": description,
            "tags": tags or {},
            "created_at": datetime.utcnow().isoformat(),
            "table_path": self._table_path(name),
            "versions": [],
        }
        logger.info(f"Feature group registered | name={name} | entity_key={entity_key}")

    def write_features(
        self,
        df: DataFrame,
        feature_group: str,
        mode: str = "merge",
        partition_columns: Optional[List[str]] = None,
        entity_key: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Write feature data to a versioned Delta table.

        Args:
            df: DataFrame containing feature data.
            feature_group: Target feature group name.
            mode: Write mode - "overwrite", "append", or "merge" (upsert).
            partition_columns: Columns to partition by.
            entity_key: Entity key column (overrides registry).
            timestamp_column: Timestamp column (overrides registry).

        Returns:
            Write operation metadata including version and row count.
        """
        table_path = self._table_path(feature_group)
        meta = self._feature_registry.get(feature_group, {})
        ek = entity_key or meta.get("entity_key", "entity_id")
        ts = timestamp_column or meta.get("timestamp_column", "event_timestamp")

        df = df.withColumn("_feature_store_updated_at", F.current_timestamp())

        row_count = df.count()

        if mode == "merge":
            self._merge_features(df, table_path, ek, ts)
        elif mode == "overwrite":
            writer = df.write.format("delta").mode("overwrite")
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            writer.save(table_path)
        else:
            writer = df.write.format("delta").mode("append")
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            writer.save(table_path)

        version_info = {
            "feature_group": feature_group,
            "mode": mode,
            "row_count": row_count,
            "written_at": datetime.utcnow().isoformat(),
            "table_path": table_path,
        }

        if feature_group in self._feature_registry:
            self._feature_registry[feature_group]["versions"].append(version_info)

        logger.info(
            f"Features written | group={feature_group} | mode={mode} "
            f"| rows={row_count}"
        )
        return version_info

    def _merge_features(
        self,
        source_df: DataFrame,
        table_path: str,
        entity_key: str,
        timestamp_column: str,
    ) -> None:
        """
        Upsert features using Delta Lake MERGE operation.

        Matches on entity_key + timestamp_column; updates existing
        records and inserts new ones.
        """
        from delta.tables import DeltaTable

        try:
            target = DeltaTable.forPath(self.spark, table_path)

            merge_condition = (
                f"target.{entity_key} = source.{entity_key} "
                f"AND target.{timestamp_column} = source.{timestamp_column}"
            )

            (
                target.alias("target")
                .merge(source_df.alias("source"), merge_condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            logger.info(f"Delta MERGE executed | path={table_path}")

        except Exception:
            logger.info(
                f"Target table does not exist, writing initial data | path={table_path}"
            )
            source_df.write.format("delta").mode("overwrite").save(table_path)

    def read_features(
        self,
        feature_group: str,
        version: Optional[int] = None,
        columns: Optional[List[str]] = None,
        filter_condition: Optional[str] = None,
    ) -> DataFrame:
        """
        Read features from a feature group table.

        Args:
            feature_group: Feature group name to read.
            version: Specific Delta version for time-travel.
            columns: Column subset to return.
            filter_condition: SQL WHERE clause.

        Returns:
            DataFrame containing requested features.
        """
        table_path = self._table_path(feature_group)

        reader = self.spark.read.format("delta")
        if version is not None:
            reader = reader.option("versionAsOf", version)

        df = reader.load(table_path)

        if columns:
            df = df.select(*columns)
        if filter_condition:
            df = df.filter(filter_condition)

        logger.info(
            f"Features read | group={feature_group} | version={version}"
        )
        return df

    def point_in_time_join(
        self,
        entity_df: DataFrame,
        feature_groups: List[str],
        entity_key: str,
        event_timestamp_column: str,
        feature_columns: Optional[Dict[str, List[str]]] = None,
    ) -> DataFrame:
        """
        Perform point-in-time correct feature joins.

        For each entity + event timestamp in entity_df, retrieves
        the latest feature values that were available at that point
        in time, preventing data leakage from the future.

        Args:
            entity_df: DataFrame with entities and event timestamps.
            feature_groups: List of feature group names to join.
            entity_key: Column name used as the join key.
            event_timestamp_column: Column with event timestamps.
            feature_columns: Optional dict mapping feature_group -> column list.

        Returns:
            DataFrame with point-in-time correct features joined.
        """
        result_df = entity_df

        for fg_name in feature_groups:
            meta = self._feature_registry.get(fg_name, {})
            fg_ts_col = meta.get("timestamp_column", "event_timestamp")
            table_path = self._table_path(fg_name)

            feature_df = self.spark.read.format("delta").load(table_path)

            selected_cols = None
            if feature_columns and fg_name in feature_columns:
                selected_cols = feature_columns[fg_name]
                keep_cols = [entity_key, fg_ts_col] + selected_cols
                feature_df = feature_df.select(
                    *[c for c in keep_cols if c in feature_df.columns]
                )

            fg_alias = f"fg_{fg_name}"
            feature_df = feature_df.alias(fg_alias)

            feature_df = feature_df.withColumnRenamed(
                fg_ts_col, f"__{fg_name}_ts"
            )

            joined = result_df.join(
                feature_df,
                on=(
                    (F.col(f"{fg_alias}.{entity_key}") == result_df[entity_key])
                    & (F.col(f"__{fg_name}_ts") <= result_df[event_timestamp_column])
                ),
                how="left",
            )

            from pyspark.sql import Window

            pit_window = (
                Window
                .partitionBy(result_df[entity_key], result_df[event_timestamp_column])
                .orderBy(F.col(f"__{fg_name}_ts").desc())
            )

            joined = joined.withColumn("_pit_rank", F.row_number().over(pit_window))
            joined = joined.filter(F.col("_pit_rank") == 1).drop(
                "_pit_rank", f"__{fg_name}_ts"
            )

            if entity_key in [f.name for f in feature_df.schema.fields]:
                joined = joined.drop(F.col(f"{fg_alias}.{entity_key}"))

            result_df = joined

        logger.info(
            f"Point-in-time join | feature_groups={feature_groups} "
            f"| entity_key={entity_key}"
        )
        return result_df

    def list_feature_groups(self) -> List[Dict[str, Any]]:
        """List all registered feature groups with metadata."""
        return [
            {"name": name, **meta}
            for name, meta in self._feature_registry.items()
        ]

    def get_feature_group_schema(self, feature_group: str) -> Optional[Dict[str, str]]:
        """
        Get the schema of a feature group table.

        Args:
            feature_group: Feature group name.

        Returns:
            Dictionary mapping column names to type strings, or None.
        """
        try:
            table_path = self._table_path(feature_group)
            df = self.spark.read.format("delta").load(table_path)
            return {f.name: str(f.dataType) for f in df.schema.fields}
        except Exception as e:
            logger.warning(f"Could not read schema for {feature_group}: {e}")
            return None

    def compact_table(self, feature_group: str, num_files: int = 16) -> None:
        """
        Compact small files in a feature group Delta table.

        Args:
            feature_group: Feature group name to compact.
            num_files: Target number of output files.
        """
        table_path = self._table_path(feature_group)
        df = self.spark.read.format("delta").load(table_path)
        df.coalesce(num_files).write.format("delta").mode("overwrite").save(table_path)
        logger.info(f"Table compacted | group={feature_group} | target_files={num_files}")

    def vacuum_table(self, feature_group: str, retention_hours: int = 168) -> None:
        """
        Remove old files from a feature group Delta table.

        Args:
            feature_group: Feature group name.
            retention_hours: Hours of history to retain (default: 7 days).
        """
        from delta.tables import DeltaTable

        table_path = self._table_path(feature_group)
        dt = DeltaTable.forPath(self.spark, table_path)
        dt.vacuum(retention_hours)
        logger.info(
            f"Table vacuumed | group={feature_group} "
            f"| retention_hours={retention_hours}"
        )
