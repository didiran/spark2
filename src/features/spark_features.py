"""
PySpark-based feature engineering engine.

Implements distributed feature transformations including window
functions, temporal features, aggregations, normalization, encoding,
and interaction feature generation for ML training pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    Bucketizer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SparkFeatureEngine:
    """
    Distributed feature engineering engine built on PySpark.

    Provides a composable API for building feature transformation
    pipelines that run at scale on Spark clusters.

    Attributes:
        transformations: Ordered list of applied transformations for lineage.
    """

    def __init__(self):
        self.transformations: List[Dict[str, Any]] = []
        logger.info("SparkFeatureEngine initialized")

    def add_temporal_features(
        self,
        df: DataFrame,
        timestamp_column: str,
        features: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Extract temporal features from a timestamp column.

        Generates: year, month, day, hour, minute, day_of_week,
        day_of_year, week_of_year, quarter, is_weekend.

        Args:
            df: Input DataFrame.
            timestamp_column: Column containing timestamps.
            features: Subset of temporal features to extract.

        Returns:
            DataFrame with additional temporal feature columns.
        """
        all_features = {
            "year": F.year(F.col(timestamp_column)),
            "month": F.month(F.col(timestamp_column)),
            "day": F.dayofmonth(F.col(timestamp_column)),
            "hour": F.hour(F.col(timestamp_column)),
            "minute": F.minute(F.col(timestamp_column)),
            "day_of_week": F.dayofweek(F.col(timestamp_column)),
            "day_of_year": F.dayofyear(F.col(timestamp_column)),
            "week_of_year": F.weekofyear(F.col(timestamp_column)),
            "quarter": F.quarter(F.col(timestamp_column)),
            "is_weekend": (
                F.when(F.dayofweek(F.col(timestamp_column)).isin([1, 7]), 1).otherwise(0)
            ),
        }

        selected = features or list(all_features.keys())
        prefix = f"{timestamp_column}_"

        for feat_name in selected:
            if feat_name in all_features:
                col_name = f"{prefix}{feat_name}"
                df = df.withColumn(col_name, all_features[feat_name])

        self._record_transform("temporal_features", {
            "column": timestamp_column,
            "features": selected,
        })
        logger.info(f"Temporal features added | column={timestamp_column} | count={len(selected)}")
        return df

    def add_window_features(
        self,
        df: DataFrame,
        partition_columns: List[str],
        order_column: str,
        value_columns: List[str],
        window_sizes: List[int],
        aggregations: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Compute window-based aggregate features.

        Supports rolling mean, sum, min, max, stddev, and count
        over configurable window sizes.

        Args:
            df: Input DataFrame.
            partition_columns: Columns defining the partition key.
            order_column: Column for ordering within the window.
            value_columns: Numeric columns to aggregate.
            window_sizes: List of window row counts.
            aggregations: Aggregation functions (default: mean, stddev, min, max).

        Returns:
            DataFrame with windowed aggregate columns.
        """
        agg_funcs = aggregations or ["mean", "stddev", "min", "max"]

        agg_map = {
            "mean": F.avg,
            "sum": F.sum,
            "min": F.min,
            "max": F.max,
            "stddev": F.stddev,
            "count": F.count,
        }

        for window_size in window_sizes:
            window_spec = (
                Window
                .partitionBy(*partition_columns)
                .orderBy(order_column)
                .rowsBetween(-window_size, 0)
            )

            for col_name in value_columns:
                for agg_name in agg_funcs:
                    if agg_name in agg_map:
                        output_col = f"{col_name}_w{window_size}_{agg_name}"
                        df = df.withColumn(
                            output_col,
                            agg_map[agg_name](F.col(col_name)).over(window_spec),
                        )

        self._record_transform("window_features", {
            "partition_columns": partition_columns,
            "order_column": order_column,
            "value_columns": value_columns,
            "window_sizes": window_sizes,
            "aggregations": agg_funcs,
        })
        logger.info(
            f"Window features added | windows={window_sizes} "
            f"| columns={value_columns} | aggs={agg_funcs}"
        )
        return df

    def add_lag_features(
        self,
        df: DataFrame,
        partition_columns: List[str],
        order_column: str,
        value_columns: List[str],
        lag_sizes: List[int],
    ) -> DataFrame:
        """
        Add lagged value features for time-series analysis.

        Args:
            df: Input DataFrame.
            partition_columns: Columns defining the partition key.
            order_column: Column for ordering.
            value_columns: Columns to create lags for.
            lag_sizes: Number of rows to lag (e.g., [1, 3, 7]).

        Returns:
            DataFrame with lag feature columns.
        """
        window_spec = (
            Window
            .partitionBy(*partition_columns)
            .orderBy(order_column)
        )

        for col_name in value_columns:
            for lag in lag_sizes:
                output_col = f"{col_name}_lag_{lag}"
                df = df.withColumn(output_col, F.lag(F.col(col_name), lag).over(window_spec))

                diff_col = f"{col_name}_diff_{lag}"
                df = df.withColumn(
                    diff_col,
                    F.col(col_name) - F.col(output_col),
                )

        self._record_transform("lag_features", {
            "value_columns": value_columns,
            "lag_sizes": lag_sizes,
        })
        logger.info(f"Lag features added | lags={lag_sizes} | columns={value_columns}")
        return df

    def add_aggregation_features(
        self,
        df: DataFrame,
        group_columns: List[str],
        value_columns: List[str],
        aggregations: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Add group-level aggregation features via a left join.

        Args:
            df: Input DataFrame.
            group_columns: Columns to group by.
            value_columns: Columns to aggregate.
            aggregations: Aggregation types (default: mean, sum, count).

        Returns:
            DataFrame enriched with group aggregation features.
        """
        agg_funcs = aggregations or ["mean", "sum", "count"]

        agg_exprs = []
        for col_name in value_columns:
            for agg_name in agg_funcs:
                alias = f"{col_name}_grp_{agg_name}"
                if agg_name == "mean":
                    agg_exprs.append(F.avg(col_name).alias(alias))
                elif agg_name == "sum":
                    agg_exprs.append(F.sum(col_name).alias(alias))
                elif agg_name == "count":
                    agg_exprs.append(F.count(col_name).alias(alias))
                elif agg_name == "min":
                    agg_exprs.append(F.min(col_name).alias(alias))
                elif agg_name == "max":
                    agg_exprs.append(F.max(col_name).alias(alias))
                elif agg_name == "stddev":
                    agg_exprs.append(F.stddev(col_name).alias(alias))

        agg_df = df.groupBy(*group_columns).agg(*agg_exprs)
        df = df.join(agg_df, on=group_columns, how="left")

        self._record_transform("aggregation_features", {
            "group_columns": group_columns,
            "value_columns": value_columns,
            "aggregations": agg_funcs,
        })
        logger.info(
            f"Aggregation features added | groups={group_columns} | aggs={agg_funcs}"
        )
        return df

    def encode_categorical(
        self,
        df: DataFrame,
        columns: List[str],
        method: str = "onehot",
    ) -> Tuple[DataFrame, Pipeline]:
        """
        Encode categorical columns using StringIndexer + OneHotEncoder.

        Args:
            df: Input DataFrame.
            columns: Categorical columns to encode.
            method: Encoding method ("onehot" or "index").

        Returns:
            Tuple of (transformed DataFrame, fitted Pipeline).
        """
        stages = []
        for col_name in columns:
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_indexed",
                handleInvalid="keep",
            )
            stages.append(indexer)

            if method == "onehot":
                encoder = OneHotEncoder(
                    inputCol=f"{col_name}_indexed",
                    outputCol=f"{col_name}_encoded",
                )
                stages.append(encoder)

        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        df_encoded = model.transform(df)

        self._record_transform("encode_categorical", {
            "columns": columns,
            "method": method,
        })
        logger.info(f"Categorical encoding | columns={columns} | method={method}")
        return df_encoded, pipeline

    def assemble_features(
        self,
        df: DataFrame,
        input_columns: List[str],
        output_column: str = "features",
        with_scaling: bool = False,
    ) -> Tuple[DataFrame, Pipeline]:
        """
        Assemble feature columns into a single vector column.

        Optionally applies standard scaling to the assembled vector.

        Args:
            df: Input DataFrame.
            input_columns: List of numeric feature column names.
            output_column: Name for the output vector column.
            with_scaling: Whether to apply StandardScaler.

        Returns:
            Tuple of (transformed DataFrame, fitted Pipeline).
        """
        stages = []

        assembler = VectorAssembler(
            inputCols=input_columns,
            outputCol=f"{output_column}_raw" if with_scaling else output_column,
            handleInvalid="keep",
        )
        stages.append(assembler)

        if with_scaling:
            scaler = StandardScaler(
                inputCol=f"{output_column}_raw",
                outputCol=output_column,
                withStd=True,
                withMean=True,
            )
            stages.append(scaler)

        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        df_assembled = model.transform(df)

        self._record_transform("assemble_features", {
            "input_columns": input_columns,
            "output_column": output_column,
            "with_scaling": with_scaling,
        })
        logger.info(
            f"Features assembled | inputs={len(input_columns)} "
            f"| output={output_column} | scaled={with_scaling}"
        )
        return df_assembled, pipeline

    def add_interaction_features(
        self,
        df: DataFrame,
        column_pairs: List[Tuple[str, str]],
    ) -> DataFrame:
        """
        Create interaction features by multiplying column pairs.

        Args:
            df: Input DataFrame.
            column_pairs: List of (column_a, column_b) tuples.

        Returns:
            DataFrame with interaction feature columns.
        """
        for col_a, col_b in column_pairs:
            interaction_col = f"{col_a}_x_{col_b}"
            df = df.withColumn(interaction_col, F.col(col_a) * F.col(col_b))

        self._record_transform("interaction_features", {
            "column_pairs": column_pairs,
        })
        logger.info(f"Interaction features added | pairs={len(column_pairs)}")
        return df

    def add_ratio_features(
        self,
        df: DataFrame,
        numerator_denominator_pairs: List[Tuple[str, str]],
        fill_value: float = 0.0,
    ) -> DataFrame:
        """
        Create ratio features from column pairs.

        Args:
            df: Input DataFrame.
            numerator_denominator_pairs: List of (numerator, denominator) tuples.
            fill_value: Value to use when denominator is zero.

        Returns:
            DataFrame with ratio feature columns.
        """
        for num_col, den_col in numerator_denominator_pairs:
            ratio_col = f"{num_col}_per_{den_col}"
            df = df.withColumn(
                ratio_col,
                F.when(F.col(den_col) != 0, F.col(num_col) / F.col(den_col))
                .otherwise(fill_value),
            )

        self._record_transform("ratio_features", {
            "pairs": numerator_denominator_pairs,
        })
        logger.info(f"Ratio features added | pairs={len(numerator_denominator_pairs)}")
        return df

    def fill_nulls(
        self,
        df: DataFrame,
        strategy: str = "median",
        columns: Optional[List[str]] = None,
        fill_values: Optional[Dict[str, Any]] = None,
    ) -> DataFrame:
        """
        Fill null values using specified strategy.

        Args:
            df: Input DataFrame.
            strategy: "mean", "median", "zero", "mode", or "custom".
            columns: Columns to impute (default: all numeric columns).
            fill_values: Custom fill values per column (for strategy="custom").

        Returns:
            DataFrame with nulls filled.
        """
        if strategy == "custom" and fill_values:
            df = df.fillna(fill_values)
        elif strategy == "zero":
            target_cols = columns or [
                f.name for f in df.schema.fields
                if str(f.dataType) in ("DoubleType()", "FloatType()", "IntegerType()", "LongType()")
            ]
            df = df.fillna(0, subset=target_cols)
        elif strategy in ("mean", "median"):
            target_cols = columns or [
                f.name for f in df.schema.fields
                if str(f.dataType) in ("DoubleType()", "FloatType()", "IntegerType()", "LongType()")
            ]
            if strategy == "mean":
                stats = df.select([F.mean(c).alias(c) for c in target_cols]).first()
            else:
                stats_dict = {}
                for c in target_cols:
                    median_val = df.approxQuantile(c, [0.5], 0.01)
                    stats_dict[c] = median_val[0] if median_val else 0
                df = df.fillna(stats_dict)
                self._record_transform("fill_nulls", {"strategy": strategy, "columns": target_cols})
                return df

            if stats:
                fill_map = {c: stats[c] for c in target_cols if stats[c] is not None}
                df = df.fillna(fill_map)

        self._record_transform("fill_nulls", {"strategy": strategy})
        logger.info(f"Nulls filled | strategy={strategy}")
        return df

    def _record_transform(self, name: str, params: Dict[str, Any]) -> None:
        """Record a transformation step for lineage tracking."""
        self.transformations.append({
            "transform": name,
            "params": params,
        })

    def get_lineage(self) -> List[Dict[str, Any]]:
        """Return the ordered list of applied transformations."""
        return list(self.transformations)
