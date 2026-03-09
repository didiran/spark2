"""
Data processing engine for the ML Training Pipeline.

Implements data cleaning, transformations, and aggregations with
pandas as the primary engine, designed to mirror Spark DataFrame
operations for seamless transition to distributed execution.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SparkProcessor:
    """
    Data processing engine using pandas with Spark-compatible API patterns.

    Provides data cleaning, type casting, deduplication, filtering,
    aggregation, and join operations that mirror PySpark DataFrame
    semantics for easy migration to distributed execution.

    Attributes:
        processing_log: Ordered list of transformations applied.
        stats: Processing statistics for monitoring.
    """

    def __init__(self):
        self.processing_log: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "total_rows_processed": 0,
            "total_rows_dropped": 0,
            "total_nulls_filled": 0,
            "total_duplicates_removed": 0,
            "start_time": datetime.utcnow().isoformat(),
        }
        logger.info("SparkProcessor initialized")

    def clean_data(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        subset_for_duplicates: Optional[List[str]] = None,
        drop_all_null_rows: bool = True,
        drop_all_null_cols: bool = False,
        null_threshold_row: float = 0.5,
    ) -> pd.DataFrame:
        """
        Apply standard data cleaning operations.

        Args:
            df: Input DataFrame.
            drop_duplicates: Remove duplicate rows.
            subset_for_duplicates: Columns to consider for duplicate detection.
            drop_all_null_rows: Remove rows where all values are null.
            drop_all_null_cols: Remove columns where all values are null.
            null_threshold_row: Drop rows with null fraction above this.

        Returns:
            Cleaned DataFrame.
        """
        initial_rows = len(df)
        initial_cols = len(df.columns)

        if drop_all_null_cols:
            df = df.dropna(axis=1, how="all")
            dropped_cols = initial_cols - len(df.columns)
            if dropped_cols > 0:
                logger.info(f"Dropped {dropped_cols} all-null columns")

        if drop_all_null_rows:
            df = df.dropna(how="all")

        if null_threshold_row < 1.0:
            null_fractions = df.isnull().mean(axis=1)
            df = df[null_fractions <= null_threshold_row]

        if drop_duplicates:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=subset_for_duplicates)
            dupes_removed = before_dedup - len(df)
            self.stats["total_duplicates_removed"] += dupes_removed
            if dupes_removed > 0:
                logger.info(f"Removed {dupes_removed} duplicate rows")

        rows_dropped = initial_rows - len(df)
        self.stats["total_rows_dropped"] += rows_dropped
        self.stats["total_rows_processed"] += initial_rows

        self._log_step("clean_data", {
            "initial_rows": initial_rows,
            "final_rows": len(df),
            "rows_dropped": rows_dropped,
        })

        logger.info(
            f"Data cleaning complete | {initial_rows} -> {len(df)} rows "
            f"| dropped={rows_dropped}"
        )
        return df.reset_index(drop=True)

    def cast_types(
        self,
        df: pd.DataFrame,
        type_mapping: Dict[str, str],
        errors: str = "coerce",
    ) -> pd.DataFrame:
        """
        Cast columns to specified data types.

        Args:
            df: Input DataFrame.
            type_mapping: Dict of column_name -> target_type.
            errors: Error handling strategy (coerce, raise, ignore).

        Returns:
            DataFrame with recast columns.
        """
        for col, target_type in type_mapping.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping type cast")
                continue

            try:
                if target_type in ("float", "float64", "float32"):
                    df[col] = pd.to_numeric(df[col], errors=errors)
                elif target_type in ("int", "int64", "int32"):
                    df[col] = pd.to_numeric(df[col], errors=errors)
                    df[col] = df[col].astype("Int64")
                elif target_type in ("datetime", "timestamp"):
                    df[col] = pd.to_datetime(df[col], errors=errors)
                elif target_type in ("str", "string", "object"):
                    df[col] = df[col].astype(str)
                elif target_type in ("bool", "boolean"):
                    df[col] = df[col].astype(bool)
                elif target_type == "category":
                    df[col] = df[col].astype("category")
                else:
                    df[col] = df[col].astype(target_type)
            except Exception as e:
                logger.warning(f"Type cast failed | col={col} | target={target_type} | error={e}")

        self._log_step("cast_types", {"mapping": type_mapping})
        return df

    def fill_nulls(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
        columns: Optional[List[str]] = None,
        fill_values: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fill null values using specified strategy.

        Args:
            df: Input DataFrame.
            strategy: Imputation strategy (mean, median, mode, zero, ffill, bfill, custom).
            columns: Columns to impute (default: all numeric).
            fill_values: Custom fill values per column (for strategy=custom).

        Returns:
            DataFrame with nulls filled.
        """
        initial_nulls = int(df.isnull().sum().sum())

        if strategy == "custom" and fill_values:
            df = df.fillna(fill_values)
        elif strategy == "zero":
            target_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
            df[target_cols] = df[target_cols].fillna(0)
        elif strategy == "ffill":
            target_cols = columns or df.columns.tolist()
            df[target_cols] = df[target_cols].ffill()
        elif strategy == "bfill":
            target_cols = columns or df.columns.tolist()
            df[target_cols] = df[target_cols].bfill()
        elif strategy in ("mean", "median", "mode"):
            target_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
            for col in target_cols:
                if col in df.columns:
                    if strategy == "mean":
                        fill_val = df[col].mean()
                    elif strategy == "median":
                        fill_val = df[col].median()
                    else:
                        mode_vals = df[col].mode()
                        fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 0
                    df[col] = df[col].fillna(fill_val)

        final_nulls = int(df.isnull().sum().sum())
        nulls_filled = initial_nulls - final_nulls
        self.stats["total_nulls_filled"] += nulls_filled

        self._log_step("fill_nulls", {
            "strategy": strategy,
            "nulls_filled": nulls_filled,
        })

        logger.info(f"Nulls filled | strategy={strategy} | filled={nulls_filled}")
        return df

    def filter_rows(
        self,
        df: pd.DataFrame,
        conditions: Dict[str, Any],
        mode: str = "and",
    ) -> pd.DataFrame:
        """
        Filter rows based on column conditions.

        Args:
            df: Input DataFrame.
            conditions: Dict of column -> (operator, value) or column -> value.
            mode: How to combine conditions ("and" or "or").

        Returns:
            Filtered DataFrame.
        """
        initial_rows = len(df)
        masks = []

        for col, condition in conditions.items():
            if col not in df.columns:
                continue

            if isinstance(condition, tuple) and len(condition) == 2:
                op, value = condition
                if op == "==":
                    masks.append(df[col] == value)
                elif op == "!=":
                    masks.append(df[col] != value)
                elif op == ">":
                    masks.append(df[col] > value)
                elif op == ">=":
                    masks.append(df[col] >= value)
                elif op == "<":
                    masks.append(df[col] < value)
                elif op == "<=":
                    masks.append(df[col] <= value)
                elif op == "in":
                    masks.append(df[col].isin(value))
                elif op == "not_in":
                    masks.append(~df[col].isin(value))
                elif op == "between":
                    masks.append(df[col].between(value[0], value[1]))
            else:
                masks.append(df[col] == condition)

        if not masks:
            return df

        if mode == "and":
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask

        df = df[combined_mask]

        self._log_step("filter_rows", {
            "initial_rows": initial_rows,
            "final_rows": len(df),
            "conditions": str(conditions),
            "mode": mode,
        })

        logger.info(f"Rows filtered | {initial_rows} -> {len(df)}")
        return df.reset_index(drop=True)

    def aggregate(
        self,
        df: pd.DataFrame,
        group_columns: List[str],
        aggregations: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """
        Perform grouped aggregation.

        Args:
            df: Input DataFrame.
            group_columns: Columns to group by.
            aggregations: Dict of column -> list of agg functions.

        Returns:
            Aggregated DataFrame with flattened column names.
        """
        agg_df = df.groupby(group_columns).agg(aggregations)
        agg_df.columns = [f"{col}_{agg}" for col, agg in agg_df.columns]
        agg_df = agg_df.reset_index()

        self._log_step("aggregate", {
            "group_columns": group_columns,
            "aggregations": {k: v for k, v in aggregations.items()},
            "result_rows": len(agg_df),
        })

        logger.info(
            f"Aggregation complete | groups={group_columns} | result_rows={len(agg_df)}"
        )
        return agg_df

    def join(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = "inner",
        suffix: str = "_right",
    ) -> pd.DataFrame:
        """
        Join two DataFrames.

        Args:
            left_df: Left DataFrame.
            right_df: Right DataFrame.
            on: Common join column.
            left_on: Left join column.
            right_on: Right join column.
            how: Join type (inner, left, right, outer).
            suffix: Suffix for duplicate columns.

        Returns:
            Joined DataFrame.
        """
        if on:
            result = left_df.merge(right_df, on=on, how=how, suffixes=("", suffix))
        else:
            result = left_df.merge(
                right_df,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=("", suffix),
            )

        self._log_step("join", {
            "left_rows": len(left_df),
            "right_rows": len(right_df),
            "result_rows": len(result),
            "how": how,
        })

        logger.info(
            f"Join complete | left={len(left_df)} | right={len(right_df)} "
            f"| result={len(result)} | type={how}"
        )
        return result

    def normalize_columns(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Normalize numeric columns.

        Args:
            df: Input DataFrame.
            columns: Columns to normalize (default: all numeric).
            method: Normalization method (standard, minmax, robust).

        Returns:
            Tuple of (normalized DataFrame, normalization parameters).
        """
        target_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        params: Dict[str, Dict[str, float]] = {}

        for col in target_cols:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            if method == "standard":
                mean = float(values.mean())
                std = float(values.std())
                if std > 0:
                    df[col] = (df[col] - mean) / std
                params[col] = {"mean": mean, "std": std}

            elif method == "minmax":
                min_val = float(values.min())
                max_val = float(values.max())
                range_val = max_val - min_val
                if range_val > 0:
                    df[col] = (df[col] - min_val) / range_val
                params[col] = {"min": min_val, "max": max_val}

            elif method == "robust":
                median = float(values.median())
                q1 = float(values.quantile(0.25))
                q3 = float(values.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    df[col] = (df[col] - median) / iqr
                params[col] = {"median": median, "q1": q1, "q3": q3, "iqr": iqr}

        self._log_step("normalize", {
            "method": method,
            "columns": target_cols,
        })

        logger.info(f"Normalization | method={method} | columns={len(target_cols)}")
        return df, params

    def add_derived_columns(
        self,
        df: pd.DataFrame,
        derivations: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    ) -> pd.DataFrame:
        """
        Add computed columns to the DataFrame.

        Args:
            df: Input DataFrame.
            derivations: Dict of new_column_name -> function(df) -> Series.

        Returns:
            DataFrame with additional derived columns.
        """
        for col_name, func in derivations.items():
            try:
                df[col_name] = func(df)
                logger.info(f"Derived column added | name={col_name}")
            except Exception as e:
                logger.warning(f"Derived column failed | name={col_name} | error={e}")

        self._log_step("add_derived_columns", {
            "columns": list(derivations.keys()),
        })
        return df

    def clip_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> pd.DataFrame:
        """
        Clip outlier values in numeric columns.

        Args:
            df: Input DataFrame.
            columns: Columns to clip (default: all numeric).
            method: Clipping method (iqr, percentile).
            iqr_multiplier: IQR multiplier for bounds.
            lower_percentile: Lower percentile for clipping.
            upper_percentile: Upper percentile for clipping.

        Returns:
            DataFrame with clipped values.
        """
        target_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in target_cols:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            if method == "iqr":
                q1 = float(values.quantile(0.25))
                q3 = float(values.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
            else:
                lower = float(values.quantile(lower_percentile))
                upper = float(values.quantile(upper_percentile))

            df[col] = df[col].clip(lower=lower, upper=upper)

        self._log_step("clip_outliers", {
            "method": method,
            "columns": target_cols,
        })

        logger.info(f"Outliers clipped | method={method} | columns={len(target_cols)}")
        return df

    def get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile.

        Args:
            df: DataFrame to profile.

        Returns:
            Dictionary with profiling statistics.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "numeric_summary": {},
            "categorical_summary": {},
        }

        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                profile["numeric_summary"][col] = {
                    "mean": round(float(values.mean()), 4),
                    "std": round(float(values.std()), 4),
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                    "median": round(float(values.median()), 4),
                    "skewness": round(float(values.skew()), 4),
                    "kurtosis": round(float(values.kurtosis()), 4),
                }

        for col in categorical_cols:
            profile["categorical_summary"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict(),
            }

        return profile

    def _log_step(self, step_name: str, details: Dict[str, Any]) -> None:
        """Record a processing step for lineage tracking."""
        self.processing_log.append({
            "step": step_name,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_processing_summary(self) -> Dict[str, Any]:
        """Return a summary of all processing operations."""
        return {
            "stats": self.stats,
            "steps": len(self.processing_log),
            "log": self.processing_log,
        }
