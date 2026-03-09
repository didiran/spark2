"""
Feature engineering engine for the ML Training Pipeline.

Implements feature transformations including temporal features, window
aggregations, interaction features, encoding, and feature selection
using pandas for local development and testing.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering engine using pandas and scikit-learn.

    Provides a composable API for building feature transformation
    pipelines including temporal extraction, window aggregations,
    categorical encoding, interaction features, and feature selection.

    Attributes:
        transformations: Ordered list of applied transformations.
        encoders: Fitted label encoders for categorical columns.
        scalers: Fitted scalers for numeric columns.
    """

    def __init__(self):
        self.transformations: List[Dict[str, Any]] = []
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        logger.info("FeatureEngineer initialized")

    def add_temporal_features(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
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
        if timestamp_column not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_column}' not found")
            return df

        ts = pd.to_datetime(df[timestamp_column])
        prefix = f"{timestamp_column}_"

        all_features = {
            "year": ts.dt.year,
            "month": ts.dt.month,
            "day": ts.dt.day,
            "hour": ts.dt.hour,
            "minute": ts.dt.minute,
            "day_of_week": ts.dt.dayofweek,
            "day_of_year": ts.dt.dayofyear,
            "week_of_year": ts.dt.isocalendar().week.astype(int),
            "quarter": ts.dt.quarter,
            "is_weekend": (ts.dt.dayofweek >= 5).astype(int),
        }

        selected = features or list(all_features.keys())

        for feat_name in selected:
            if feat_name in all_features:
                col_name = f"{prefix}{feat_name}"
                df[col_name] = all_features[feat_name].values

        self._record_transform("temporal_features", {
            "column": timestamp_column,
            "features": selected,
        })
        logger.info(
            f"Temporal features added | column={timestamp_column} "
            f"| count={len(selected)}"
        )
        return df

    def add_window_features(
        self,
        df: pd.DataFrame,
        partition_columns: List[str],
        order_column: str,
        value_columns: List[str],
        window_sizes: List[int],
        aggregations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute window-based aggregate features.

        Args:
            df: Input DataFrame.
            partition_columns: Columns defining the partition key.
            order_column: Column for ordering within the window.
            value_columns: Numeric columns to aggregate.
            window_sizes: List of window row counts.
            aggregations: Aggregation functions (default: mean, std, min, max).

        Returns:
            DataFrame with windowed aggregate columns.
        """
        agg_funcs = aggregations or ["mean", "std", "min", "max"]

        df = df.sort_values(by=partition_columns + [order_column]).reset_index(drop=True)

        for window_size in window_sizes:
            for col_name in value_columns:
                if col_name not in df.columns:
                    continue

                grouped = df.groupby(partition_columns)[col_name]

                for agg_name in agg_funcs:
                    output_col = f"{col_name}_w{window_size}_{agg_name}"
                    if agg_name == "mean":
                        df[output_col] = grouped.transform(
                            lambda x: x.rolling(window_size, min_periods=1).mean()
                        )
                    elif agg_name == "std":
                        df[output_col] = grouped.transform(
                            lambda x: x.rolling(window_size, min_periods=1).std()
                        )
                    elif agg_name == "min":
                        df[output_col] = grouped.transform(
                            lambda x: x.rolling(window_size, min_periods=1).min()
                        )
                    elif agg_name == "max":
                        df[output_col] = grouped.transform(
                            lambda x: x.rolling(window_size, min_periods=1).max()
                        )
                    elif agg_name == "sum":
                        df[output_col] = grouped.transform(
                            lambda x: x.rolling(window_size, min_periods=1).sum()
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
        df: pd.DataFrame,
        partition_columns: List[str],
        order_column: str,
        value_columns: List[str],
        lag_sizes: List[int],
    ) -> pd.DataFrame:
        """
        Add lagged value features for time-series analysis.

        Args:
            df: Input DataFrame.
            partition_columns: Columns defining the partition key.
            order_column: Column for ordering.
            value_columns: Columns to create lags for.
            lag_sizes: Number of rows to lag.

        Returns:
            DataFrame with lag feature columns.
        """
        df = df.sort_values(by=partition_columns + [order_column]).reset_index(drop=True)

        for col_name in value_columns:
            if col_name not in df.columns:
                continue

            grouped = df.groupby(partition_columns)[col_name]

            for lag in lag_sizes:
                lag_col = f"{col_name}_lag_{lag}"
                df[lag_col] = grouped.shift(lag)

                diff_col = f"{col_name}_diff_{lag}"
                df[diff_col] = df[col_name] - df[lag_col]

        self._record_transform("lag_features", {
            "value_columns": value_columns,
            "lag_sizes": lag_sizes,
        })
        logger.info(f"Lag features added | lags={lag_sizes} | columns={value_columns}")
        return df

    def add_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Create interaction features by multiplying column pairs.

        Args:
            df: Input DataFrame.
            column_pairs: List of (column_a, column_b) tuples.

        Returns:
            DataFrame with interaction feature columns.
        """
        for col_a, col_b in column_pairs:
            if col_a in df.columns and col_b in df.columns:
                interaction_col = f"{col_a}_x_{col_b}"
                df[interaction_col] = df[col_a] * df[col_b]

        self._record_transform("interaction_features", {
            "column_pairs": column_pairs,
        })
        logger.info(f"Interaction features added | pairs={len(column_pairs)}")
        return df

    def add_ratio_features(
        self,
        df: pd.DataFrame,
        numerator_denominator_pairs: List[Tuple[str, str]],
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        """
        Create ratio features from column pairs.

        Args:
            df: Input DataFrame.
            numerator_denominator_pairs: List of (numerator, denominator).
            fill_value: Value to use when denominator is zero.

        Returns:
            DataFrame with ratio feature columns.
        """
        for num_col, den_col in numerator_denominator_pairs:
            if num_col in df.columns and den_col in df.columns:
                ratio_col = f"{num_col}_per_{den_col}"
                df[ratio_col] = np.where(
                    df[den_col] != 0,
                    df[num_col] / df[den_col],
                    fill_value,
                )

        self._record_transform("ratio_features", {
            "pairs": numerator_denominator_pairs,
        })
        logger.info(f"Ratio features added | pairs={len(numerator_denominator_pairs)}")
        return df

    def add_aggregation_features(
        self,
        df: pd.DataFrame,
        group_columns: List[str],
        value_columns: List[str],
        aggregations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add group-level aggregation features via merge.

        Args:
            df: Input DataFrame.
            group_columns: Columns to group by.
            value_columns: Columns to aggregate.
            aggregations: Aggregation types (default: mean, sum, count).

        Returns:
            DataFrame enriched with group aggregation features.
        """
        agg_funcs = aggregations or ["mean", "sum", "count"]

        agg_dict = {}
        for col_name in value_columns:
            for agg_name in agg_funcs:
                agg_dict[(col_name, agg_name)] = (col_name, agg_name)

        agg_mapping = {col: agg_funcs for col in value_columns}
        agg_df = df.groupby(group_columns).agg(agg_mapping)
        agg_df.columns = [f"{col}_grp_{agg}" for col, agg in agg_df.columns]
        agg_df = agg_df.reset_index()

        df = df.merge(agg_df, on=group_columns, how="left")

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
        df: pd.DataFrame,
        columns: List[str],
        method: str = "label",
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """
        Encode categorical columns.

        Args:
            df: Input DataFrame.
            columns: Categorical columns to encode.
            method: Encoding method (label, onehot, frequency).
            drop_original: Whether to drop original columns.

        Returns:
            DataFrame with encoded columns.
        """
        for col in columns:
            if col not in df.columns:
                continue

            if method == "label":
                encoder = LabelEncoder()
                df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder

            elif method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df, dummies], axis=1)

            elif method == "frequency":
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f"{col}_freq"] = df[col].map(freq_map)

        if drop_original:
            df = df.drop(columns=[c for c in columns if c in df.columns])

        self._record_transform("encode_categorical", {
            "columns": columns,
            "method": method,
        })
        logger.info(f"Categorical encoding | columns={columns} | method={method}")
        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
    ) -> pd.DataFrame:
        """
        Scale numeric features.

        Args:
            df: Input DataFrame.
            columns: Columns to scale (default: all numeric).
            method: Scaling method (standard, minmax, robust).

        Returns:
            DataFrame with scaled features.
        """
        target_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in target_cols:
            if col not in df.columns:
                continue

            values = df[[col]].values
            scaler = StandardScaler()

            if method == "standard":
                df[col] = scaler.fit_transform(values).flatten()
            elif method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(values).flatten()
            elif method == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(values).flatten()

            self.scalers[col] = scaler

        self._record_transform("scale_features", {
            "method": method,
            "columns": target_cols,
        })
        logger.info(f"Feature scaling | method={method} | columns={len(target_cols)}")
        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        method: str = "correlation",
        top_k: int = 20,
        threshold: float = 0.01,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top features based on importance or correlation.

        Args:
            df: Input DataFrame.
            target_column: Target column for feature selection.
            method: Selection method (correlation, variance).
            top_k: Number of top features to select.
            threshold: Minimum threshold for feature selection.

        Returns:
            Tuple of (DataFrame with selected features, list of selected column names).
        """
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_column
        ]

        if method == "correlation":
            correlations = df[numeric_cols].corrwith(df[target_column]).abs()
            correlations = correlations.dropna().sort_values(ascending=False)
            selected = correlations[correlations >= threshold].head(top_k).index.tolist()

        elif method == "variance":
            variances = df[numeric_cols].var()
            variances = variances.sort_values(ascending=False)
            selected = variances[variances >= threshold].head(top_k).index.tolist()

        else:
            selected = numeric_cols[:top_k]

        keep_cols = selected + [target_column]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != target_column]
        keep_cols = non_numeric_cols + keep_cols

        self._record_transform("select_features", {
            "method": method,
            "top_k": top_k,
            "selected_count": len(selected),
            "selected_features": selected,
        })
        logger.info(
            f"Feature selection | method={method} | selected={len(selected)}/{len(numeric_cols)}"
        )
        return df[keep_cols], selected

    def _record_transform(self, name: str, params: Dict[str, Any]) -> None:
        """Record a transformation step for lineage tracking."""
        self.transformations.append({
            "transform": name,
            "params": params,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_lineage(self) -> List[Dict[str, Any]]:
        """Return the ordered list of applied transformations."""
        return list(self.transformations)
