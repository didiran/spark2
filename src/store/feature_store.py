"""
Feature store for the ML Training Pipeline.

Provides versioned feature management with metadata tracking,
feature registration, retrieval, and lineage capabilities
using pandas DataFrames and local file storage.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Local feature store with versioning and metadata management.

    Manages versioned feature sets with support for registration,
    retrieval, listing, metadata tracking, and lineage.

    Attributes:
        base_path: Root directory for feature storage.
        registry: In-memory feature group registry.
    """

    def __init__(self, base_path: str = "./feature_store"):
        """
        Initialize the feature store.

        Args:
            base_path: Root directory for feature storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.base_path / "_registry.json"
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()
        logger.info(f"FeatureStore initialized | base_path={self.base_path}")

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the feature registry from disk."""
        if self._registry_path.exists():
            with open(self._registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        """Persist the feature registry to disk."""
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_feature_group(
        self,
        name: str,
        description: str = "",
        entity_key: str = "entity_id",
        timestamp_column: str = "timestamp",
        tags: Optional[Dict[str, str]] = None,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new feature group in the store.

        Args:
            name: Unique feature group identifier.
            description: Human-readable description.
            entity_key: Column name for entity join key.
            timestamp_column: Column for event timestamps.
            tags: Optional key-value tags.
            schema: Optional schema definition.

        Returns:
            Registration metadata dictionary.
        """
        group_path = self.base_path / name
        group_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "name": name,
            "description": description,
            "entity_key": entity_key,
            "timestamp_column": timestamp_column,
            "tags": tags or {},
            "schema": schema,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "versions": [],
            "latest_version": 0,
            "total_rows": 0,
        }

        self.registry[name] = metadata
        self._save_registry()

        logger.info(
            f"Feature group registered | name={name} | entity_key={entity_key}"
        )
        return metadata

    def ingest_features(
        self,
        name: str,
        df: pd.DataFrame,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Ingest a DataFrame as a new version of a feature group.

        Args:
            name: Feature group name.
            df: DataFrame containing feature data.
            description: Version description.

        Returns:
            Version metadata dictionary.

        Raises:
            ValueError: If the feature group is not registered.
        """
        if name not in self.registry:
            self.register_feature_group(name)

        meta = self.registry[name]
        new_version = meta["latest_version"] + 1
        version_dir = self.base_path / name / f"v{new_version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        data_path = version_dir / "data.parquet"
        df.to_parquet(str(data_path), index=False)

        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()

        columns_info = {
            col: str(df[col].dtype)
            for col in df.columns
        }

        version_meta = {
            "version": new_version,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns_info,
            "data_hash": data_hash,
            "file_path": str(data_path),
            "size_bytes": data_path.stat().st_size,
        }

        meta["versions"].append(version_meta)
        meta["latest_version"] = new_version
        meta["updated_at"] = datetime.utcnow().isoformat()
        meta["total_rows"] = len(df)
        meta["schema"] = columns_info
        self._save_registry()

        version_meta_path = version_dir / "metadata.json"
        with open(version_meta_path, "w", encoding="utf-8") as f:
            json.dump(version_meta, f, indent=2, default=str)

        logger.info(
            f"Features ingested | group={name} | version={new_version} "
            f"| rows={len(df)} | columns={len(df.columns)}"
        )
        return version_meta

    def get_features(
        self,
        name: str,
        version: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve features from the store.

        Args:
            name: Feature group name.
            version: Specific version to retrieve (default: latest).
            columns: Column subset to return.

        Returns:
            DataFrame containing the requested features.

        Raises:
            ValueError: If the feature group or version is not found.
        """
        if name not in self.registry:
            raise ValueError(f"Feature group '{name}' not found in registry")

        meta = self.registry[name]
        target_version = version or meta["latest_version"]

        if target_version < 1 or target_version > meta["latest_version"]:
            raise ValueError(
                f"Version {target_version} not found for group '{name}'. "
                f"Available: 1-{meta['latest_version']}"
            )

        data_path = self.base_path / name / f"v{target_version}" / "data.parquet"

        if not data_path.exists():
            raise FileNotFoundError(f"Feature data not found: {data_path}")

        df = pd.read_parquet(str(data_path))

        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        logger.info(
            f"Features retrieved | group={name} | version={target_version} "
            f"| rows={len(df)} | columns={len(df.columns)}"
        )
        return df

    def list_feature_groups(self) -> List[Dict[str, Any]]:
        """
        List all registered feature groups with metadata.

        Returns:
            List of feature group metadata dictionaries.
        """
        groups = []
        for name, meta in self.registry.items():
            groups.append({
                "name": meta["name"],
                "description": meta["description"],
                "entity_key": meta["entity_key"],
                "latest_version": meta["latest_version"],
                "total_rows": meta["total_rows"],
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
                "tags": meta.get("tags", {}),
            })
        return groups

    def get_feature_group_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a feature group.

        Args:
            name: Feature group name.

        Returns:
            Full metadata dictionary for the feature group.

        Raises:
            ValueError: If the feature group is not found.
        """
        if name not in self.registry:
            raise ValueError(f"Feature group '{name}' not found")
        return self.registry[name]

    def get_version_history(self, name: str) -> List[Dict[str, Any]]:
        """
        Get version history for a feature group.

        Args:
            name: Feature group name.

        Returns:
            List of version metadata dictionaries.
        """
        if name not in self.registry:
            raise ValueError(f"Feature group '{name}' not found")
        return self.registry[name]["versions"]

    def compare_versions(
        self,
        name: str,
        version_a: int,
        version_b: int,
    ) -> Dict[str, Any]:
        """
        Compare two versions of a feature group.

        Args:
            name: Feature group name.
            version_a: First version number.
            version_b: Second version number.

        Returns:
            Comparison report dictionary.
        """
        df_a = self.get_features(name, version=version_a)
        df_b = self.get_features(name, version=version_b)

        comparison = {
            "group": name,
            "version_a": version_a,
            "version_b": version_b,
            "rows_a": len(df_a),
            "rows_b": len(df_b),
            "row_diff": len(df_b) - len(df_a),
            "columns_a": list(df_a.columns),
            "columns_b": list(df_b.columns),
            "added_columns": list(set(df_b.columns) - set(df_a.columns)),
            "removed_columns": list(set(df_a.columns) - set(df_b.columns)),
        }

        common_numeric = [
            c for c in df_a.select_dtypes(include=["number"]).columns
            if c in df_b.columns
        ]

        stats_diff = {}
        for col in common_numeric:
            stats_diff[col] = {
                "mean_diff": float(df_b[col].mean() - df_a[col].mean()),
                "std_diff": float(df_b[col].std() - df_a[col].std()),
            }

        comparison["stats_diff"] = stats_diff

        logger.info(
            f"Version comparison | group={name} | v{version_a} vs v{version_b}"
        )
        return comparison

    def delete_version(self, name: str, version: int) -> bool:
        """
        Delete a specific version of a feature group.

        Args:
            name: Feature group name.
            version: Version number to delete.

        Returns:
            True if the version was deleted.

        Raises:
            ValueError: If the feature group or version is not found.
        """
        if name not in self.registry:
            raise ValueError(f"Feature group '{name}' not found")

        version_dir = self.base_path / name / f"v{version}"
        if version_dir.exists():
            shutil.rmtree(str(version_dir))

        self.registry[name]["versions"] = [
            v for v in self.registry[name]["versions"]
            if v["version"] != version
        ]
        self._save_registry()

        logger.info(f"Version deleted | group={name} | version={version}")
        return True

    def cleanup(self) -> None:
        """Remove all feature store data and registry."""
        if self.base_path.exists():
            shutil.rmtree(str(self.base_path))
        self.registry = {}
        logger.info("Feature store cleaned up")
