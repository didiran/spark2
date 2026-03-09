"""
Tests for FeatureStore.

Author: Gabriel Demetrios Lafis
"""

import pandas as pd
import pytest

from src.store.feature_store import FeatureStore


class TestFeatureStore:
    """Tests for the parquet-based feature store."""

    def test_register_feature_group(self, tmp_dir):
        store = FeatureStore(base_path=str(tmp_dir / "fs"))
        store.register_feature_group(
            name="test_group",
            description="Test feature group",
            entity_key="user_id",
            timestamp_column="ts",
        )
        groups = store.list_feature_groups()
        assert "test_group" in groups

    def test_ingest_and_retrieve_features(self, tmp_dir):
        store = FeatureStore(base_path=str(tmp_dir / "fs"))
        store.register_feature_group(
            name="features",
            description="Test features",
            entity_key="id",
            timestamp_column="ts",
        )
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0], "ts": pd.Timestamp.now()})
        version_info = store.ingest_features(name="features", df=df, description="v1")
        assert version_info["version"] == 1
        assert version_info["row_count"] == 3

        retrieved = store.get_features("features")
        assert len(retrieved) == 3

    def test_multiple_versions(self, tmp_dir):
        store = FeatureStore(base_path=str(tmp_dir / "fs"))
        store.register_feature_group(name="grp", description="d", entity_key="id", timestamp_column="ts")

        df1 = pd.DataFrame({"id": [1], "val": [10.0], "ts": pd.Timestamp.now()})
        df2 = pd.DataFrame({"id": [1, 2], "val": [11.0, 22.0], "ts": pd.Timestamp.now()})

        store.ingest_features("grp", df1, "first version")
        store.ingest_features("grp", df2, "second version")

        history = store.get_version_history("grp")
        assert len(history) == 2

        latest = store.get_features("grp")
        assert len(latest) == 2

        first = store.get_features("grp", version=1)
        assert len(first) == 1

    def test_feature_group_info(self, tmp_dir):
        store = FeatureStore(base_path=str(tmp_dir / "fs"))
        store.register_feature_group(
            name="info_test",
            description="Info test group",
            entity_key="id",
            timestamp_column="ts",
            tags={"env": "test"},
        )
        info = store.get_feature_group_info("info_test")
        assert info["description"] == "Info test group"
        assert info["entity_key"] == "id"
        assert info["tags"]["env"] == "test"

    def test_register_duplicate_raises(self, tmp_dir):
        store = FeatureStore(base_path=str(tmp_dir / "fs"))
        store.register_feature_group(name="dup", description="d", entity_key="id", timestamp_column="ts")
        with pytest.raises(ValueError, match="already registered"):
            store.register_feature_group(name="dup", description="d2", entity_key="id", timestamp_column="ts")
