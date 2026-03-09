"""
Tests for StandaloneDistributedTrainer.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import pytest

from src.training.distributed_trainer_standalone import StandaloneDistributedTrainer


class TestStandaloneDistributedTrainer:
    """Tests for the standalone distributed trainer."""

    @pytest.fixture
    def classification_data(self):
        np.random.seed(42)
        n = 300
        X = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, n),
                "f2": np.random.normal(5, 2, n),
                "f3": np.random.uniform(0, 10, n),
            }
        )
        y = pd.Series((X["f1"] + X["f2"] > 5).astype(int), name="target")
        return X, y

    def test_train_returns_results(self, classification_data):
        X, y = classification_data
        trainer = StandaloneDistributedTrainer(
            algorithms=["random_forest", "logistic_regression"],
            target_column="target",
            primary_metric="f1",
            cv_folds=2,
            max_workers=2,
            seed=42,
            task_type="classification",
        )
        results = trainer.train(X, y, feature_columns=list(X.columns), use_grid_search=False)
        assert len(results) == 2
        for r in results:
            assert r.model is not None
            assert "f1" in r.metrics
            assert "accuracy" in r.metrics

    def test_split_data_ratios(self, classification_data):
        X, y = classification_data
        df = X.copy()
        df["target"] = y

        trainer = StandaloneDistributedTrainer(
            algorithms=["random_forest"],
            target_column="target",
            primary_metric="f1",
            seed=42,
            task_type="classification",
        )

        X_train, y_train, X_val, y_val, X_test, y_test = trainer.split_data(
            df, target_column="target", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(df)
        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)

    def test_training_result_has_feature_importance(self, classification_data):
        X, y = classification_data
        trainer = StandaloneDistributedTrainer(
            algorithms=["random_forest"],
            target_column="target",
            primary_metric="f1",
            cv_folds=2,
            seed=42,
            task_type="classification",
        )
        results = trainer.train(X, y, feature_columns=list(X.columns), use_grid_search=False)
        assert results[0].feature_importance is not None
        assert len(results[0].feature_importance) == 3

    def test_metrics_are_valid_range(self, classification_data):
        X, y = classification_data
        trainer = StandaloneDistributedTrainer(
            algorithms=["logistic_regression"],
            target_column="target",
            primary_metric="accuracy",
            cv_folds=2,
            seed=42,
            task_type="classification",
        )
        results = trainer.train(X, y, feature_columns=list(X.columns), use_grid_search=False)
        metrics = results[0].metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
