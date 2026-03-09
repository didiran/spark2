"""
Tests for StandaloneModelSelector.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import pytest

from src.training.distributed_trainer_standalone import StandaloneDistributedTrainer
from src.training.model_selector_standalone import StandaloneModelSelector


class TestStandaloneModelSelector:
    """Tests for the standalone model selector."""

    @pytest.fixture
    def trained_results(self):
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

        # Create validation set
        X_val = X.iloc[:50]
        y_val = y.iloc[:50]

        return results, X_val, y_val

    def test_select_best_returns_report(self, trained_results):
        results, X_val, y_val = trained_results
        selector = StandaloneModelSelector(
            primary_metric="f1",
            metric_threshold=0.3,
            higher_is_better=True,
        )
        report = selector.select_best(training_results=results, X_val=X_val, y_val=y_val)
        assert report.winner is not None
        assert report.winner.algorithm in ["random_forest", "logistic_regression"]

    def test_generate_report_summary(self, trained_results):
        results, X_val, y_val = trained_results
        selector = StandaloneModelSelector(
            primary_metric="f1",
            metric_threshold=0.3,
            higher_is_better=True,
        )
        report = selector.select_best(training_results=results, X_val=X_val, y_val=y_val)
        summary = selector.generate_report_summary(report)
        assert "winner_algorithm" in summary
        assert "winner_metric" in summary
        assert "threshold_met" in summary
        assert "candidates" in summary

    def test_high_threshold_marks_not_met(self, trained_results):
        results, X_val, y_val = trained_results
        selector = StandaloneModelSelector(
            primary_metric="f1",
            metric_threshold=0.999,
            higher_is_better=True,
        )
        report = selector.select_best(training_results=results, X_val=X_val, y_val=y_val)
        summary = selector.generate_report_summary(report)
        assert summary["threshold_met"] is False
