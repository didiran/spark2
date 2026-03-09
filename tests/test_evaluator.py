"""
Tests for StandalonePipelineEvaluator.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.evaluator_standalone import StandalonePipelineEvaluator


class TestStandalonePipelineEvaluator:
    """Tests for the standalone pipeline evaluator."""

    @pytest.fixture
    def trained_model_and_data(self):
        np.random.seed(42)
        n = 200
        X_train = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, n),
                "f2": np.random.normal(5, 2, n),
            }
        )
        y_train = pd.Series((X_train["f1"] + X_train["f2"] > 5).astype(int))

        X_test = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 60),
                "f2": np.random.normal(5, 2, 60),
            }
        )
        y_test = pd.Series((X_test["f1"] + X_test["f2"] > 5).astype(int))

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        return model, X_train, y_train, X_test, y_test

    def test_evaluate_returns_result(self, trained_model_and_data):
        model, X_train, y_train, X_test, y_test = trained_model_and_data
        evaluator = StandalonePipelineEvaluator(task_type="classification")
        result = evaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name="rf_test",
        )
        assert result.model_name == "rf_test"
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics

    def test_quality_gates_pass(self, trained_model_and_data):
        model, X_train, y_train, X_test, y_test = trained_model_and_data
        evaluator = StandalonePipelineEvaluator(task_type="classification")
        evaluator.add_quality_gate("min_accuracy", "accuracy", 0.5)
        result = evaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name="rf_gates",
        )
        assert result.quality_gates_passed is True

    def test_quality_gates_fail(self, trained_model_and_data):
        model, X_train, y_train, X_test, y_test = trained_model_and_data
        evaluator = StandalonePipelineEvaluator(task_type="classification")
        evaluator.add_quality_gate("impossible_accuracy", "accuracy", 1.0)
        result = evaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name="rf_fail",
        )
        assert result.quality_gates_passed is False

    def test_cross_validation_scores(self, trained_model_and_data):
        model, X_train, y_train, X_test, y_test = trained_model_and_data
        evaluator = StandalonePipelineEvaluator(task_type="classification")
        result = evaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name="rf_cv",
            cv_folds=3,
            X_train=X_train,
            y_train=y_train,
        )
        assert result.cross_val_scores is not None
        assert len(result.cross_val_scores) == 3

    def test_confusion_matrix_present(self, trained_model_and_data):
        model, X_train, y_train, X_test, y_test = trained_model_and_data
        evaluator = StandalonePipelineEvaluator(task_type="classification")
        result = evaluator.evaluate(
            model=model, X_test=X_test, y_test=y_test, model_name="rf_cm"
        )
        assert result.confusion_matrix_data is not None
