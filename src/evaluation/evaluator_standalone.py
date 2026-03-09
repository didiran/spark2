"""
Standalone model evaluation engine for the ML Training Pipeline.

Performs comprehensive model evaluation with cross-validation,
stratified analysis, quality gates, and structured reporting
using scikit-learn for local development and testing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityGate:
    """A quality gate definition for model deployment readiness."""
    name: str
    metric: str
    threshold: float
    operator: str = ">="
    description: str = ""


@dataclass
class EvaluationResult:
    """Comprehensive model evaluation result."""
    model_name: str
    task_type: str
    metrics: Dict[str, float]
    quality_gates_passed: bool
    quality_gate_details: List[Dict[str, Any]]
    prediction_distribution: Dict[str, Any]
    evaluation_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    dataset_size: int = 0
    confusion_matrix_data: Optional[Dict[str, Any]] = None
    cross_val_scores: Optional[Dict[str, List[float]]] = None
    classification_report_data: Optional[Dict[str, Any]] = None


class StandalonePipelineEvaluator:
    """
    Comprehensive model evaluator with quality gates.

    Computes classification and regression metrics, performs
    cross-validation, stratified analysis, and enforces deployment
    quality gates before models proceed to production.

    Attributes:
        quality_gates: List of quality gates to enforce.
        task_type: Type of ML task (classification or regression).
    """

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.quality_gates: List[QualityGate] = []
        logger.info(f"StandalonePipelineEvaluator initialized | task_type={task_type}")

    def add_quality_gate(
        self,
        name: str,
        metric: str,
        threshold: float,
        operator: str = ">=",
        description: str = "",
    ) -> "StandalonePipelineEvaluator":
        """Add a quality gate (fluent interface)."""
        gate = QualityGate(
            name=name,
            metric=metric,
            threshold=threshold,
            operator=operator,
            description=description,
        )
        self.quality_gates.append(gate)
        logger.info(f"Quality gate added | {name}: {metric} {operator} {threshold}")
        return self

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        cv_folds: int = 5,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation suite on a model.

        Args:
            model: Trained scikit-learn model.
            X_test: Test features.
            y_test: Test labels.
            model_name: Identifier for the model.
            cv_folds: Number of cross-validation folds.
            X_train: Training features for cross-validation.
            y_train: Training labels for cross-validation.

        Returns:
            EvaluationResult with metrics, quality gates, and distributions.
        """
        X_numeric = X_test.select_dtypes(include=[np.number]).fillna(0)
        y_pred = model.predict(X_numeric)
        dataset_size = len(X_test)

        logger.info(
            f"Evaluating model | name={model_name} | task={self.task_type} "
            f"| test_size={dataset_size}"
        )

        if self.task_type == "classification":
            metrics = self._compute_classification_metrics(y_test, y_pred, model, X_numeric)
        else:
            metrics = self._compute_regression_metrics(y_test, y_pred)

        gate_details = self._check_quality_gates(metrics)
        all_passed = all(g["passed"] for g in gate_details) if gate_details else True

        distribution = self._compute_prediction_distribution(y_test, y_pred)

        confusion_data = None
        report_data = None
        if self.task_type == "classification":
            confusion_data = self._compute_confusion_matrix(y_test, y_pred)
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_data = {
                    k: v for k, v in report.items()
                    if isinstance(v, dict)
                }
            except Exception:
                pass

        cv_scores = None
        if X_train is not None and y_train is not None:
            cv_scores = self._perform_cross_validation(
                model, X_train, y_train, cv_folds,
            )

        result = EvaluationResult(
            model_name=model_name,
            task_type=self.task_type,
            metrics=metrics,
            quality_gates_passed=all_passed,
            quality_gate_details=gate_details,
            prediction_distribution=distribution,
            dataset_size=dataset_size,
            confusion_matrix_data=confusion_data,
            cross_val_scores=cv_scores,
            classification_report_data=report_data,
        )

        logger.info(
            f"Evaluation complete | name={model_name} "
            f"| gates_passed={all_passed} | metrics={metrics}"
        )
        return result

    def _compute_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: Any,
        X: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        metrics: Dict[str, float] = {}

        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)
        metrics["f1"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
        metrics["precision"] = round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6)

        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)
                if y_proba.shape[1] == 2:
                    metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 6)
                else:
                    metrics["auc_roc"] = round(
                        float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")),
                        6,
                    )
        except Exception:
            pass

        return metrics

    def _compute_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute comprehensive regression metrics."""
        metrics: Dict[str, float] = {}

        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6)
        metrics["mse"] = round(float(mean_squared_error(y_true, y_pred)), 6)
        metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 6)
        metrics["r2"] = round(float(r2_score(y_true, y_pred)), 6)

        errors = y_pred - y_true.values
        metrics["mean_error"] = round(float(np.mean(errors)), 6)
        metrics["std_error"] = round(float(np.std(errors)), 6)
        metrics["median_abs_error"] = round(float(np.median(np.abs(errors))), 6)
        metrics["p95_abs_error"] = round(float(np.percentile(np.abs(errors), 95)), 6)

        return metrics

    def _compute_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute confusion matrix for classification tasks."""
        labels = sorted(set(y_true.unique()) | set(np.unique(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            "labels": [str(l) for l in labels],
            "matrix": cm.tolist(),
            "true_positives": int(cm[1, 1]) if cm.shape[0] > 1 else 0,
            "true_negatives": int(cm[0, 0]) if cm.shape[0] > 1 else 0,
            "false_positives": int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            "false_negatives": int(cm[1, 0]) if cm.shape[0] > 1 else 0,
        }

    def _compute_prediction_distribution(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute distribution statistics for predictions."""
        if self.task_type == "classification":
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            label_counts = y_true.value_counts().sort_index()
            return {
                "prediction_counts": {str(k): int(v) for k, v in pred_counts.items()},
                "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            }
        else:
            return {
                "pred_mean": round(float(np.mean(y_pred)), 4),
                "pred_std": round(float(np.std(y_pred)), 4),
                "pred_min": round(float(np.min(y_pred)), 4),
                "pred_max": round(float(np.max(y_pred)), 4),
                "true_mean": round(float(y_true.mean()), 4),
                "true_std": round(float(y_true.std()), 4),
            }

    def _perform_cross_validation(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation with multiple metrics."""
        X_numeric = X_train.select_dtypes(include=[np.number]).fillna(0)

        cv_results: Dict[str, List[float]] = {}

        if self.task_type == "classification":
            scoring_metrics = {
                "f1": "f1_weighted",
                "accuracy": "accuracy",
                "precision": "precision_weighted",
            }
        else:
            scoring_metrics = {
                "neg_rmse": "neg_root_mean_squared_error",
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
            }

        folds = min(cv_folds, len(y_train))

        for name, scoring in scoring_metrics.items():
            try:
                scores = cross_val_score(
                    model.__class__(**model.get_params()),
                    X_numeric,
                    y_train,
                    cv=folds,
                    scoring=scoring,
                )
                cv_results[name] = [round(float(s), 6) for s in scores]
            except Exception as e:
                logger.warning(f"Cross-validation failed for {name}: {e}")

        return cv_results

    def _check_quality_gates(
        self,
        metrics: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Evaluate all quality gates against computed metrics."""
        results = []

        ops = {
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            "==": lambda a, b: abs(a - b) < 1e-9,
        }

        for gate in self.quality_gates:
            actual_value = metrics.get(gate.metric)
            if actual_value is None:
                results.append({
                    "gate": gate.name,
                    "metric": gate.metric,
                    "threshold": gate.threshold,
                    "actual": None,
                    "passed": False,
                    "reason": f"Metric '{gate.metric}' not computed",
                })
                continue

            op_fn = ops.get(gate.operator, ops[">="])
            passed = op_fn(actual_value, gate.threshold)

            results.append({
                "gate": gate.name,
                "metric": gate.metric,
                "threshold": gate.threshold,
                "actual": actual_value,
                "operator": gate.operator,
                "passed": passed,
            })

            status = "PASS" if passed else "FAIL"
            logger.info(
                f"Quality gate [{status}] | {gate.name}: "
                f"{gate.metric}={actual_value} {gate.operator} {gate.threshold}"
            )

        return results
