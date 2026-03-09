"""
Distributed model evaluation engine for the ML Training Pipeline.

Performs scalable model evaluation on Spark with comprehensive
metrics computation, stratified analysis, quality gates, and
structured reporting for deployment decision-making.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityGate:
    """A quality gate definition for model deployment readiness."""
    name: str
    metric: str
    threshold: float
    operator: str = ">="  # ">=", "<=", ">", "<", "=="
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
    stratified_metrics: Optional[Dict[str, Dict[str, float]]] = None


class PipelineEvaluator:
    """
    Distributed model evaluator with quality gates.

    Computes classification and regression metrics at scale,
    performs stratified analysis, and enforces deployment quality
    gates before models proceed to production.

    Attributes:
        spark: Active SparkSession.
        quality_gates: List of quality gates to enforce.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.quality_gates: List[QualityGate] = []
        logger.info("PipelineEvaluator initialized")

    def add_quality_gate(
        self,
        name: str,
        metric: str,
        threshold: float,
        operator: str = ">=",
        description: str = "",
    ) -> "PipelineEvaluator":
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
        model: PipelineModel,
        test_df: DataFrame,
        model_name: str = "model",
        label_col: str = "label",
        task_type: str = "classification",
        stratify_columns: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation suite on a model.

        Args:
            model: Fitted PipelineModel to evaluate.
            test_df: Test DataFrame for evaluation.
            model_name: Identifier for the model.
            label_col: Label column name.
            task_type: "classification" or "regression".
            stratify_columns: Columns for stratified metric analysis.

        Returns:
            EvaluationResult with metrics, quality gates, and distribution.
        """
        predictions = model.transform(test_df)
        dataset_size = predictions.count()

        logger.info(
            f"Evaluating model | name={model_name} | task={task_type} "
            f"| test_size={dataset_size}"
        )

        if task_type == "classification":
            metrics = self._compute_classification_metrics(predictions, label_col)
        else:
            metrics = self._compute_regression_metrics(predictions, label_col)

        gate_details = self._check_quality_gates(metrics)
        all_passed = all(g["passed"] for g in gate_details)

        distribution = self._compute_prediction_distribution(
            predictions, label_col, task_type
        )

        stratified = None
        if stratify_columns:
            stratified = self._compute_stratified_metrics(
                predictions, label_col, stratify_columns, task_type
            )

        result = EvaluationResult(
            model_name=model_name,
            task_type=task_type,
            metrics=metrics,
            quality_gates_passed=all_passed,
            quality_gate_details=gate_details,
            prediction_distribution=distribution,
            dataset_size=dataset_size,
            stratified_metrics=stratified,
        )

        logger.info(
            f"Evaluation complete | name={model_name} "
            f"| gates_passed={all_passed} | metrics={metrics}"
        )

        return result

    def _compute_classification_metrics(
        self,
        predictions: DataFrame,
        label_col: str,
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        metrics = {}

        mc_evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
        )

        for metric_name in [
            "f1", "accuracy", "weightedPrecision", "weightedRecall",
            "logLoss", "hammingLoss",
        ]:
            try:
                mc_evaluator.setMetricName(metric_name)
                metrics[metric_name] = round(mc_evaluator.evaluate(predictions), 6)
            except Exception:
                pass

        try:
            bc_evaluator = BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="rawPrediction",
            )
            for metric_name in ["areaUnderROC", "areaUnderPR"]:
                bc_evaluator.setMetricName(metric_name)
                metrics[metric_name] = round(bc_evaluator.evaluate(predictions), 6)
        except Exception:
            pass

        try:
            confusion = self._compute_confusion_matrix(predictions, label_col)
            metrics["confusion_matrix"] = confusion
        except Exception:
            pass

        return metrics

    def _compute_regression_metrics(
        self,
        predictions: DataFrame,
        label_col: str,
    ) -> Dict[str, float]:
        """Compute comprehensive regression metrics."""
        metrics = {}

        reg_evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
        )

        for metric_name in ["rmse", "mse", "mae", "r2", "var"]:
            try:
                reg_evaluator.setMetricName(metric_name)
                metrics[metric_name] = round(reg_evaluator.evaluate(predictions), 6)
            except Exception:
                pass

        try:
            error_col = F.col("prediction") - F.col(label_col)
            error_stats = predictions.select(
                F.mean(error_col).alias("mean_error"),
                F.stddev(error_col).alias("std_error"),
                F.percentile_approx(F.abs(error_col), 0.5).alias("median_abs_error"),
                F.percentile_approx(F.abs(error_col), 0.95).alias("p95_abs_error"),
            ).first()

            metrics["mean_error"] = round(float(error_stats["mean_error"] or 0), 6)
            metrics["std_error"] = round(float(error_stats["std_error"] or 0), 6)
            metrics["median_abs_error"] = round(float(error_stats["median_abs_error"] or 0), 6)
            metrics["p95_abs_error"] = round(float(error_stats["p95_abs_error"] or 0), 6)
        except Exception:
            pass

        return metrics

    def _compute_confusion_matrix(
        self,
        predictions: DataFrame,
        label_col: str,
    ) -> Dict[str, Any]:
        """Compute confusion matrix for classification tasks."""
        cm_df = (
            predictions
            .groupBy(label_col, "prediction")
            .count()
            .orderBy(label_col, "prediction")
        )

        rows = cm_df.collect()
        labels = sorted(set(row[label_col] for row in rows))

        matrix = {}
        for row in rows:
            actual = str(int(row[label_col]))
            predicted = str(int(row["prediction"]))
            if actual not in matrix:
                matrix[actual] = {}
            matrix[actual][predicted] = row["count"]

        return {"labels": [str(int(l)) for l in labels], "matrix": matrix}

    def _compute_prediction_distribution(
        self,
        predictions: DataFrame,
        label_col: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Compute distribution statistics for predictions."""
        if task_type == "classification":
            pred_counts = (
                predictions
                .groupBy("prediction")
                .count()
                .orderBy("prediction")
                .collect()
            )
            label_counts = (
                predictions
                .groupBy(label_col)
                .count()
                .orderBy(label_col)
                .collect()
            )
            return {
                "prediction_counts": {
                    str(int(r["prediction"])): r["count"] for r in pred_counts
                },
                "label_counts": {
                    str(int(r[label_col])): r["count"] for r in label_counts
                },
            }
        else:
            stats = predictions.select(
                F.mean("prediction").alias("mean"),
                F.stddev("prediction").alias("stddev"),
                F.min("prediction").alias("min"),
                F.max("prediction").alias("max"),
            ).first()
            return {
                "mean": round(float(stats["mean"] or 0), 4),
                "stddev": round(float(stats["stddev"] or 0), 4),
                "min": round(float(stats["min"] or 0), 4),
                "max": round(float(stats["max"] or 0), 4),
            }

    def _compute_stratified_metrics(
        self,
        predictions: DataFrame,
        label_col: str,
        stratify_columns: List[str],
        task_type: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics stratified by specified columns."""
        stratified = {}

        for strat_col in stratify_columns:
            if strat_col not in predictions.columns:
                continue

            strat_values = [
                row[strat_col]
                for row in predictions.select(strat_col).distinct().collect()
            ]

            for value in strat_values:
                subset = predictions.filter(F.col(strat_col) == value)
                if subset.count() == 0:
                    continue

                key = f"{strat_col}={value}"
                if task_type == "classification":
                    evaluator = MulticlassClassificationEvaluator(
                        labelCol=label_col,
                        predictionCol="prediction",
                        metricName="f1",
                    )
                    stratified[key] = {
                        "f1": round(evaluator.evaluate(subset), 6),
                        "count": subset.count(),
                    }
                else:
                    evaluator = RegressionEvaluator(
                        labelCol=label_col,
                        predictionCol="prediction",
                        metricName="rmse",
                    )
                    stratified[key] = {
                        "rmse": round(evaluator.evaluate(subset), 6),
                        "count": subset.count(),
                    }

        return stratified

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
