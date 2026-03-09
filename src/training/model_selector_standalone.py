"""
Model selection and comparison engine for the ML Training Pipeline.

Compares trained models by evaluation metrics, selects the best
candidate, and generates comprehensive comparison reports using
scikit-learn models for local development.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.training.distributed_trainer_standalone import TrainingResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCandidate:
    """A model candidate with comparison metadata."""
    algorithm: str
    model: Any
    primary_metric_value: float
    all_metrics: Dict[str, float]
    best_params: Dict[str, Any]
    rank: int
    training_time_seconds: float
    cv_scores: List[float]
    feature_importances: Optional[Dict[str, float]] = None


@dataclass
class SelectionReport:
    """Report summarizing model comparison and selection."""
    winner: ModelCandidate
    candidates: List[ModelCandidate]
    primary_metric: str
    threshold_met: bool
    threshold_value: float
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class StandaloneModelSelector:
    """
    Compare trained models and select the best performer.

    Ranks models by a primary evaluation metric, applies quality
    gates, and produces a selection report with full comparison.

    Attributes:
        primary_metric: Metric used for ranking.
        metric_threshold: Minimum metric value for deployment.
        higher_is_better: Whether higher metric values are better.
    """

    def __init__(
        self,
        primary_metric: str = "f1",
        metric_threshold: float = 0.75,
        higher_is_better: bool = True,
    ):
        self.primary_metric = primary_metric
        self.metric_threshold = metric_threshold
        self.higher_is_better = higher_is_better
        logger.info(
            f"StandaloneModelSelector initialized | metric={primary_metric} "
            f"| threshold={metric_threshold}"
        )

    def select_best(
        self,
        training_results: List[TrainingResult],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> SelectionReport:
        """
        Select the best model from training results.

        Args:
            training_results: List of TrainingResult from the trainer.
            X_val: Optional validation features for re-evaluation.
            y_val: Optional validation labels for re-evaluation.

        Returns:
            SelectionReport with the winning model and comparison data.

        Raises:
            ValueError: If no training results are provided.
        """
        if not training_results:
            raise ValueError("No training results to compare")

        candidates = []
        for result in training_results:
            metric_value = result.metrics.get(self.primary_metric, 0.0)

            if X_val is not None and y_val is not None:
                metric_value = self._evaluate_on_validation(
                    result.model, X_val, y_val,
                )

            all_metrics = dict(result.metrics)

            if X_val is not None and y_val is not None:
                val_metrics = self._compute_all_validation_metrics(
                    result.model, X_val, y_val,
                )
                all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            candidates.append(ModelCandidate(
                algorithm=result.algorithm,
                model=result.model,
                primary_metric_value=metric_value,
                all_metrics=all_metrics,
                best_params=result.best_params,
                rank=0,
                training_time_seconds=result.training_time_seconds,
                cv_scores=result.cross_validation_scores,
                feature_importances=result.feature_importances,
            ))

        candidates.sort(
            key=lambda c: c.primary_metric_value,
            reverse=self.higher_is_better,
        )

        for rank, candidate in enumerate(candidates, 1):
            candidate.rank = rank

        winner = candidates[0]
        threshold_met = (
            winner.primary_metric_value >= self.metric_threshold
            if self.higher_is_better
            else winner.primary_metric_value <= self.metric_threshold
        )

        report = SelectionReport(
            winner=winner,
            candidates=candidates,
            primary_metric=self.primary_metric,
            threshold_met=threshold_met,
            threshold_value=self.metric_threshold,
        )

        logger.info(
            f"Model selected | winner={winner.algorithm} "
            f"| {self.primary_metric}={winner.primary_metric_value:.4f} "
            f"| threshold_met={threshold_met}"
        )

        for candidate in candidates:
            logger.info(
                f"  Rank {candidate.rank}: {candidate.algorithm} "
                f"| {self.primary_metric}={candidate.primary_metric_value:.4f} "
                f"| time={candidate.training_time_seconds:.1f}s"
            )

        return report

    def _evaluate_on_validation(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """
        Re-evaluate model on validation set for the primary metric.

        Args:
            model: Trained scikit-learn model.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Primary metric value on validation set.
        """
        X_numeric = X_val.select_dtypes(include=[np.number]).fillna(0)
        y_pred = model.predict(X_numeric)

        try:
            if self.primary_metric == "f1":
                return float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
            elif self.primary_metric == "accuracy":
                return float(accuracy_score(y_val, y_pred))
            elif self.primary_metric == "precision":
                return float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
            elif self.primary_metric == "recall":
                return float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
            elif self.primary_metric == "rmse":
                return float(np.sqrt(mean_squared_error(y_val, y_pred)))
            elif self.primary_metric == "r2":
                return float(r2_score(y_val, y_pred))
            else:
                return float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        except Exception as e:
            logger.warning(f"Validation evaluation failed: {e}")
            return 0.0

    def _compute_all_validation_metrics(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """Compute all validation metrics."""
        X_numeric = X_val.select_dtypes(include=[np.number]).fillna(0)
        y_pred = model.predict(X_numeric)
        metrics: Dict[str, float] = {}

        try:
            metrics["accuracy"] = round(float(accuracy_score(y_val, y_pred)), 6)
            metrics["f1"] = round(float(f1_score(y_val, y_pred, average="weighted", zero_division=0)), 6)
            metrics["precision"] = round(float(precision_score(y_val, y_pred, average="weighted", zero_division=0)), 6)
            metrics["recall"] = round(float(recall_score(y_val, y_pred, average="weighted", zero_division=0)), 6)
        except Exception:
            pass

        return metrics

    def compare_with_baseline(
        self,
        winner: ModelCandidate,
        baseline_metrics: Dict[str, float],
        improvement_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Compare the winning model against a baseline.

        Args:
            winner: Selected model candidate.
            baseline_metrics: Baseline metric values.
            improvement_threshold: Minimum improvement fraction.

        Returns:
            Comparison report dictionary.
        """
        comparison: Dict[str, Any] = {
            "algorithm": winner.algorithm,
            "improvements": {},
            "regressions": {},
            "passes_threshold": False,
        }

        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in winner.all_metrics:
                candidate_value = winner.all_metrics[metric_name]
                diff = candidate_value - baseline_value
                relative_change = diff / max(abs(baseline_value), 1e-9)

                entry = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "absolute_diff": round(diff, 6),
                    "relative_change": round(relative_change, 6),
                }

                if (self.higher_is_better and diff > 0) or (
                    not self.higher_is_better and diff < 0
                ):
                    comparison["improvements"][metric_name] = entry
                else:
                    comparison["regressions"][metric_name] = entry

        primary_baseline = baseline_metrics.get(self.primary_metric, 0.0)
        primary_candidate = winner.primary_metric_value
        primary_improvement = (primary_candidate - primary_baseline) / max(
            abs(primary_baseline), 1e-9
        )
        comparison["passes_threshold"] = primary_improvement >= improvement_threshold

        logger.info(
            f"Baseline comparison | improvement={primary_improvement:.4f} "
            f"| threshold={improvement_threshold} "
            f"| passes={comparison['passes_threshold']}"
        )
        return comparison

    def generate_report_summary(self, report: SelectionReport) -> Dict[str, Any]:
        """
        Generate a human-readable summary of the selection report.

        Args:
            report: SelectionReport from select_best.

        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            "winner_algorithm": report.winner.algorithm,
            "winner_metric": round(report.winner.primary_metric_value, 4),
            "primary_metric": report.primary_metric,
            "threshold_met": report.threshold_met,
            "threshold_value": report.threshold_value,
            "total_candidates": len(report.candidates),
            "candidate_rankings": [
                {
                    "rank": c.rank,
                    "algorithm": c.algorithm,
                    report.primary_metric: round(c.primary_metric_value, 4),
                    "training_time_s": round(c.training_time_seconds, 1),
                    "cv_mean": round(float(np.mean(c.cv_scores)), 4) if c.cv_scores else None,
                    "cv_std": round(float(np.std(c.cv_scores)), 4) if c.cv_scores else None,
                }
                for c in report.candidates
            ],
            "metric_spread": round(
                abs(
                    report.candidates[0].primary_metric_value
                    - report.candidates[-1].primary_metric_value
                ),
                6,
            ) if len(report.candidates) > 1 else 0.0,
            "generated_at": report.generated_at,
        }
        return summary
