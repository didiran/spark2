"""
Model selection and comparison engine.

Compares trained models by evaluation metrics, selects the best
candidate, and manages model registration for deployment.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from src.training.distributed_trainer import TrainingResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCandidate:
    """A model candidate with comparison metadata."""
    algorithm: str
    model: PipelineModel
    primary_metric_value: float
    all_metrics: Dict[str, float]
    best_params: Dict[str, Any]
    rank: int
    training_time_seconds: float
    cv_scores: List[float]


@dataclass
class SelectionReport:
    """Report summarizing model comparison and selection."""
    winner: ModelCandidate
    candidates: List[ModelCandidate]
    primary_metric: str
    threshold_met: bool
    threshold_value: float


class ModelSelector:
    """
    Compare trained models and select the best performer.

    Ranks models by a primary evaluation metric, applies quality
    gates, and produces a selection report with full comparison.

    Attributes:
        primary_metric: Metric used for ranking (e.g., "f1", "accuracy").
        metric_threshold: Minimum metric value for deployment readiness.
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
            f"ModelSelector initialized | metric={primary_metric} "
            f"| threshold={metric_threshold}"
        )

    def select_best(
        self,
        training_results: List[TrainingResult],
        validation_df: Optional[DataFrame] = None,
    ) -> SelectionReport:
        """
        Select the best model from training results.

        Args:
            training_results: List of TrainingResult from DistributedTrainer.
            validation_df: Optional validation DataFrame for re-evaluation.

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

            if validation_df is not None:
                metric_value = self._evaluate_on_validation(
                    result.model, validation_df, self.primary_metric
                )

            candidates.append(ModelCandidate(
                algorithm=result.algorithm,
                model=result.model,
                primary_metric_value=metric_value,
                all_metrics=result.metrics,
                best_params=result.best_params,
                rank=0,
                training_time_seconds=result.training_time_seconds,
                cv_scores=result.cross_validation_scores,
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
        model: PipelineModel,
        validation_df: DataFrame,
        metric_name: str,
    ) -> float:
        """
        Re-evaluate a model on the validation set.

        Args:
            model: Fitted PipelineModel.
            validation_df: Validation DataFrame.
            metric_name: Metric to compute.

        Returns:
            Metric value on the validation set.
        """
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator

        predictions = model.transform(validation_df)
        evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            metricName=metric_name,
        )

        try:
            return round(evaluator.evaluate(predictions), 6)
        except Exception as e:
            logger.warning(f"Validation evaluation failed: {e}")
            return 0.0

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
            baseline_metrics: Dictionary of baseline metric values.
            improvement_threshold: Minimum improvement fraction required.

        Returns:
            Comparison report with improvement details.
        """
        comparison = {
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
            "winner_metric": report.winner.primary_metric_value,
            "primary_metric": report.primary_metric,
            "threshold_met": report.threshold_met,
            "total_candidates": len(report.candidates),
            "candidate_rankings": [
                {
                    "rank": c.rank,
                    "algorithm": c.algorithm,
                    f"{report.primary_metric}": c.primary_metric_value,
                    "training_time_s": round(c.training_time_seconds, 1),
                }
                for c in report.candidates
            ],
            "metric_spread": round(
                report.candidates[0].primary_metric_value
                - report.candidates[-1].primary_metric_value,
                6,
            ) if len(report.candidates) > 1 else 0.0,
        }

        return summary
