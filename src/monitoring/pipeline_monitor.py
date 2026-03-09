"""
Pipeline monitoring engine for the ML Training Pipeline.

Tracks pipeline health, stage durations, data volumes, model
performance metrics, and generates alerting conditions for
operational monitoring of ML training workflows.
"""

import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineMonitor:
    """
    Real-time pipeline monitoring and alerting system.

    Tracks stage execution metrics, data volume statistics,
    model performance trends, and generates alerts when
    configured thresholds are breached.

    Attributes:
        pipeline_name: Name of the monitored pipeline.
        metrics_history: Time-series of collected metrics.
        alerts: List of triggered alerts.
    """

    def __init__(self, pipeline_name: str = "ml-training-pipeline"):
        """
        Initialize the pipeline monitor.

        Args:
            pipeline_name: Name identifier for the pipeline.
        """
        self.pipeline_name = pipeline_name
        self.metrics_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.stage_timings: Dict[str, List[float]] = defaultdict(list)
        self.data_volumes: Dict[str, List[int]] = defaultdict(list)
        self.model_metrics: Dict[str, List[float]] = defaultdict(list)
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._start_time = datetime.utcnow()

        logger.info(f"PipelineMonitor initialized | pipeline={pipeline_name}")

    def set_threshold(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        alert_level: str = "warning",
    ) -> None:
        """
        Set alerting thresholds for a metric.

        Args:
            metric_name: Name of the metric to monitor.
            min_value: Minimum acceptable value.
            max_value: Maximum acceptable value.
            alert_level: Alert severity (info, warning, critical).
        """
        self._thresholds[metric_name] = {
            "min_value": min_value,
            "max_value": max_value,
            "alert_level": alert_level,
        }
        logger.info(
            f"Threshold set | metric={metric_name} "
            f"| min={min_value} | max={max_value} | level={alert_level}"
        )

    def record_stage_duration(
        self,
        stage_name: str,
        duration_seconds: float,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record the duration of a pipeline stage execution.

        Args:
            stage_name: Name of the executed stage.
            duration_seconds: Execution time in seconds.
            status: Stage completion status.
            metadata: Additional metadata to record.
        """
        self.stage_timings[stage_name].append(duration_seconds)

        metric_entry = {
            "type": "stage_duration",
            "stage": stage_name,
            "duration_seconds": round(duration_seconds, 3),
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.metrics_history.append(metric_entry)

        self._check_threshold(f"stage_{stage_name}_duration", duration_seconds)

        logger.info(
            f"Stage duration recorded | stage={stage_name} "
            f"| duration={duration_seconds:.2f}s | status={status}"
        )

    def record_data_volume(
        self,
        stage_name: str,
        row_count: int,
        column_count: int = 0,
        size_bytes: int = 0,
    ) -> None:
        """
        Record data volume at a pipeline stage.

        Args:
            stage_name: Name of the stage.
            row_count: Number of rows processed.
            column_count: Number of columns.
            size_bytes: Data size in bytes.
        """
        self.data_volumes[stage_name].append(row_count)

        metric_entry = {
            "type": "data_volume",
            "stage": stage_name,
            "row_count": row_count,
            "column_count": column_count,
            "size_bytes": size_bytes,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.metrics_history.append(metric_entry)

        self._check_threshold(f"data_{stage_name}_rows", row_count)

        logger.info(
            f"Data volume recorded | stage={stage_name} "
            f"| rows={row_count} | cols={column_count}"
        )

    def record_model_metric(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a model performance metric.

        Args:
            model_name: Name of the model.
            metric_name: Name of the metric.
            metric_value: Metric value.
            metadata: Additional metadata.
        """
        key = f"{model_name}_{metric_name}"
        self.model_metrics[key].append(metric_value)

        metric_entry = {
            "type": "model_metric",
            "model_name": model_name,
            "metric_name": metric_name,
            "metric_value": round(metric_value, 6),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.metrics_history.append(metric_entry)

        self._check_threshold(f"model_{key}", metric_value)

        logger.info(
            f"Model metric recorded | model={model_name} "
            f"| {metric_name}={metric_value:.4f}"
        )

    def record_pipeline_run(
        self,
        run_result: Dict[str, Any],
    ) -> None:
        """
        Record a full pipeline run result.

        Args:
            run_result: Pipeline run result dictionary.
        """
        for stage_result in run_result.get("stage_results", []):
            if stage_result.get("status") == "completed":
                self.record_stage_duration(
                    stage_result["stage"],
                    stage_result.get("duration_seconds", 0),
                    stage_result.get("status", "unknown"),
                )

        metric_entry = {
            "type": "pipeline_run",
            "run_id": run_result.get("run_id", "unknown"),
            "status": run_result.get("status", "unknown"),
            "duration_seconds": run_result.get("duration_seconds", 0),
            "total_stages": run_result.get("total_stages", 0),
            "completed_stages": run_result.get("completed_stages", 0),
            "failed_stages": run_result.get("failed_stages", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.metrics_history.append(metric_entry)

    def _check_threshold(self, metric_name: str, value: float) -> None:
        """Check a metric value against configured thresholds."""
        if metric_name not in self._thresholds:
            return

        threshold = self._thresholds[metric_name]
        min_val = threshold.get("min_value")
        max_val = threshold.get("max_value")
        level = threshold.get("alert_level", "warning")

        breached = False
        reason = ""

        if min_val is not None and value < min_val:
            breached = True
            reason = f"{metric_name}={value:.4f} below minimum {min_val}"
        elif max_val is not None and value > max_val:
            breached = True
            reason = f"{metric_name}={value:.4f} above maximum {max_val}"

        if breached:
            alert = {
                "metric": metric_name,
                "value": value,
                "level": level,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(f"ALERT [{level.upper()}] | {reason}")

    def get_stage_statistics(self, stage_name: str) -> Dict[str, Any]:
        """
        Get statistical summary for a pipeline stage.

        Args:
            stage_name: Name of the stage.

        Returns:
            Dictionary with timing statistics.
        """
        timings = self.stage_timings.get(stage_name, [])
        if not timings:
            return {"stage": stage_name, "no_data": True}

        return {
            "stage": stage_name,
            "total_runs": len(timings),
            "mean_duration_s": round(statistics.mean(timings), 3),
            "median_duration_s": round(statistics.median(timings), 3),
            "min_duration_s": round(min(timings), 3),
            "max_duration_s": round(max(timings), 3),
            "std_duration_s": round(statistics.stdev(timings), 3) if len(timings) > 1 else 0,
            "last_duration_s": round(timings[-1], 3),
        }

    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive pipeline health report.

        Returns:
            Dictionary with overall pipeline health metrics.
        """
        total_runs = sum(
            1 for m in self.metrics_history if m["type"] == "pipeline_run"
        )
        failed_runs = sum(
            1 for m in self.metrics_history
            if m["type"] == "pipeline_run" and m.get("status") == "failed"
        )
        success_rate = ((total_runs - failed_runs) / max(total_runs, 1)) * 100

        stage_stats = {
            stage: self.get_stage_statistics(stage)
            for stage in self.stage_timings
        }

        active_alerts = [
            a for a in self.alerts
            if a["level"] in ("warning", "critical")
        ]

        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "pipeline_name": self.pipeline_name,
            "status": "healthy" if not active_alerts else "degraded",
            "uptime_seconds": round(uptime, 2),
            "total_runs": total_runs,
            "failed_runs": failed_runs,
            "success_rate_pct": round(success_rate, 2),
            "active_alerts": len(active_alerts),
            "total_alerts": len(self.alerts),
            "stage_statistics": stage_stats,
            "recent_alerts": active_alerts[-5:],
            "total_metrics_collected": len(self.metrics_history),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_alerts(
        self,
        level: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve alerts, optionally filtered by level.

        Args:
            level: Filter by alert level (info, warning, critical).
            limit: Maximum number of alerts to return.

        Returns:
            List of alert dictionaries.
        """
        alerts = self.alerts
        if level:
            alerts = [a for a in alerts if a["level"] == level]
        return alerts[-limit:]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary with metric summaries by type.
        """
        summary = {
            "total_metrics": len(self.metrics_history),
            "stage_timings": {
                stage: {
                    "count": len(timings),
                    "avg_seconds": round(statistics.mean(timings), 3) if timings else 0,
                }
                for stage, timings in self.stage_timings.items()
            },
            "data_volumes": {
                stage: {
                    "count": len(volumes),
                    "avg_rows": int(statistics.mean(volumes)) if volumes else 0,
                    "total_rows": sum(volumes),
                }
                for stage, volumes in self.data_volumes.items()
            },
            "model_metrics": {
                key: {
                    "count": len(values),
                    "latest": round(values[-1], 6) if values else None,
                    "mean": round(statistics.mean(values), 6) if values else None,
                }
                for key, values in self.model_metrics.items()
            },
        }
        return summary

    def clear(self) -> None:
        """Reset all monitoring data."""
        self.metrics_history.clear()
        self.alerts.clear()
        self.stage_timings.clear()
        self.data_volumes.clear()
        self.model_metrics.clear()
        self._thresholds.clear()
        self._start_time = datetime.utcnow()
        logger.info("Pipeline monitor cleared")
