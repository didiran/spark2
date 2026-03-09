"""
Tests for PipelineMonitor.

Author: Gabriel Demetrios Lafis
"""

import pytest

from src.monitoring.pipeline_monitor import PipelineMonitor


class TestPipelineMonitor:
    """Tests for the pipeline monitoring module."""

    def test_record_stage_duration(self):
        monitor = PipelineMonitor(pipeline_name="test")
        monitor.record_stage_duration("ingestion", 12.5)
        monitor.record_stage_duration("ingestion", 14.0)
        stats = monitor.get_stage_statistics("ingestion")
        assert stats["count"] == 2
        assert stats["mean"] == pytest.approx(13.25)

    def test_record_data_volume(self):
        monitor = PipelineMonitor(pipeline_name="test")
        monitor.record_data_volume("processing", rows=5000, columns=25)
        summary = monitor.get_metrics_summary()
        assert "data_volumes" in summary

    def test_record_model_metric(self):
        monitor = PipelineMonitor(pipeline_name="test")
        monitor.record_model_metric("random_forest", "f1", 0.85)
        monitor.record_model_metric("logistic_regression", "f1", 0.78)
        summary = monitor.get_metrics_summary()
        assert "model_metrics" in summary

    def test_health_report_structure(self):
        monitor = PipelineMonitor(pipeline_name="health-test")
        monitor.record_pipeline_run({"status": "completed", "duration_seconds": 10.0})
        health = monitor.get_health_report()
        assert "status" in health
        assert "success_rate_pct" in health
        assert "total_runs" in health

    def test_threshold_alerts(self):
        monitor = PipelineMonitor(pipeline_name="alert-test")
        monitor.set_threshold("stage_duration_ingestion", max_value=5.0)
        monitor.record_stage_duration("ingestion", 10.0)
        alerts = monitor.get_alerts()
        assert len(alerts) >= 1

    def test_health_report_with_multiple_runs(self):
        monitor = PipelineMonitor(pipeline_name="multi-run")
        monitor.record_pipeline_run({"status": "completed", "duration_seconds": 5.0})
        monitor.record_pipeline_run({"status": "completed", "duration_seconds": 6.0})
        monitor.record_pipeline_run({"status": "failed", "duration_seconds": 2.0})
        health = monitor.get_health_report()
        assert health["total_runs"] == 3
        assert health["success_rate_pct"] == pytest.approx(66.67, abs=1.0)
