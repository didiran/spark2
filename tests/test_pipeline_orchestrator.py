"""
Tests for MLPipelineOrchestrator.

Author: Gabriel Demetrios Lafis
"""

import pytest

from src.orchestration.pipeline import MLPipelineOrchestrator


class TestMLPipelineOrchestrator:
    """Tests for the pipeline orchestrator."""

    def test_add_and_run_single_stage(self):
        orch = MLPipelineOrchestrator(pipeline_name="test-pipeline")
        orch.add_stage("stage_a", lambda ctx: 42, description="Returns 42")
        result = orch.run()
        assert result["status"] == "completed"
        assert result["completed_stages"] == 1

    def test_run_multiple_stages_in_order(self):
        orch = MLPipelineOrchestrator(pipeline_name="test-pipeline")
        execution_order = []

        def stage_1(ctx):
            execution_order.append("s1")
            return 1

        def stage_2(ctx):
            execution_order.append("s2")
            return ctx["result_s1"] + 1

        orch.add_stage("s1", stage_1, "First stage")
        orch.add_stage("s2", stage_2, "Second stage", dependencies=["s1"])
        result = orch.run()

        assert result["status"] == "completed"
        assert execution_order == ["s1", "s2"]
        assert result["completed_stages"] == 2

    def test_stage_context_passing(self):
        orch = MLPipelineOrchestrator(pipeline_name="ctx-test")

        def produce(ctx):
            return {"value": 100}

        def consume(ctx):
            return ctx["result_produce"]["value"] * 2

        orch.add_stage("produce", produce, "Producer")
        orch.add_stage("consume", consume, "Consumer", dependencies=["produce"])
        result = orch.run()
        assert result["status"] == "completed"

    def test_failing_stage_stops_pipeline(self):
        orch = MLPipelineOrchestrator(pipeline_name="fail-test", fail_fast=True)

        def good_stage(ctx):
            return "ok"

        def bad_stage(ctx):
            raise ValueError("Intentional failure")

        def after_bad(ctx):
            return "should not run"

        orch.add_stage("good", good_stage, "Good stage")
        orch.add_stage("bad", bad_stage, "Bad stage", dependencies=["good"])
        orch.add_stage("after", after_bad, "After bad", dependencies=["bad"])

        result = orch.run()
        assert result["status"] == "failed"
        assert result["completed_stages"] < result["total_stages"]

    def test_pipeline_duration_tracked(self):
        orch = MLPipelineOrchestrator(pipeline_name="duration-test")
        orch.add_stage("fast", lambda ctx: True, "Fast stage")
        result = orch.run()
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0.0

    def test_pipeline_summary(self):
        orch = MLPipelineOrchestrator(pipeline_name="summary-test")
        orch.add_stage("a", lambda ctx: 1, "Stage A")
        orch.add_stage("b", lambda ctx: 2, "Stage B", dependencies=["a"])
        orch.run()
        summary = orch.get_summary()
        assert summary["pipeline_name"] == "summary-test"
        assert summary["total_stages"] == 2
