"""
Pipeline orchestrator for the ML Training Pipeline.

Coordinates the full ML pipeline flow: ingest, validate, process,
feature engineering, training, evaluation, and model selection.
Built on top of the DAG-based PipelineOrchestrator for task management.
"""

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPipelineOrchestrator:
    """
    End-to-end ML pipeline orchestrator.

    Coordinates the full pipeline flow from data ingestion through
    model selection, with configurable stage parameters, error
    handling, and comprehensive reporting.

    Attributes:
        pipeline_name: Name identifier for the pipeline.
        stages: Ordered list of pipeline stage definitions.
        results: Dictionary of stage results from the last run.
        run_history: History of all pipeline runs.
    """

    def __init__(
        self,
        pipeline_name: str = "ml-training-pipeline",
        fail_fast: bool = True,
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            pipeline_name: Name identifier for the pipeline.
            fail_fast: Stop pipeline on first stage failure.
        """
        self.pipeline_name = pipeline_name
        self.fail_fast = fail_fast
        self.stages: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self.run_history: List[Dict[str, Any]] = []
        self._stage_registry: Dict[str, Callable] = {}
        logger.info(f"MLPipelineOrchestrator initialized | name={pipeline_name}")

    def add_stage(
        self,
        name: str,
        func: Callable,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
    ) -> "MLPipelineOrchestrator":
        """
        Register a pipeline stage.

        Args:
            name: Unique stage identifier.
            func: Callable to execute for this stage.
            description: Human-readable description.
            dependencies: List of stage names this stage depends on.
            retry_count: Number of retries on failure.
            retry_delay: Delay between retries in seconds.

        Returns:
            Self for method chaining.
        """
        stage = {
            "name": name,
            "func": func,
            "description": description,
            "dependencies": dependencies or [],
            "retry_count": retry_count,
            "retry_delay": retry_delay,
        }
        self.stages.append(stage)
        self._stage_registry[name] = func
        logger.info(f"Stage registered | name={name} | deps={stage['dependencies']}")
        return self

    def run(
        self,
        context: Optional[Dict[str, Any]] = None,
        skip_stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline in registered order.

        Args:
            context: Shared context dictionary passed between stages.
            skip_stages: Stage names to skip.

        Returns:
            Pipeline run result dictionary.
        """
        run_id = f"{self.pipeline_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        ctx = context or {}
        skip_set = set(skip_stages or [])
        stage_results = []
        failed = False

        run_start = time.time()

        logger.info(
            f"Pipeline started | run_id={run_id} | stages={len(self.stages)}"
        )

        for stage in self.stages:
            stage_name = stage["name"]

            if stage_name in skip_set:
                stage_results.append({
                    "stage": stage_name,
                    "status": "skipped",
                    "duration_seconds": 0,
                })
                logger.info(f"Stage skipped | name={stage_name}")
                continue

            if failed and self.fail_fast:
                stage_results.append({
                    "stage": stage_name,
                    "status": "skipped",
                    "reason": "previous stage failed",
                    "duration_seconds": 0,
                })
                continue

            stage_result = self._execute_stage(stage, ctx)
            stage_results.append(stage_result)

            if stage_result["status"] == "completed":
                self.results[stage_name] = stage_result.get("result")
                ctx[f"result_{stage_name}"] = stage_result.get("result")
            else:
                failed = True
                if self.fail_fast:
                    logger.error(
                        f"Pipeline stopping (fail_fast) | failed_stage={stage_name}"
                    )

        run_duration = time.time() - run_start

        run_result = {
            "run_id": run_id,
            "pipeline_name": self.pipeline_name,
            "status": "failed" if failed else "completed",
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": round(run_duration, 2),
            "total_stages": len(self.stages),
            "completed_stages": sum(
                1 for r in stage_results if r["status"] == "completed"
            ),
            "failed_stages": sum(
                1 for r in stage_results if r["status"] == "failed"
            ),
            "skipped_stages": sum(
                1 for r in stage_results if r["status"] == "skipped"
            ),
            "stage_results": stage_results,
        }

        self.run_history.append(run_result)

        logger.info(
            f"Pipeline finished | run_id={run_id} | status={run_result['status']} "
            f"| duration={run_duration:.2f}s | completed={run_result['completed_stages']}/{run_result['total_stages']}"
        )

        return run_result

    def _execute_stage(
        self,
        stage: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a single pipeline stage with retry logic.

        Args:
            stage: Stage definition dictionary.
            context: Shared pipeline context.

        Returns:
            Stage execution result dictionary.
        """
        stage_name = stage["name"]
        max_attempts = stage["retry_count"] + 1

        for attempt in range(1, max_attempts + 1):
            start_time = time.time()

            logger.info(
                f"Stage executing | name={stage_name} "
                f"| attempt={attempt}/{max_attempts}"
            )

            try:
                result = stage["func"](context)
                duration = time.time() - start_time

                return {
                    "stage": stage_name,
                    "status": "completed",
                    "duration_seconds": round(duration, 2),
                    "attempt": attempt,
                    "result": result,
                }

            except Exception as exc:
                duration = time.time() - start_time
                error_msg = f"{type(exc).__name__}: {exc}"

                logger.error(
                    f"Stage failed | name={stage_name} | attempt={attempt} "
                    f"| error={error_msg}"
                )

                if attempt < max_attempts:
                    logger.info(
                        f"Retrying stage | name={stage_name} "
                        f"| delay={stage['retry_delay']}s"
                    )
                    time.sleep(stage["retry_delay"])
                else:
                    return {
                        "stage": stage_name,
                        "status": "failed",
                        "duration_seconds": round(duration, 2),
                        "attempt": attempt,
                        "error": error_msg,
                    }

        return {
            "stage": stage_name,
            "status": "failed",
            "error": "Max retries exhausted",
            "duration_seconds": 0,
        }

    def get_stage_result(self, stage_name: str) -> Any:
        """Retrieve the result of a completed stage."""
        return self.results.get(stage_name)

    def get_run_history(self) -> List[Dict[str, Any]]:
        """Return the history of all pipeline runs."""
        return list(self.run_history)

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline configuration.

        Returns:
            Summary dictionary with stage information.
        """
        return {
            "pipeline_name": self.pipeline_name,
            "total_stages": len(self.stages),
            "stages": [
                {
                    "name": s["name"],
                    "description": s["description"],
                    "dependencies": s["dependencies"],
                    "retry_count": s["retry_count"],
                }
                for s in self.stages
            ],
            "total_runs": len(self.run_history),
            "fail_fast": self.fail_fast,
        }

    def clear(self) -> None:
        """Reset the pipeline stages and results."""
        self.stages.clear()
        self.results.clear()
        self._stage_registry.clear()
        logger.info("Pipeline stages cleared")
