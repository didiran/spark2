"""
DAG-based pipeline orchestrator for the ML Training Pipeline.

Manages task dependencies, execution order, retry logic, status
tracking, and error handling for end-to-end pipeline runs.
"""

import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Execution status for pipeline tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class PipelineTask:
    """Definition of a single pipeline task."""
    name: str
    callable: Callable[..., Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    retry_delay_seconds: int = 30
    timeout_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class TaskExecution:
    """Record of a single task execution attempt."""
    task_name: str
    status: TaskStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    result: Any = None
    error: Optional[str] = None
    attempt: int = 1


@dataclass
class PipelineRun:
    """Record of a full pipeline execution."""
    run_id: str
    status: TaskStatus
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    task_executions: List[TaskExecution] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class PipelineOrchestrator:
    """
    DAG-based task orchestrator for ML pipeline execution.

    Manages a directed acyclic graph of tasks with dependency
    resolution, topological execution ordering, configurable
    retry logic, and comprehensive status tracking.

    Attributes:
        pipeline_name: Name of the pipeline.
        tasks: Dictionary of registered tasks.
        max_retries: Default maximum retry count.
        retry_delay: Default delay between retries in seconds.
    """

    def __init__(
        self,
        pipeline_name: str = "ml-training-pipeline",
        max_retries: int = 3,
        retry_delay: int = 30,
    ):
        self.pipeline_name = pipeline_name
        self.tasks: Dict[str, PipelineTask] = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._run_history: List[PipelineRun] = []
        self._task_results: Dict[str, Any] = {}
        logger.info(
            f"PipelineOrchestrator initialized | name={pipeline_name} "
            f"| max_retries={max_retries}"
        )

    def add_task(
        self,
        name: str,
        callable: Callable[..., Any],
        dependencies: Optional[List[str]] = None,
        max_retries: Optional[int] = None,
        retry_delay_seconds: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> "PipelineOrchestrator":
        """
        Register a task in the pipeline DAG.

        Args:
            name: Unique task identifier.
            callable: Function to execute for this task.
            dependencies: List of task names this task depends on.
            max_retries: Override default max retries.
            retry_delay_seconds: Override default retry delay.
            timeout_seconds: Optional per-task timeout.
            tags: Optional tags for categorization.
            description: Human-readable task description.

        Returns:
            Self for method chaining.
        """
        task = PipelineTask(
            name=name,
            callable=callable,
            dependencies=dependencies or [],
            max_retries=max_retries if max_retries is not None else self.max_retries,
            retry_delay_seconds=(
                retry_delay_seconds
                if retry_delay_seconds is not None
                else self.retry_delay
            ),
            timeout_seconds=timeout_seconds,
            tags=tags or [],
            description=description,
        )
        self.tasks[name] = task
        logger.info(
            f"Task registered | name={name} | deps={task.dependencies}"
        )
        return self

    def validate_dag(self) -> bool:
        """
        Validate the task DAG for cycles and missing dependencies.

        Returns:
            True if the DAG is valid.

        Raises:
            ValueError: If cycles or missing dependencies are found.
        """
        for task_name, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(
                        f"Task '{task_name}' depends on unknown task '{dep}'"
                    )

        if self._has_cycle():
            raise ValueError("Circular dependency detected in task DAG")

        logger.info("DAG validation passed")
        return True

    def _has_cycle(self) -> bool:
        """Detect cycles in the task DAG using DFS."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep_name, dep_task in self.tasks.items():
                if node in dep_task.dependencies:
                    continue

            task = self.tasks[node]
            for dep in task.dependencies:
                if dep not in visited:
                    if _dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.discard(node)
            return False

        for task_name in self.tasks:
            if task_name not in visited:
                if _dfs(task_name):
                    return True
        return False

    def _topological_sort(self) -> List[str]:
        """
        Compute topological execution order for the task DAG.

        Uses Kahn's algorithm for deterministic ordering.

        Returns:
            List of task names in execution order.

        Raises:
            ValueError: If the DAG contains cycles.
        """
        in_degree: Dict[str, int] = defaultdict(int)
        for task in self.tasks.values():
            if task.name not in in_degree:
                in_degree[task.name] = 0
            for dep in task.dependencies:
                in_degree[task.name] += 1

        queue = deque([
            name for name, degree in in_degree.items() if degree == 0
        ])

        order = []
        while queue:
            current = queue.popleft()
            order.append(current)

            for task_name, task in self.tasks.items():
                if current in task.dependencies:
                    in_degree[task_name] -= 1
                    if in_degree[task_name] == 0:
                        queue.append(task_name)

        if len(order) != len(self.tasks):
            raise ValueError("Cycle detected during topological sort")

        return order

    def run(
        self,
        context: Optional[Dict[str, Any]] = None,
        skip_tasks: Optional[List[str]] = None,
        fail_fast: bool = True,
    ) -> PipelineRun:
        """
        Execute the full pipeline in topological order.

        Args:
            context: Shared context dictionary passed to all tasks.
            skip_tasks: Task names to skip during execution.
            fail_fast: Stop pipeline on first task failure if True.

        Returns:
            PipelineRun with execution details for all tasks.
        """
        self.validate_dag()
        execution_order = self._topological_sort()

        run_id = f"{self.pipeline_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        pipeline_context = context or {}
        self._task_results = {}
        skip_set = set(skip_tasks or [])

        pipeline_run = PipelineRun(
            run_id=run_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.utcnow().isoformat(),
            context=pipeline_context,
        )

        logger.info(
            f"Pipeline started | run_id={run_id} | "
            f"tasks={len(execution_order)} | order={execution_order}"
        )

        failed_tasks: Set[str] = set()

        for task_name in execution_order:
            if task_name in skip_set:
                execution = TaskExecution(
                    task_name=task_name,
                    status=TaskStatus.SKIPPED,
                )
                pipeline_run.task_executions.append(execution)
                logger.info(f"Task skipped | name={task_name}")
                continue

            task = self.tasks[task_name]
            deps_failed = any(dep in failed_tasks for dep in task.dependencies)
            if deps_failed:
                execution = TaskExecution(
                    task_name=task_name,
                    status=TaskStatus.SKIPPED,
                    error="Dependency failed",
                )
                pipeline_run.task_executions.append(execution)
                failed_tasks.add(task_name)
                logger.warning(f"Task skipped (dependency failed) | name={task_name}")
                continue

            execution = self._execute_task(task, pipeline_context)
            pipeline_run.task_executions.append(execution)

            if execution.status == TaskStatus.COMPLETED:
                self._task_results[task_name] = execution.result
                pipeline_context[f"result_{task_name}"] = execution.result
            else:
                failed_tasks.add(task_name)
                if fail_fast:
                    logger.error(f"Pipeline stopping (fail_fast) | failed_task={task_name}")
                    break

        pipeline_run.end_time = datetime.utcnow().isoformat()

        if failed_tasks:
            pipeline_run.status = TaskStatus.FAILED
        else:
            pipeline_run.status = TaskStatus.COMPLETED

        start_dt = datetime.fromisoformat(pipeline_run.start_time)
        end_dt = datetime.fromisoformat(pipeline_run.end_time)
        pipeline_run.duration_seconds = (end_dt - start_dt).total_seconds()

        self._run_history.append(pipeline_run)

        logger.info(
            f"Pipeline finished | run_id={run_id} | status={pipeline_run.status.value} "
            f"| duration={pipeline_run.duration_seconds:.1f}s"
        )

        return pipeline_run

    def _execute_task(
        self,
        task: PipelineTask,
        context: Dict[str, Any],
    ) -> TaskExecution:
        """
        Execute a single task with retry logic.

        Args:
            task: PipelineTask to execute.
            context: Shared pipeline context.

        Returns:
            TaskExecution with result or error details.
        """
        for attempt in range(1, task.max_retries + 1):
            start_time = datetime.utcnow()

            logger.info(
                f"Task executing | name={task.name} | attempt={attempt}/{task.max_retries}"
            )

            try:
                result = task.callable(context)

                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                return TaskExecution(
                    task_name=task.name,
                    status=TaskStatus.COMPLETED,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    result=result,
                    attempt=attempt,
                )

            except Exception as exc:
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                error_msg = f"{type(exc).__name__}: {exc}"

                logger.error(
                    f"Task failed | name={task.name} | attempt={attempt} "
                    f"| error={error_msg}"
                )

                if attempt < task.max_retries:
                    logger.info(
                        f"Retrying task | name={task.name} "
                        f"| delay={task.retry_delay_seconds}s"
                    )
                    time.sleep(task.retry_delay_seconds)
                else:
                    return TaskExecution(
                        task_name=task.name,
                        status=TaskStatus.FAILED,
                        start_time=start_time.isoformat(),
                        end_time=end_time.isoformat(),
                        duration_seconds=duration,
                        error=error_msg,
                        attempt=attempt,
                    )

        return TaskExecution(
            task_name=task.name,
            status=TaskStatus.FAILED,
            error="Max retries exhausted",
        )

    def get_task_result(self, task_name: str) -> Any:
        """Retrieve the result of a completed task."""
        return self._task_results.get(task_name)

    def get_run_history(self) -> List[PipelineRun]:
        """Return the history of all pipeline runs."""
        return list(self._run_history)

    def get_dag_visualization(self) -> Dict[str, List[str]]:
        """
        Get the DAG structure as an adjacency list.

        Returns:
            Dictionary mapping each task to its dependencies.
        """
        return {
            name: task.dependencies
            for name, task in self.tasks.items()
        }

    def clear_tasks(self) -> None:
        """Remove all registered tasks."""
        self.tasks.clear()
        self._task_results.clear()
        logger.info("All tasks cleared")
