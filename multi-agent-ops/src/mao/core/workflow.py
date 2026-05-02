"""Workflow DAG engine — schedules and executes workflow steps with dependency resolution."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from mao.core.types import TaskStatus, WorkflowStep


class WorkflowEngine:
    """Executes workflow steps respecting DAG dependencies."""

    def __init__(self, steps: list[WorkflowStep]):
        self.steps: dict[str, WorkflowStep] = {s.id: s for s in steps}
        self._in_degree: dict[str, int] = {}
        self._dependents: dict[str, list[str]] = {}
        self._compute_graph()

    def _compute_graph(self) -> None:
        """Compute in-degrees and reverse dependency map."""
        for sid in self.steps:
            self._in_degree[sid] = 0
            self._dependents[sid] = []

        for sid, step in self.steps.items():
            self._in_degree[sid] = len(step.depends_on)
            for dep in step.depends_on:
                if dep in self._dependents:
                    self._dependents[dep].append(sid)

    def get_ready_steps(self) -> list[str]:
        """Return step IDs with all dependencies completed (in-degree = 0)."""
        ready = []
        for sid, deg in self._in_degree.items():
            if deg == 0 and self.steps[sid].status == TaskStatus.CREATED:
                ready.append(sid)
        return ready

    def mark_completed(self, step_id: str) -> list[str]:
        """Mark a step as completed and return newly unblocked steps."""
        step = self.steps.get(step_id)
        if not step:
            return []
        step.status = TaskStatus.COMPLETED

        newly_ready = []
        for dep_id in self._dependents.get(step_id, []):
            self._in_degree[dep_id] -= 1
            if self._in_degree[dep_id] == 0:
                newly_ready.append(dep_id)
        return newly_ready

    def mark_failed(self, step_id: str) -> None:
        """Mark a step as failed."""
        step = self.steps.get(step_id)
        if step:
            step.status = TaskStatus.FAILED

    def is_complete(self) -> bool:
        """Check if all steps are in a terminal state."""
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.REJECTED}
        return all(s.status in terminal for s in self.steps.values())

    def get_execution_order(self) -> list[list[str]]:
        """Return steps grouped by parallel execution rounds (topological levels)."""
        indeg = dict(self._in_degree)
        queue = deque([sid for sid, deg in indeg.items() if deg == 0])
        levels: list[list[str]] = []

        while queue:
            level = list(queue)
            levels.append(level)
            queue.clear()
            for sid in level:
                for dep_id in self._dependents.get(sid, []):
                    indeg[dep_id] -= 1
                    if indeg[dep_id] == 0:
                        queue.append(dep_id)

        return levels

    @classmethod
    def from_template(cls, template: dict[str, Any]) -> WorkflowEngine:
        """Create engine from a workflow template dict."""
        steps = []
        for step_data in template.get("steps", []):
            steps.append(WorkflowStep(
                id=step_data["id"],
                agent=step_data.get("agent", "executor"),
                description=step_data.get("description", ""),
                depends_on=step_data.get("depends_on", []),
                tools=step_data.get("tools", []),
            ))
        return cls(steps)
