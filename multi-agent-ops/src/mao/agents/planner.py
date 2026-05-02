"""Planner Agent — task decomposition and strategy."""

from __future__ import annotations

import json
import re

from mao.core.agent import BaseAgent
from mao.core.types import (
    AgentRole,
    Message,
    MessageType,
    SubTask,
    TaskStatus,
)


class PlannerAgent(BaseAgent):
    """Planner: decomposes user goals into executable subtask plans."""

    role = AgentRole.PLANNER
    name = "规划师"

    async def handle_message(self, message: Message) -> Message:
        """Receive a task description and return a decomposition plan."""
        task_desc = message.payload.get("description", "")
        self.context.add_log(f"规划师收到任务: {task_desc[:80]}...")

        plan = await self._generate_plan(task_desc)
        return self.send_message(
            to="coordinator",
            msg_type=MessageType.RESULT,
            payload={"plan": plan},
            parent_id=message.id,
        )

    async def execute_subtask(self, subtask: SubTask) -> SubTask:
        """Execute a planning subtask — generate a plan for it."""
        plan = await self._generate_plan(subtask.description)
        subtask.result = json.dumps(plan, ensure_ascii=False, indent=2)
        subtask.status = TaskStatus.COMPLETED
        return subtask

    async def _generate_plan(self, goal: str) -> dict:
        prompt = f"""请分析以下任务目标并将其分解为可执行的子任务计划：

任务目标：{goal}

请以JSON格式输出计划（只输出JSON，不要其他文字）：
{{
  "goal": "对任务的简要概述",
  "subtasks": [
    {{
      "id": "sub_1",
      "title": "子任务标题",
      "description": "详细的执行描述",
      "assigned_role": "executor",
      "dependencies": [],
      "expected_output": "期望产出",
      "success_criteria": "成功的判断标准"
    }}
  ],
  "estimated_rounds": 3,
  "strategy": "总体策略说明"
}}

要求：
- 子任务数量控制在3-6个
- 角色的可选值: executor, reviewer, coordinator
- 明确子任务之间的依赖关系（仅依赖前面的子任务）
- 确保子任务可独立验证"""
        response = await self.think(prompt)
        return self._parse_json(response.content)

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from code blocks
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"goal": "无法解析计划", "subtasks": [], "strategy": text}
