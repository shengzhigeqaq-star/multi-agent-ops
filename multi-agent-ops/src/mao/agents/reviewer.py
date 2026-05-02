"""Reviewer Agent — quality inspection and validation."""

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


class ReviewerAgent(BaseAgent):
    """Reviewer: checks execution results against success criteria."""

    role = AgentRole.REVIEWER
    name = "审核员"

    async def handle_message(self, message: Message) -> Message:
        """Review an execution result."""
        task_desc = message.payload.get("task", "")
        result_content = message.payload.get("result", "")
        criteria = message.payload.get("criteria", "")

        self.context.add_log(f"审核员审查: {task_desc[:80]}...")

        review = await self._review(task_desc, result_content, criteria)
        verdict = review.get("verdict", "PASS")

        return self.send_message(
            to="coordinator",
            msg_type=MessageType.REVIEW,
            payload={
                "task": task_desc,
                "result": result_content,
                "review": review,
            },
            parent_id=message.id,
        )

    async def execute_subtask(self, subtask: SubTask) -> SubTask:
        """Review a subtask's execution result."""
        self.context.add_log(f"审核员审查子任务: {subtask.title}")

        review = await self._review(
            subtask.description,
            subtask.result or "",
            subtask.success_criteria,
        )

        subtask.review_verdict = review.get("verdict", "PASS")
        subtask.review_score = review.get("score", 5)

        if review.get("verdict") == "PASS":
            subtask.status = TaskStatus.COMPLETED
        else:
            subtask.status = TaskStatus.REVISING
            subtask.error = json.dumps(review.get("suggestions", []), ensure_ascii=False)

        return subtask

    async def _review(self, task: str, result: str, criteria: str = "") -> dict:
        prompt = f"""请审查以下任务执行结果：

任务：{task}
{"成功标准：" + criteria if criteria else ""}
执行结果：
{result[:5000]}

请以JSON格式给出审核结论（只输出JSON，不要其他文字）：
{{
  "verdict": "PASS 或 REVISE",
  "score": 1-10,
  "issues": ["发现的问题"],
  "suggestions": ["改进建议"],
  "summary": "审核总结"
}}

审查要点：
- 完整性：执行结果是否覆盖了任务要求的所有内容
- 准确性：信息是否准确、逻辑是否合理
- 质量：输出是否达到可直接使用的标准
- 一致性：是否与整体目标一致"""
        response = await self.think(prompt)
        return self._parse_json(response.content)

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"verdict": "PASS", "score": 5, "issues": [], "suggestions": [], "summary": text}
