"""Coordinator Agent — global scheduling, conflict resolution, and final synthesis."""

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


class CoordinatorAgent(BaseAgent):
    """Coordinator: orchestrates execution, resolves conflicts, synthesizes final output."""

    role = AgentRole.COORDINATOR
    name = "协调员"

    async def handle_message(self, message: Message) -> Message:
        """Handle various message types from other agents."""
        msg_type = message.type
        payload = message.payload

        if msg_type == MessageType.RESULT and "plan" in payload:
            # Planner sent a plan — acknowledge
            plan = payload["plan"]
            subtask_count = len(plan.get("subtasks", []))
            self.context.add_log(f"协调员收到计划: {subtask_count}个子任务")
            return self.send_message(
                to="planner",
                msg_type=MessageType.RESULT,
                payload={"acknowledged": True, "subtask_count": subtask_count},
                parent_id=message.id,
            )

        elif msg_type == MessageType.RESULT:
            # Executor sent a result — forward to reviewer
            self.context.add_log("协调员将执行结果转发给审核员")
            return self.send_message(
                to="reviewer",
                msg_type=MessageType.QUERY,
                payload=payload,
                parent_id=message.id,
            )

        elif msg_type == MessageType.REVIEW:
            # Reviewer sent a review — record and proceed
            review = payload.get("review", {})
            verdict = review.get("verdict", "PASS")
            self.context.add_log(f"审核结果: {verdict}")
            return self.send_message(
                to="broadcast",
                msg_type=MessageType.RESULT,
                payload={"review_complete": True, "verdict": verdict},
                parent_id=message.id,
            )

        else:
            return self.send_message(
                to=message.from_agent,
                msg_type=MessageType.RESULT,
                payload={"acknowledged": True},
                parent_id=message.id,
            )

    async def execute_subtask(self, subtask: SubTask) -> SubTask:
        """Execute a coordination subtask — typically the final synthesis."""
        self.context.add_log(f"协调员执行: {subtask.title}")
        subtask.result = f"协调任务完成: {subtask.description}"
        subtask.status = TaskStatus.COMPLETED
        return subtask

    async def synthesize(self, task_desc: str, subtasks: list[SubTask]) -> str:
        """Synthesize all subtask results into a final cohesive output."""
        results_text = ""
        for st in subtasks:
            results_text += f"\n\n### {st.title}\n状态: {st.status.value}\n审核: {st.review_verdict or 'N/A'} (评分: {st.review_score or 'N/A'})\n结果:\n{st.result or '(无)'}"

        prompt = f"""请将以下子任务执行结果综合为一份完整的最终输出：

原始任务：{task_desc}

各子任务结果：
{results_text[:8000]}

请生成一份专业、连贯的最终综合报告。不要简单罗列各子任务结果，而是整合为一个完整的交付物。
如果子任务之间有矛盾或不一致的地方，请指出并尝试调和。"""
        response = await self.think(prompt)
        return response.content

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}
