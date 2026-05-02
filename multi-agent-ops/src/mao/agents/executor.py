"""Executor Agent — performs concrete operations using tools."""

from __future__ import annotations

from mao.core.agent import BaseAgent
from mao.core.types import (
    AgentRole,
    Message,
    MessageType,
    SubTask,
    TaskStatus,
)


class ExecutorAgent(BaseAgent):
    """Executor: carries out subtasks using tools and produces results."""

    role = AgentRole.EXECUTOR
    name = "执行者"

    async def handle_message(self, message: Message) -> Message:
        """Execute a task described in the message."""
        task_desc = message.payload.get("description", message.payload.get("title", ""))
        context_info = message.payload.get("context", "")

        self.context.add_log(f"执行者收到任务: {task_desc[:80]}...")

        result = await self._execute_task(task_desc, context_info)

        return self.send_message(
            to="reviewer",
            msg_type=MessageType.RESULT,
            payload={
                "task": task_desc,
                "result": result,
                "context": context_info,
            },
            parent_id=message.id,
        )

    async def execute_subtask(self, subtask: SubTask) -> SubTask:
        """Execute a formal Subtask."""
        self.context.add_log(f"执行者开始执行子任务: {subtask.title}")

        result = await self._execute_task(
            subtask.description,
            context_info=f"期望产出: {subtask.expected_output}\n成功标准: {subtask.success_criteria}",
        )

        subtask.result = result
        subtask.status = TaskStatus.COMPLETED
        return subtask

    async def _execute_task(self, description: str, context_info: str = "") -> str:
        """Core execution loop — think, use tools, iterate."""
        prompt = f"""请完成以下任务：

任务：{description}
{"上下文：" + context_info if context_info else ""}

要求：
1. 如果任务需要获取外部信息，请使用可用的工具（web_search, file_read等）
2. 如果任务需要写入文件，请使用 file_write 工具
3. 如果任务涉及代码分析，请使用 python_exec 工具
4. 请直接给出结果，不要多余的解释框架

现在开始执行任务。"""
        response = await self.think(prompt)

        # Handle tool calls if any
        max_tool_rounds = 3
        for _ in range(max_tool_rounds):
            if not response.tool_calls:
                break
            tool_results = await self.use_tools(response.tool_calls)

            # Feed tool results back to LLM
            follow_up = "工具执行结果如下：\n"
            for tr in tool_results:
                if hasattr(tr, 'success'):
                    follow_up += f"\n[{tr.tool_name}]: {'成功' if tr.success else '失败'}\n输出: {tr.output if tr.success else tr.error}\n"
                else:
                    follow_up += f"\n{tr}\n"

            follow_up += "\n请根据以上工具结果继续完成任务。如果任务已完成，请给出最终结果。"
            response = await self.think(follow_up)

        return response.content
