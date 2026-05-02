"""BaseAgent — abstract base for all role agents."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from mao.core.types import (
    AgentRole,
    LLMResponse,
    Message,
    MessageType,
    SubTask,
    TaskStatus,
    ToolResult,
)
from mao.llm.base import BaseLLM
from mao.tools.registry import ToolRegistry


class AgentContext:
    """Runtime context shared across agents within a single task."""

    def __init__(self, task_id: str, max_rounds: int = 20):
        self.task_id = task_id
        self.max_rounds = max_rounds
        self.round: int = 0
        self.history: list[dict[str, Any]] = []
        self.shared_memory: dict[str, Any] = {}
        self.total_tokens: int = 0
        self.log: list[str] = []

    def add_log(self, entry: str) -> None:
        self.log.append(f"[Round {self.round}] {entry}")


class BaseAgent(ABC):
    """Abstract base for all role-specialized agents."""

    role: AgentRole
    name: str

    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.llm = llm
        self.tools = tool_registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._context: AgentContext | None = None

    def set_context(self, ctx: AgentContext) -> None:
        self._context = ctx

    @property
    def context(self) -> AgentContext:
        if self._context is None:
            raise RuntimeError("Agent context not set")
        return self._context

    async def think(self, user_message: str) -> LLMResponse:
        """Send a message to the LLM and get a response."""
        tool_defs = self.tools.get_definitions()
        messages = [{"role": "user", "content": user_message}]

        # Include recent history for context
        for h in self.context.history[-10:]:
            messages.insert(-1, h)

        response = await self.llm.chat(
            messages=messages,
            system=self.system_prompt,
            tools=tool_defs if tool_defs else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.context.total_tokens += response.usage.get("total_tokens", 0)
        self.context.history.append({"role": "user", "content": user_message})
        self.context.history.append({"role": "assistant", "content": response.content})
        return response

    async def use_tools(self, tool_calls: list[dict[str, Any]]) -> list[Any]:
        """Execute tool calls from an LLM response."""
        results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = await self.tools.execute(tool_name, **args)
            results.append(result)
            self.context.add_log(f"{self.name} 使用工具 {tool_name}: {result}")
        return results

    @abstractmethod
    async def handle_message(self, message: Message) -> Message:
        """Process an incoming message and return a response."""
        ...

    @abstractmethod
    async def execute_subtask(self, subtask: SubTask) -> SubTask:
        """Execute a subtask assigned to this agent."""
        ...

    def send_message(
        self,
        to: str,
        msg_type: MessageType,
        payload: dict[str, Any],
        parent_id: str | None = None,
        priority: int = 3,
    ) -> Message:
        """Create a message from this agent."""
        return Message(
            from_agent=self.name,
            to_agent=to,
            type=msg_type,
            payload=payload,
            parent_id=parent_id,
            priority=priority,
        )
