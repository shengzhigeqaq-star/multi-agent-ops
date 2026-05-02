"""Abstract tool base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mao.core.types import ToolResult as ToolExecResult


class BaseTool(ABC):
    """Abstract base for all tools agents can use."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolExecResult:
        """Execute the tool with given parameters."""
        ...

    def to_definition(self) -> dict[str, Any]:
        """Return OpenAI/Anthropic-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
