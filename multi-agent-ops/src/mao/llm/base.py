"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from mao.core.types import LLMResponse, ToolDefinition


class BaseLLM(ABC):
    """Abstract base for all LLM backends."""

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""
        ...
