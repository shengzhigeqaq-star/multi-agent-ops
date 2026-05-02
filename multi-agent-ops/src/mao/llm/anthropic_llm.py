"""Anthropic Claude API backend."""

from __future__ import annotations

from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from mao.core.types import LLMResponse, ProviderType, ToolDefinition
from mao.llm.base import BaseLLM


class AnthropicLLM(BaseLLM):
    """LLM backend for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        **kwargs: Any,
    ):
        super().__init__(model, api_key, **kwargs)
        self._client = AsyncAnthropic(api_key=api_key)

    def _build_tools(self, tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # Convert OpenAI-format messages to Anthropic format
        anthropic_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            if role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": m.get("content", "")})
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": f"[Tool Result: {m.get('content', '')}]",
                })
            else:
                anthropic_messages.append({"role": "user", "content": m.get("content", "")})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = await self._client.messages.create(**kwargs)

        tool_calls = []
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input if isinstance(block.input, dict) else {},
                })

        return LLMResponse(
            content=text_content,
            model=response.model,
            provider=ProviderType.ANTHROPIC,
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
            },
            finish_reason=response.stop_reason or "stop",
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        anthropic_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            if role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": m.get("content", "")})
            else:
                anthropic_messages.append({"role": "user", "content": m.get("content", "")})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
