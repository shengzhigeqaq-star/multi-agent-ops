"""LLM backend factory."""

from __future__ import annotations

from mao.core.types import ProviderType
from mao.llm.anthropic_llm import AnthropicLLM
from mao.llm.base import BaseLLM
from mao.llm.openai_llm import OpenAILLM


def create_llm(
    provider: ProviderType,
    model: str,
    api_key: str,
    api_base: str | None = None,
    **kwargs,
) -> BaseLLM:
    """Create an LLM backend instance based on provider type."""
    if provider == ProviderType.ANTHROPIC:
        return AnthropicLLM(model=model, api_key=api_key, **kwargs)
    else:
        return OpenAILLM(model=model, api_key=api_key, api_base=api_base, **kwargs)
