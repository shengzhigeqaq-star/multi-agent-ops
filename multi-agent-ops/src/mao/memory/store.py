"""Abstract memory store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    """Abstract interface for agent memory/storage backends."""

    @abstractmethod
    async def save_message(self, task_id: str, message: dict[str, Any]) -> None:
        """Persist an inter-agent message."""
        ...

    @abstractmethod
    async def get_messages(self, task_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a task."""
        ...

    @abstractmethod
    async def save_task(self, task: dict[str, Any]) -> None:
        """Save or update a task."""
        ...

    @abstractmethod
    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a task by ID."""
        ...

    @abstractmethod
    async def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent tasks."""
        ...

    @abstractmethod
    async def save_context(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a context snippet (for long-term / vector memory)."""
        ...

    @abstractmethod
    async def search_context(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search over stored contexts."""
        ...
