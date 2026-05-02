"""Core shared types, enums, and Pydantic models for the MAO system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"


class TaskStatus(str, Enum):
    CREATED = "created"
    DECOMPOSING = "decomposing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class MessageType(str, Enum):
    TASK = "task"
    QUERY = "query"
    RESULT = "result"
    REVIEW = "review"
    HANDOFF = "handoff"
    BROADCAST = "broadcast"


class ProviderType(str, Enum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"


# ── Message ──────────────────────────────────────────────────────────────────


class Message(BaseModel):
    """Inter-agent communication message."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_agent: str
    to_agent: str  # agent role name or "broadcast"
    type: MessageType = MessageType.TASK
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = Field(default=3, ge=1, le=5)


# ── Task & SubTask ───────────────────────────────────────────────────────────


class SubTask(BaseModel):
    """A single sub-task within a larger task."""

    id: str = Field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    assigned_role: AgentRole = AgentRole.EXECUTOR
    dependencies: list[str] = Field(default_factory=list)
    expected_output: str = ""
    success_criteria: str = ""
    status: TaskStatus = TaskStatus.CREATED
    result: Optional[str] = None
    review_verdict: Optional[str] = None  # PASS or REVISE
    review_score: Optional[int] = None
    error: Optional[str] = None
    token_usage: int = 0


class Task(BaseModel):
    """Top-level task representing a user request."""

    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    description: str
    status: TaskStatus = TaskStatus.CREATED
    subtasks: list[SubTask] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    final_output: Optional[str] = None
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_tokens: int = 0
    rounds: int = 0


# ── LLM Types ────────────────────────────────────────────────────────────────


class LLMResponse(BaseModel):
    """Unified response from any LLM backend."""

    content: str
    model: str
    provider: ProviderType
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """Tool definition for LLM function calling."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── Tool Result ──────────────────────────────────────────────────────────────


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0


# ── Agent Config ─────────────────────────────────────────────────────────────


class AgentConfig(BaseModel):
    """Configuration for a single agent role."""

    name: str
    role: AgentRole
    provider: ProviderType = ProviderType.OPENAI_COMPATIBLE
    model: str = "deepseek-chat"
    api_base: Optional[str] = None
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)


class SystemConfig(BaseModel):
    """Top-level system configuration."""

    api_keys: dict[str, str] = Field(default_factory=dict)
    api_bases: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    storage: dict[str, str] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    workflows: dict[str, Any] = Field(default_factory=dict)


# ── Workflow Types ───────────────────────────────────────────────────────────


class WorkflowStep(BaseModel):
    """A step in a workflow template."""

    id: str
    agent: str  # agent role name
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.CREATED
