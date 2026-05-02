"""Configuration loader — reads YAML configs and merges with environment."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from mao.core.types import AgentConfig, AgentRole, ProviderType, SystemConfig

load_dotenv()

_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env(value: Any) -> Any:
    """Recursively resolve ${VAR} placeholders in config values."""
    if isinstance(value, str):
        return _VAR_RE.sub(lambda m: os.getenv(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str = "config") -> SystemConfig:
    """Load and merge all configuration files."""
    base = Path(config_dir)

    defaults = _load_yaml(str(base / "default.yaml"))
    agents_raw = _load_yaml(str(base / "agents.yaml"))
    tools_raw = _load_yaml(str(base / "tools.yaml"))
    workflows_raw = _load_yaml(str(base / "workflows.yaml"))

    # Resolve environment variables
    defaults = _resolve_env(defaults)

    # Build agent configs
    agents: dict[str, AgentConfig] = {}
    for key, raw in agents_raw.items():
        provider_str = raw.get("provider", defaults.get("defaults", {}).get("default_provider", "openai_compatible"))
        agents[key] = AgentConfig(
            name=raw.get("name", key),
            role=AgentRole(raw.get("role", key)),
            provider=ProviderType(provider_str),
            model=raw.get("model", defaults.get("defaults", {}).get("default_model", "deepseek-chat")),
            api_base=raw.get("api_base", defaults.get("api_bases", {}).get(key)),
            temperature=raw.get("temperature", defaults.get("defaults", {}).get("temperature", 0.7)),
            max_tokens=raw.get("max_tokens", defaults.get("defaults", {}).get("max_tokens", 4096)),
            system_prompt=raw.get("system_prompt", ""),
            tools=raw.get("tools", []),
        )

    return SystemConfig(
        api_keys=defaults.get("api_keys", {}),
        api_bases=defaults.get("api_bases", {}),
        defaults=defaults.get("defaults", {}),
        storage=defaults.get("storage", {}),
        agents=agents,
        tools=tools_raw.get("tools", {}),
        workflows=workflows_raw.get("workflows", {}),
    )
