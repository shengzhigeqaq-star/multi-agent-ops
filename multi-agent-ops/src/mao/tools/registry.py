"""Tool registry — manages available tools and provides lookup."""

from __future__ import annotations

from typing import Any

from mao.core.types import ToolDefinition
from mao.tools.base import BaseTool
from mao.tools.code_tools import PythonExecTool
from mao.tools.file_tools import FileListTool, FileReadTool, FileWriteTool
from mao.tools.shell_tools import ShellTool
from mao.tools.web_tools import WebFetchTool, WebSearchTool


class ToolRegistry:
    """Central registry for all available tools."""

    _default_tools: dict[str, type[BaseTool]] = {
        "file_read": FileReadTool,
        "file_write": FileWriteTool,
        "file_list": FileListTool,
        "web_search": WebSearchTool,
        "web_fetch": WebFetchTool,
        "shell": ShellTool,
        "python_exec": PythonExecTool,
    }

    def __init__(self, enabled_tools: dict[str, dict[str, Any]] | None = None):
        self._instances: dict[str, BaseTool] = {}
        enabled = enabled_tools or {}

        for name, cls in self._default_tools.items():
            cfg = enabled.get(name, {})
            if cfg.get("enabled", True):
                self._instances[name] = cls()

    def get(self, name: str) -> BaseTool | None:
        """Get a tool instance by name."""
        return self._instances.get(name)

    def list_tools(self) -> list[str]:
        """List available tool names."""
        return list(self._instances.keys())

    def get_definitions(self, tool_names: list[str] | None = None) -> list[ToolDefinition]:
        """Get ToolDefinition list for enabled tools (for LLM function calling)."""
        names = tool_names or list(self._instances.keys())
        defs: list[ToolDefinition] = []
        for name in names:
            tool = self._instances.get(name)
            if tool:
                td = tool.to_definition()
                defs.append(ToolDefinition(
                    name=td["name"],
                    description=td["description"],
                    parameters=td["parameters"],
                ))
        return defs

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool by name and return its result."""
        tool = self._instances.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
        result = await tool.execute(**kwargs)
        return result
