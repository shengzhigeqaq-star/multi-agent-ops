"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mao.core.types import ToolResult
from mao.tools.base import BaseTool

_ALLOWED_COMMANDS = {
    "ls", "dir", "cat", "head", "tail", "wc", "find", "grep",
    "python", "python3", "echo", "date", "pwd", "which",
    "mkdir", "cp", "mv", "touch", "git",
}


class ShellTool(BaseTool):
    name = "shell"
    description = (
        "执行Shell命令并返回输出。参数: command (str) — 要执行的命令。"
        f"允许的命令: {', '.join(sorted(_ALLOWED_COMMANDS))}"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的Shell命令"},
        },
        "required": ["command"],
    }

    def __init__(self, allowed_commands: set[str] | None = None):
        self._allowed = allowed_commands or _ALLOWED_COMMANDS

    async def execute(self, command: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        cmd_base = command.strip().split()[0] if command.strip() else ""
        if cmd_base not in self._allowed:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"命令 '{cmd_base}' 不在允许列表中。允许: {', '.join(sorted(self._allowed))}",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[STDERR]\n" + stderr.decode("utf-8", errors="replace")
            return ToolResult(
                tool_name=self.name,
                success=proc.returncode == 0,
                output=output[:10000],
                duration_ms=(time.time() - t0) * 1000,
            )
        except asyncio.TimeoutError:
            return ToolResult(tool_name=self.name, success=False, error="命令执行超时(60s)")
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))
