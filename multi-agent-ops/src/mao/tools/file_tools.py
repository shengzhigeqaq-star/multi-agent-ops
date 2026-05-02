"""File operation tools."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from mao.core.types import ToolResult
from mao.tools.base import BaseTool


class FileReadTool(BaseTool):
    name = "file_read"
    description = "读取指定文件的内容。参数: file_path (str) — 文件路径"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "要读取的文件路径"},
        },
        "required": ["file_path"],
    }

    async def execute(self, file_path: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            p = Path(file_path)
            if not p.exists():
                return ToolResult(tool_name=self.name, success=False, error=f"文件不存在: {file_path}")
            if p.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
                return ToolResult(tool_name=self.name, success=False, error="文件超过10MB限制")
            content = p.read_text(encoding="utf-8", errors="replace")
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=content[:50000],  # Truncate for context window
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class FileWriteTool(BaseTool):
    name = "file_write"
    description = "写入内容到指定文件。参数: file_path (str), content (str)"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "目标文件路径"},
            "content": {"type": "string", "description": "要写入的内容"},
        },
        "required": ["file_path", "content"],
    }

    async def execute(self, file_path: str = "", content: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"成功写入 {len(content)} 字符到 {file_path}",
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class FileListTool(BaseTool):
    name = "file_list"
    description = "列出目录中的文件和子目录。参数: dir_path (str) — 目录路径，默认为当前目录"
    parameters = {
        "type": "object",
        "properties": {
            "dir_path": {"type": "string", "description": "要列出的目录路径，默认'.'"},
        },
        "required": [],
    }

    async def execute(self, dir_path: str = ".", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            p = Path(dir_path)
            if not p.exists():
                return ToolResult(tool_name=self.name, success=False, error=f"目录不存在: {dir_path}")
            entries = []
            for entry in sorted(p.iterdir()):
                t = "DIR" if entry.is_dir() else "FILE"
                try:
                    size = entry.stat().st_size if entry.is_file() else 0
                except OSError:
                    size = 0
                entries.append(f"[{t}] {entry.name}  ({size}B)" if entry.is_file() else f"[{t}] {entry.name}/")
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n".join(entries[:200]),
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))
