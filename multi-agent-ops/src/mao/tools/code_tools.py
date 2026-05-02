"""Python code execution tool."""

from __future__ import annotations

import io
import sys
import time
import traceback
from typing import Any

from mao.core.types import ToolResult
from mao.tools.base import BaseTool


class PythonExecTool(BaseTool):
    name = "python_exec"
    description = "执行Python代码片段并返回输出。参数: code (str) — Python代码"
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的Python代码"},
        },
        "required": ["code"],
    }

    async def execute(self, code: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        safe_builtins = {
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
            "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
            "format", "frozenset", "getattr", "hasattr", "hash", "hex", "int",
            "isinstance", "issubclass", "iter", "len", "list", "map", "max",
            "min", "next", "oct", "ord", "pow", "print", "range", "repr",
            "reversed", "round", "set", "slice", "sorted", "str", "sum",
            "tuple", "type", "zip", "__import__",
        }

        safe_globals: dict[str, Any] = {
            "__builtins__": {k: __builtins__[k] for k in safe_builtins if k in __builtins__},
        }

        try:
            exec(code, safe_globals)
            output = sys.stdout.getvalue()
            err_output = sys.stderr.getvalue()
            if err_output:
                output += "\n[STDERR]\n" + err_output
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output.strip() or "(无输出)",
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception:
            tb = traceback.format_exc()
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=tb,
                duration_ms=(time.time() - t0) * 1000,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
