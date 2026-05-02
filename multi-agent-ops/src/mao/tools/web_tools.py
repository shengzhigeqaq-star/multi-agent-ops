"""Web search and fetch tools."""

from __future__ import annotations

import time
from typing import Any

import httpx

from mao.core.types import ToolResult
from mao.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "搜索网络信息（使用DuckDuckGo即时回答API）。参数: query (str) — 搜索查询"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询关键词"},
        },
        "required": ["query"],
    }

    async def execute(self, query: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Use DuckDuckGo Instant Answer API
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                    headers={"User-Agent": "MAO-Agent/1.0"},
                )
                data = resp.json()
                results: list[str] = []

                abstract = data.get("AbstractText", "")
                if abstract:
                    results.append(f"[摘要] {abstract}")
                    source = data.get("AbstractURL", "")
                    if source:
                        results.append(f"[来源] {source}")

                related = data.get("RelatedTopics", [])
                for topic in related[:5]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(f"[相关] {topic['Text']}")

                if not results:
                    results.append(f"未找到关于 '{query}' 的直接结果。建议使用 web_fetch 工具访问具体网页。")

                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output="\n\n".join(results),
                    duration_ms=(time.time() - t0) * 1000,
                )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = "获取指定URL的网页内容。参数: url (str) — 网页URL"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要获取的网页URL"},
        },
        "required": ["url"],
    }

    async def execute(self, url: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "MAO-Agent/1.0 (compatible; +https://github.com/mao)"},
                )
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type:
                    # Simple HTML text extraction
                    text = resp.text
                    # Remove script and style content
                    import re
                    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    text = text[:10000]  # Limit output size
                else:
                    text = resp.text[:10000]

                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=text,
                    duration_ms=(time.time() - t0) * 1000,
                )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=str(e))
