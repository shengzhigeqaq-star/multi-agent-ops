"""CLI command implementations."""

from __future__ import annotations

import asyncio
from typing import Any

import typer

from mao.cli.display import (
    show_banner,
    show_final_output,
    show_task_list,
    show_task_start,
    show_task_status,
)
from mao.config.loader import load_config
from mao.core.orchestrator import Orchestrator


def _get_orchestrator(config_dir: str = "config") -> Orchestrator:
    config = load_config(config_dir)
    return Orchestrator(config)


async def _run_task(
    description: str,
    workflow: str | None = None,
    config_dir: str = "config",
    deep: bool = False,
) -> None:
    show_banner()
    orch = _get_orchestrator(config_dir)
    task = await orch.run(description, workflow, deep_reasoning=deep)
    show_final_output(task)


def run_command(
    description: str = typer.Argument(..., help="任务描述"),
    workflow: str = typer.Option(None, "-f", "--workflow", help="使用预定义工作流模板"),
    deep: bool = typer.Option(False, "--deep", help="启用深度推理模式（多轮审核-修改循环 + 仲裁机制）"),
    config_dir: str = typer.Option("config", "-c", "--config", help="配置文件目录"),
) -> None:
    """启动一个新的多Agent协同任务.

    系统将自动调用规划师→执行者→审核员→协调员四个角色协作完成任务。
    使用 --deep 标志启用长链深度推理，包含多轮审核-修改循环和协调员仲裁。
    """
    asyncio.run(_run_task(description, workflow, config_dir, deep))


def status_command(
    task_id: str = typer.Argument(..., help="任务ID"),
    config_dir: str = typer.Option("config", "-c", "--config", help="配置文件目录"),
) -> None:
    """查看指定任务的执行状态."""
    orch = _get_orchestrator(config_dir)
    task = orch.get_task(task_id)
    if not task:
        typer.echo(f"任务 {task_id} 未找到")
        return
    show_task_status(task)

    if task.get("final_output"):
        show_final_output(task)


def list_command(
    limit: int = typer.Option(20, "-n", "--limit", help="显示数量"),
    config_dir: str = typer.Option("config", "-c", "--config", help="配置文件目录"),
) -> None:
    """列出最近的任务."""
    orch = _get_orchestrator(config_dir)
    tasks = orch.list_tasks(limit)
    show_task_list(tasks)


def inspect_command(
    task_id: str = typer.Argument(..., help="任务ID"),
    show_chain: bool = typer.Option(False, "--chain", help="显示完整长链推理过程"),
    config_dir: str = typer.Option("config", "-c", "--config", help="配置文件目录"),
) -> None:
    """查看任务详细执行日志和推理链."""
    orch = _get_orchestrator(config_dir)
    task = orch.get_task(task_id)
    if not task:
        typer.echo(f"任务 {task_id} 未找到")
        return

    show_task_status(task)

    stats = task.get("stats", {})

    # Show reasoning chain if requested
    if show_chain:
        chain = stats.get("reasoning_chain", [])
        if chain:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            console.print("\n[bold cyan]━━━ 长链推理过程 (Long-Chain Reasoning Trace) ━━━[/bold cyan]")

            table = Table(title=f"推理链追踪 (共{len(chain)}步)", box=box.ROUNDED)
            table.add_column("步骤", style="dim", width=6)
            table.add_column("Agent", style="cyan", width=12)
            table.add_column("动作", style="yellow", width=12)
            table.add_column("详情", style="white", width=50)
            table.add_column("结论", style="green", width=10)

            for rs in chain:
                detail = (rs.get("output") or rs.get("output_summary", ""))[:80]
                verdict = rs.get("verdict", "") or ""
                if rs.get("score"):
                    verdict += f" ({rs['score']}/10)"

                table.add_row(
                    rs.get("step_id", ""),
                    rs.get("agent", ""),
                    rs.get("action", ""),
                    detail,
                    verdict,
                )
            console.print(table)
        else:
            typer.echo("\n[dim]未找到推理链记录（可能是标准模式运行的任务）[/dim]")

    # Show execution log
    log = stats.get("log", [])
    if log:
        print("\n执行日志:")
        for entry in log:
            print(f"  {entry}")

    if task.get("final_output"):
        show_final_output(task)


def agents_command(
    config_dir: str = typer.Option("config", "-c", "--config", help="配置文件目录"),
) -> None:
    """列出所有可用的Agent及其配置."""
    from mao.cli.display import show_agents
    config = load_config(config_dir)
    show_agents(config.agents)


def tools_command(
    config_dir: str = typer.Option("config", "-c", "--config", help="配置文件目录"),
) -> None:
    """列出所有可用的工具."""
    from rich.table import Table
    from rich import box
    from rich.console import Console

    config = load_config(config_dir)
    tools = config.tools

    table = Table(title="可用工具", box=box.ROUNDED)
    table.add_column("工具名", style="cyan")
    table.add_column("描述", style="white")
    table.add_column("状态", style="green")

    for name, cfg in tools.items():
        enabled = "✅ 启用" if cfg.get("enabled", True) else "❌ 禁用"
        table.add_row(name, cfg.get("description", ""), enabled)

    Console().print(table)


def config_show_command(
    config_dir: str = typer.Option("config", "-c", "--config", help="配置文件目录"),
) -> None:
    """显示当前系统配置."""
    from pathlib import Path
    import re

    base = Path(config_dir)
    for fname in ["default.yaml", "agents.yaml", "tools.yaml", "workflows.yaml"]:
        fpath = base / fname
        if fpath.exists():
            print(f"\n{'='*60}")
            print(f"  {fname}")
            print(f"{'='*60}")
            content = fpath.read_text(encoding="utf-8")
            content = re.sub(r'(api_key|key|secret):\s*"[^"]+"', r'\1: "***"', content)
            content = re.sub(r"(sk-[a-zA-Z0-9_-]+)", "***", content)
            content = re.sub(r"(sk-ant-[a-zA-Z0-9_-]+)", "***", content)
            print(content)
