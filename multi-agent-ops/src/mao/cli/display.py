"""Rich-based display utilities for the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from mao.core.types import Task as TaskType

console = Console()


def show_banner() -> None:
    """Display the MAO system banner."""
    banner = """
[bold cyan]╔══════════════════════════════════════════════════╗
║   🔄 多Agent协同运营自动化系统 (MAO) v1.0        ║
║   Multi-Agent Collaborative Operations System    ║
╚══════════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(banner)


def show_task_start(description: str, task_id: str) -> None:
    """Display task start information."""
    console.print(Panel(
        f"[bold]任务描述:[/bold] {description}\n[bold]任务ID:[/bold] [dim]{task_id}[/dim]",
        title="🚀 启动新任务",
        border_style="cyan",
    ))


def show_task_status(task: dict[str, Any]) -> None:
    """Display task status."""
    status_color = {
        "created": "white",
        "executing": "yellow",
        "completed": "green",
        "failed": "red",
        "rejected": "red",
    }.get(task.get("status", ""), "white")

    table = Table(title=f"任务状态: {task.get('id', 'N/A')}", box=box.ROUNDED)
    table.add_column("属性", style="cyan")
    table.add_column("值", style="white")
    table.add_row("描述", task.get("description", "")[:100])
    table.add_row("状态", f"[{status_color}]{task.get('status', '')}[/{status_color}]")
    table.add_row("Token消耗", str(task.get("total_tokens", 0)))
    table.add_row("交互轮次", str(task.get("rounds", 0)))
    table.add_row("创建时间", str(task.get("created_at", "")))
    console.print(table)


def show_task_list(tasks: list[dict[str, Any]]) -> None:
    """Display a list of tasks."""
    if not tasks:
        console.print("[dim]暂无任务记录[/dim]")
        return

    table = Table(title="任务列表", box=box.ROUNDED)
    table.add_column("ID", style="dim")
    table.add_column("描述", style="white")
    table.add_column("状态", style="cyan")
    table.add_column("Tokens", style="yellow")
    table.add_column("轮次", style="magenta")
    table.add_column("更新时间", style="dim")

    for t in tasks:
        status_color = {"completed": "green", "failed": "red"}.get(t.get("status", ""), "white")
        table.add_row(
            t.get("id", "")[:12],
            t.get("description", "")[:60],
            f"[{status_color}]{t.get('status', '')}[/{status_color}]",
            str(t.get("total_tokens", 0)),
            str(t.get("rounds", 0)),
            str(t.get("updated_at", "")),
        )
    console.print(table)


def show_final_output(task: "TaskType | dict[str, Any]") -> None:
    """Display the final task output."""
    if isinstance(task, dict):
        output = task.get("final_output", "") or "(无输出)"
        description = task.get("description", "")
        stats = task.get("stats", {})
        total_tokens = task.get("total_tokens", 0)
        rounds = task.get("rounds", 0)
    else:
        output = task.final_output or "(无输出)"
        description = task.description
        stats = task.stats
        total_tokens = task.total_tokens
        rounds = task.rounds

    console.print(Panel(
        Markdown(output[:10000]),
        title="📝 最终输出",
        border_style="green",
    ))

    # Stats summary
    table = Table(title="执行统计", box=box.SIMPLE)
    table.add_column("指标", style="cyan")
    table.add_column("值", style="white")
    table.add_row("Token消耗", f"[bold yellow]{total_tokens}[/bold yellow]")
    table.add_row("交互轮次", str(rounds))
    table.add_row("Agent交互次数", str(len(stats.get("agent_interactions", [])) if isinstance(stats, dict) else 0))

    # Show reasoning chain depth if available
    chain = stats.get("reasoning_chain", []) if isinstance(stats, dict) else []
    if chain:
        table.add_row("长链推理步数", f"[bold cyan]{len(chain)}[/bold cyan]")
        actions = [s.get("action", "") for s in chain]
        plan_count = actions.count("plan")
        exec_count = actions.count("execute")
        review_count = actions.count("review")
        revise_count = actions.count("revise")
        arb_count = actions.count("arbitrate")
        table.add_row(
            "推理链组成",
            f"规划×{plan_count} 执行×{exec_count} 审核×{review_count} 修改×{revise_count} 仲裁×{arb_count}"
        )
    deep_mode = stats.get("deep_reasoning", False) if isinstance(stats, dict) else False
    table.add_row("深度推理模式", "[bold green]开启[/bold green]" if deep_mode else "[dim]标准[/dim]")
    console.print(table)


def show_agents(config: dict[str, Any]) -> None:
    """Display configured agents."""
    table = Table(title="可用Agent", box=box.ROUNDED)
    table.add_column("角色", style="cyan")
    table.add_column("名称", style="white")
    table.add_column("模型", style="yellow")
    table.add_column("提供者", style="magenta")

    for key, cfg in config.items():
        if hasattr(cfg, 'name'):
            table.add_row(key, cfg.name, cfg.model, cfg.provider.value)
        elif isinstance(cfg, dict):
            table.add_row(key, cfg.get("name", ""), cfg.get("model", ""), cfg.get("provider", ""))
    console.print(table)


def show_live_log(title: str) -> Live:
    """Create a Live display for real-time log output."""
    return Live(
        Panel("", title=title, border_style="blue"),
        console=console,
        refresh_per_second=4,
        transient=True,
    )
