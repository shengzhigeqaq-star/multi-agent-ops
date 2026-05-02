"""Typer CLI application definition."""

from __future__ import annotations

import typer

from mao.cli.commands import (
    agents_command,
    config_show_command,
    inspect_command,
    list_command,
    run_command,
    status_command,
    tools_command,
)

app = typer.Typer(
    name="mao",
    help="多Agent协同运营自动化系统 (Multi-Agent Ops)",
    no_args_is_help=True,
)

app.command(name="run")(run_command)
app.command(name="status")(status_command)
app.command(name="list")(list_command)
app.command(name="inspect")(inspect_command)
app.command(name="agents")(agents_command)
app.command(name="tools")(tools_command)
app.command(name="config")(config_show_command)
