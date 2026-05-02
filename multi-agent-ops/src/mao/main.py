"""Main entry point for the MAO CLI."""

from __future__ import annotations

from mao.cli.app import app


def main() -> None:
    """Entry point for `mao` command."""
    app()


if __name__ == "__main__":
    main()
