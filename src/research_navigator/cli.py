"""Top-level Typer CLI. Entrypoints stay thin; logic lives in packages."""

from __future__ import annotations

import typer

from research_navigator.common.qdrant import check_health
from research_navigator.config import get_settings
from research_navigator.logging import configure_logging

app = typer.Typer(help="AI Research Navigator", no_args_is_help=True)


@app.callback()  # type: ignore[misc]
def main() -> None:
    pass


@app.command()  # type: ignore[misc]
def healthcheck() -> None:
    """Verify the local Qdrant instance is reachable."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    if not check_health(settings):
        raise typer.Exit(code=1)
    typer.echo("Qdrant is healthy.")


if __name__ == "__main__":
    app()
