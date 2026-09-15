"""Top-level Typer CLI. Entrypoints stay thin; logic lives in packages."""

from __future__ import annotations

import typer

from research_navigator.common.qdrant import check_health
from research_navigator.config import get_settings
from research_navigator.logging import configure_logging

app = typer.Typer(help="AI Research Navigator", no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def healthcheck() -> None:
    """Verify the local Qdrant instance is reachable."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    if not check_health(settings):
        raise typer.Exit(code=1)
    typer.echo("Qdrant is healthy.")


@app.command()
def parse(write: bool = True) -> None:
    """Parse the corpus into normalized IR and (optionally) cache it to disk."""
    from research_navigator.parse.dispatch import parse_corpus

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    docs = parse_corpus(settings, write=write)
    total_sections = sum(len(d.sections) for d in docs)
    with_warnings = sum(1 for d in docs if d.warnings)
    typer.echo(
        f"Parsed {len(docs)} documents, {total_sections} sections, {with_warnings} with warnings."
    )


if __name__ == "__main__":
    app()
