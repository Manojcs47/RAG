.PHONY: setup lint format type test test-integration up down healthcheck clean

setup:
	uv sync
	uv run pre-commit install
	@test -f .secrets.baseline || uv run detect-secrets scan > .secrets.baseline

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

type:
	uv run mypy src

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

up:
	docker compose up -d

down:
	docker compose down

healthcheck:
	uv run research-navigator healthcheck

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

.PHONY: graph
graph:  ## Render the M3 agent graph to docs/agent_graph.mmd (offline)
	uv run python scripts/visualize_graph.py

.PHONY: graph-png
graph-png:  ## Also render docs/agent_graph.png (uses mermaid.ink — needs network)
	uv run python scripts/visualize_graph.py --png
