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

.PHONY: parse chunk prepare ingest reindex

parse:  ## Parse the corpus (PDF+Markdown) -> data/parsed/*.json
	uv run research-navigator parse

chunk:  ## Chunk cached parsed docs -> data/chunks/*.json (needs `make parse` first)
	uv run research-navigator chunk

prepare: parse chunk  ## Regenerate data/parsed + data/chunks from corpus/ (run once per corpus change)

ingest:  ## Ingest data/chunks/*.json into Qdrant (needs `make prepare` first, and `make up`)
	uv run research-navigator ingest

reindex:  ## Drop and rebuild the Qdrant collection from data/chunks/*.json
	uv run research-navigator reindex --yes

.PHONY: graph
graph:  ## Render the M3 agent graph to docs/agent_graph.mmd (offline)
	uv run python scripts/visualize_graph.py

.PHONY: graph-png
graph-png:  ## Also render docs/agent_graph.png (uses mermaid.ink — needs network)
	uv run python scripts/visualize_graph.py --png

.PHONY: eval eval-dry-run

eval:  ## Run the M4 evaluation over the golden set (needs Qdrant + OPENAI_API_KEY)
	uv run rn eval

eval-dry-run:  ## Run the eval harness on in-repo fakes (offline, no external services)
	uv run rn eval --dry-run
