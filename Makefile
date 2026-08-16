.PHONY: setup run test test-unit test-integration lint ingest benchmark benchmark-reranking evaluate ci

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

run:
	.venv/bin/uvicorn app.main:app --reload

test:
	.venv/bin/python -m pytest

test-unit:
	.venv/bin/python -m pytest tests/unit

test-integration:
	.venv/bin/python -m pytest tests/integration

lint:
	.venv/bin/ruff check .

ingest:
	.venv/bin/python -m scripts.ingest $(FILE)

benchmark:
	.venv/bin/python -m scripts.benchmark_retrieval

benchmark-reranking:
	.venv/bin/python -m scripts.benchmark_reranking

evaluate:
	.venv/bin/python -m evaluation.evaluate

ci: lint test-unit test-integration evaluate
