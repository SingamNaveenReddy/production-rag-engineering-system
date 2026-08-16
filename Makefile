.PHONY: setup run test lint ingest benchmark benchmark-reranking evaluate

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

run:
	.venv/bin/uvicorn app.main:app --reload

test:
	.venv/bin/python -m pytest

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
