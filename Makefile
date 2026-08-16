.PHONY: setup run test lint ingest evaluate

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

run:
	.venv/bin/uvicorn app.main:app --reload

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

ingest:
	.venv/bin/python scripts/ingest.py $(FILE)

evaluate:
	@echo "Evaluation is intentionally deferred until Phase 5."

