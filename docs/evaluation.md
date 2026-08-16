# Evaluation status

The golden-dataset runner and regression quality gate are deliberately deferred until Phase 5.
The test suite verifies deterministic ingestion, dense and sparse retrieval behavior, hybrid fusion,
refusal, citations, Qdrant integration, and API contracts.

Phase 2 adds a small deterministic regression benchmark for exact technical identifiers. Its latest
executed output is stored in `evaluation/results/phase2_retrieval.json` and summarized in
`evaluation/results/phase2_retrieval.md`. This fixture isolates the lexical-retrieval behavior; it
is not presented as a production corpus or overall RAG-quality benchmark.
