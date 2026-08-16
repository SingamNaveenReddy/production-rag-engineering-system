# Evaluation status

The golden-dataset runner and regression quality gate are deliberately deferred until Phase 5.
The test suite verifies deterministic ingestion, dense and sparse retrieval behavior, hybrid fusion,
refusal, citations, Qdrant integration, and API contracts.

Phase 2 adds a small deterministic regression benchmark for exact technical identifiers. Its latest
executed output is stored in `evaluation/results/phase2_retrieval.json` and summarized in
`evaluation/results/phase2_retrieval.md`. This fixture isolates the lexical-retrieval behavior; it
is not presented as a production corpus or overall RAG-quality benchmark.

Phase 3 adds a controlled benchmark using the actual configured CrossEncoder. It measures whether
the second stage corrects three deliberately misordered candidate pairs and records warm-cache
latency. The latest executed output is in `evaluation/results/phase3_reranking.json` and
`evaluation/results/phase3_reranking.md`. The fixture measures reranking behavior in isolation; it
does not substitute for the later golden-dataset end-to-end evaluation.

Phase 4 adds deterministic coverage for valid programmatic citations, fabricated chunk IDs,
answerable output without citations, inconsistent unanswerable output, generator refusals, and the
Ollama JSON-schema request/response contract. These checks verify enforcement behavior but do not
measure semantic faithfulness; that remains part of the golden evaluation phase.
