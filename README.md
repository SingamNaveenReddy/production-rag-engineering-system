# Production-Grade RAG Engineering System

A production-oriented document question-answering platform. Phase 6 implements metadata-preserving
ingestion, hybrid dense and BM25 retrieval, CrossEncoder reranking, grounded local generation,
strict citations, fail-closed answerability, offline evaluation, and CI-enforced quality gates.

## Current scope

Implemented through Phase 6:

- PDF, Markdown, and TXT ingestion
- Stable content-derived document IDs and source-aware chunk IDs
- Configurable 700-token chunks with 100-token overlap
- Sentence Transformer embeddings and Qdrant cosine retrieval
- BM25 sparse retrieval that preserves dotted and hyphenated technical identifiers
- Normalized reciprocal rank fusion across dense and sparse rankings
- Configurable hybrid candidate retrieval and CrossEncoder top-N reranking
- Retrieval, reranking, generation, and total latency measurements
- Pydantic-schema-constrained Ollama generation output
- Strict validation of every selected chunk ID against the reranked context
- Programmatic filename, page, chunk ID, and supporting-text citations
- Fail-closed refusal for missing, fabricated, or inconsistent evidence
- Golden JSONL schema and provider-neutral offline evaluation runner
- Retrieval recall@k, faithfulness, citation accuracy, refusal accuracy, and latency metrics
- Configurable regression thresholds with non-zero failure exit
- Machine-readable JSON and readable Markdown reports
- Deterministic CI profile plus a configured production-provider profile
- Pull-request CI for lint, unit tests, integration tests, and deterministic RAG evaluation
- CI failure when configured retrieval, grounding, citation, or refusal thresholds regress
- Ollama-backed generation with configurable Qwen3 4B default
- Programmatic citations derived only from retrieved chunks
- Low-evidence refusal
- Upload, ingest, query, list, delete, evaluate, and health API routes
- Unit and integration tests with deterministic provider substitutes

Langfuse and Streamlit are later phases and are not claimed here.

## Architecture

See [docs/architecture.md](docs/architecture.md) and [docs/decisions.md](docs/decisions.md).

## Local setup

Requirements: Python 3.12, Docker, and Ollama.

```bash
make setup
docker compose up -d qdrant
ollama pull qwen3:4b
make run
```

Open `http://localhost:8000/docs` for the generated API documentation.

## Ingest a document

```bash
make ingest FILE=/absolute/path/to/document.pdf
```

Or use the API:

```bash
curl -F 'file=@/absolute/path/to/document.pdf' http://localhost:8000/documents/upload
```

## Query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What controls are required?"}'
```

## Test and lint

```bash
make test
make lint
```

Run the same checks used by CI locally:

```bash
make ci
```

## Phase 2 retrieval benchmark

```bash
make benchmark
```

The latest executed three-query exact-identifier fixture measured dense recall@1 of `0.333` and
hybrid recall@1 of `1.000`, an absolute improvement of `0.667`. This deliberately narrow regression
fixture proves lexical recovery for identifiers the deterministic dense baseline cannot represent;
it is not a production-corpus quality claim. See `evaluation/results/phase2_retrieval.md`.

## Phase 3 reranking benchmark

```bash
make benchmark-reranking
```

The latest executed controlled benchmark used the actual
`cross-encoder/ms-marco-MiniLM-L-6-v2` model. Hybrid-only top-1 accuracy was `0.000`; reranked top-1
accuracy was `1.000`; warm-cache median reranking latency was `12 ms`. The three cases are
deliberately misordered candidate pairs and measure the second stage in isolation, not end-to-end
production RAG quality. See `evaluation/results/phase3_reranking.md`.

## Phase 5 golden evaluation

```bash
make evaluate
```

The latest executed deterministic seed evaluated six manually verified cases. Retrieval recall@5,
lexical citation faithfulness, citation accuracy, and refusal accuracy were all `1.000`; P95 latency
was `0 ms`; the configured quality gate passed. The profile uses deterministic local providers and
a lexical faithfulness proxy, so these are regression-fixture results rather than production-model
quality claims. See `evaluation/results/golden_evaluation.md`.

## Phase 6 CI quality gate

`.github/workflows/quality-gate.yml` runs on every pull request and on pushes to `main`. It installs
the project on Python 3.12, then runs Ruff, the unit suite, the integration suite, and the six-case
deterministic evaluation. The evaluator reads thresholds from `config/default.yaml` and exits
non-zero when any required metric falls below its configured minimum, so the workflow fails.

The CI evaluation is deliberately network-free: it uses in-memory Qdrant plus deterministic local
embedding, reranking, and generation substitutes. It does not require Ollama, an external Qdrant
service, model downloads, API credentials, or the optional Ragas dependency.

## Configuration

Defaults live in `config/default.yaml`; environment overrides are documented in `.env.example`.
No secrets, private documents, downloaded models, or Qdrant data belong in source control.

## Benchmark status

Phase 2 and Phase 3 contain narrowly scoped retrieval and reranking fixtures. Phase 5 adds an
executed six-case golden seed and threshold gate; Phase 6 enforces that seed on pull requests.
Expanding it to a 100-question manually verified corpus and running the configured production
providers remain required before publishing broader quality claims.

## Limitations

- Approximate whitespace token counts are used for chunking.
- The evidence threshold is a configurable baseline, not a calibrated confidence probability.
- BM25 statistics are currently computed from stored chunks at query time; larger corpora should
  benchmark a persisted sparse index or Qdrant sparse vectors.
- First use downloads the configured embedding and CrossEncoder model; latency results should
  distinguish model initialization from warm inference.
- Ollama and Qdrant must be running for production-provider startup and end-to-end queries.
- Deterministic faithfulness is lexical citation-token coverage, not an LLM-judged Ragas score.

## Planned phases

1. Langfuse-compatible tracing.
2. Streamlit demonstration UI.

## Screenshot

API documentation and UI screenshots will be added after the demonstration interface exists.
