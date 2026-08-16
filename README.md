# Production-Grade RAG Engineering System

A production-oriented document question-answering platform. Phase 3 implements metadata-preserving
ingestion, hybrid dense and BM25 retrieval, CrossEncoder reranking, grounded local generation,
programmatic citations, explicit low-evidence refusal, and a typed FastAPI interface.

## Current scope

Implemented through Phase 3:

- PDF, Markdown, and TXT ingestion
- Stable content-derived document IDs and source-aware chunk IDs
- Configurable 700-token chunks with 100-token overlap
- Sentence Transformer embeddings and Qdrant cosine retrieval
- BM25 sparse retrieval that preserves dotted and hyphenated technical identifiers
- Normalized reciprocal rank fusion across dense and sparse rankings
- Configurable hybrid candidate retrieval and CrossEncoder top-N reranking
- Retrieval, reranking, generation, and total latency measurements
- Ollama-backed generation with configurable Qwen3 4B default
- Programmatic citations derived only from retrieved chunks
- Low-evidence refusal
- Upload, ingest, query, list, delete, and health API routes
- Unit and integration tests with deterministic provider substitutes

Golden-dataset evaluation, CI quality gates, Langfuse, and Streamlit are later phases and are not
claimed here.

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

## Configuration

Defaults live in `config/default.yaml`; environment overrides are documented in `.env.example`.
No secrets, private documents, downloaded models, or Qdrant data belong in source control.

## Benchmark status

No benchmark values are reported yet. The golden dataset, comparative retrieval benchmarks, and
quality regression thresholds are Phase 5 work and must be based on executed evaluation runs.

## Limitations

- Approximate whitespace token counts are used for chunking.
- The evidence threshold is a configurable baseline, not a calibrated confidence probability.
- BM25 statistics are currently computed from stored chunks at query time; larger corpora should
  benchmark a persisted sparse index or Qdrant sparse vectors.
- First use downloads the configured embedding and CrossEncoder model; latency results should
  distinguish model initialization from warm inference.
- Ollama and Qdrant must be running for production-provider startup and end-to-end queries.

## Planned phases

1. Citation enforcement and richer answerability validation.
2. Golden-dataset evaluation and regression gates.
3. GitHub Actions, Langfuse-compatible tracing, and a Streamlit demonstration UI.

## Screenshot

API documentation and UI screenshots will be added after the demonstration interface exists.
