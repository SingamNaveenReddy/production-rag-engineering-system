# Decision log

## ADR-001: Programmatic citations

The generator receives chunk identifiers for grounding but does not author citation metadata.
Citations are created from the actual retrieved chunks, preventing fabricated filenames, pages,
and chunk IDs.

## ADR-002: Lazy production dependency construction

The FastAPI application exposes liveness without requiring Qdrant, Ollama, or a model download.
Provider clients initialize on the first provider-dependent request. Tests inject deterministic
providers, avoiding network calls while exercising the same service and API contracts.

## ADR-003: Approximate token chunking in Phase 1

Chunk size and overlap are configurable. Phase 1 uses whitespace-delimited token estimates because
the configured local LLM and embedding tokenizer may differ. A model-specific tokenizer benchmark
is deferred until evaluation data exists.

## ADR-004: BM25 plus reciprocal rank fusion in Phase 2

Sparse retrieval uses BM25 over the canonical chunks persisted in Qdrant. Technical identifiers
containing dots or hyphens remain intact during lexical tokenization. Dense and sparse rankings are
combined using normalized reciprocal rank fusion (RRF), which avoids treating incomparable cosine
and BM25 score scales as if they were equivalent.

For the initial 30-100 document demo corpus, BM25 builds its statistics from stored chunks at query
time. This keeps Qdrant as the source of truth and avoids a second persistence system. A larger
deployment should persist a sparse index or Qdrant sparse vectors and benchmark index freshness,
memory, and latency before migration.

## ADR-005: CrossEncoder as an isolated second-stage provider

The first-stage hybrid retriever favors recall and returns a configurable candidate set. A
`Reranker` interface then scores each query/chunk pair jointly using
`cross-encoder/ms-marco-MiniLM-L-6-v2` and returns only the configured top N. Keeping reranking
behind an interface makes tests deterministic and permits later model/latency comparisons without
changing retrieval, generation, or API code.

CrossEncoder scores replace fusion scores only after candidates pass the hybrid evidence threshold.
Both score sets and stage-level latency measurements remain available in retrieval metadata.
