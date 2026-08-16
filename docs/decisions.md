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

## ADR-006: Fail-closed structured grounding in Phase 4

The generator returns a validated `GeneratedAnswer` containing `answer`, `answerable`, and
`supporting_chunk_ids`. Ollama receives the Pydantic JSON schema through its structured-output
`format` field. The model may select evidence IDs but never supplies filenames, pages, or supporting
text.

The citation validator accepts only IDs present in the exact reranked context. Answerable output
without evidence, fabricated IDs, or citations attached to an unanswerable result are rejected.
Rejection is converted to a standard refusal with no citations, and the reason is recorded in
retrieval metadata. This favors false refusals over unsupported answers.

## ADR-007: Two evaluation profiles in Phase 5

The evaluation runner accepts any query engine and faithfulness scorer. The default deterministic
profile uses the real ingestion, Qdrant, hybrid retrieval, answerability, and citation-validation
service with stable local embedding and extractive-generation providers. This makes the small
quality gate reproducible and independent of network services.

Deterministic faithfulness is explicitly labeled as lexical token coverage against validated
citation text. It is a CI-safe proxy, not an LLM-judged Ragas score. The production profile runs the
configured Qdrant, Sentence Transformer, CrossEncoder, and Ollama providers; a release evaluation
should inject an explicit LLM-based faithfulness scorer such as Ragas. Reports always record the
profile and faithfulness method so the two cannot be confused.

## ADR-008: One deterministic pull-request quality gate in Phase 6

Pull requests run lint, unit tests, integration tests, and the small deterministic evaluation in one
GitHub Actions job on Python 3.12. Keeping the checks in one job avoids repeated installation of the
large base dependency set and gives branch protection one unambiguous required status check.

The workflow grants only `contents: read` and uses the `pull_request` event, not
`pull_request_target`, so untrusted change code does not receive write permissions or repository
secrets. Evaluation uses the checked-in thresholds and network-free deterministic profile. The
existing evaluator's non-zero regression exit is the sole source of gate truth, preventing CI logic
from duplicating metric calculations.
