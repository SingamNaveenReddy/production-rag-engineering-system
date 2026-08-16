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
