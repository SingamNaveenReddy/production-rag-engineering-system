# Phase 5 architecture

```text
PDF / Markdown / TXT
        |
        v
PyMuPDF / UTF-8 loader -> metadata-preserving chunker
        |
        v
Sentence Transformer embeddings -> Qdrant cosine collection
        |                              |
        v                              v
Dense semantic ranking         BM25 lexical ranking
        |                              |
        +-------- reciprocal rank fusion
                           |
                           v
                 configurable candidate set
                           |
                           v
                 CrossEncoder reranker -> top N
                           |
                           v
                 evidence threshold -> Ollama / Qwen3
                           |
                           v
                 structured answer + selected chunk IDs
                           |
                           v
                 answerability + citation validator
                           |
                 +---------+---------+
                 |                   |
                 v                   v
        validated answer         fail-closed refusal
        |
        v
Programmatically validated citations -> structured API response
                           |
                           v
          golden JSONL runner -> metrics -> threshold gate
                           |
                           v
                   JSON + Markdown reports
```

Provider boundaries isolate embeddings, vector storage, retrieval, reranking, generation, and
validation. Phase 4 asks Ollama for schema-constrained output containing an answerability decision
and selected chunk IDs. The validator rejects missing, invented, or inconsistent IDs and rebuilds
all citation metadata from retrieved chunks. Validation failure returns a refusal with no citations.

Phase 5 evaluates the same query service through a provider-neutral runner. Golden cases declare
expected answerability and sources. The runner measures retrieval recall@k, faithfulness, citation
accuracy, refusal accuracy, and latency, then applies configurable thresholds and emits JSON and
Markdown reports.
