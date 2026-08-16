# Phase 4 architecture

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
```

Provider boundaries isolate embeddings, vector storage, retrieval, reranking, generation, and
validation. Phase 4 asks Ollama for schema-constrained output containing an answerability decision
and selected chunk IDs. The validator rejects missing, invented, or inconsistent IDs and rebuilds
all citation metadata from retrieved chunks. Validation failure returns a refusal with no citations.
