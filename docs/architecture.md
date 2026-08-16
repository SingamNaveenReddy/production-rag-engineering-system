# Phase 3 architecture

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
Programmatically validated citations -> structured API response
```

Provider boundaries isolate embeddings, vector storage, retrieval, reranking, and generation.
Phase 3 sends a configurable hybrid candidate set to a Sentence Transformers CrossEncoder and only
passes the configured top N to generation. Retrieval, reranking, generation, and total latency are
reported separately without changing the API response schema.
