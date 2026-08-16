# Phase 2 architecture

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
                 evidence threshold -> Ollama / Qwen3
        |
        v
Programmatically validated citations -> structured API response
```

Provider boundaries isolate embeddings, vector storage, retrieval, and generation. Phase 2 adds
BM25 sparse retrieval and normalized reciprocal rank fusion without changing ingestion, generation,
or API contracts. CrossEncoder reranking remains a later phase.
