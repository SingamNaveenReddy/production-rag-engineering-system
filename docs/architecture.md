# Phase 1 architecture

```text
PDF / Markdown / TXT
        |
        v
PyMuPDF / UTF-8 loader -> metadata-preserving chunker
        |
        v
Sentence Transformer embeddings -> Qdrant cosine collection
        |
        v
Dense query retrieval -> evidence threshold -> Ollama / Qwen3
        |
        v
Programmatically validated citations -> structured API response
```

Provider boundaries isolate embeddings, vector storage, and generation so later phases can add
hybrid retrieval or remote LLM providers without rewriting ingestion or API contracts.

