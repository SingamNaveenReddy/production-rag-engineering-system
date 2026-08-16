from __future__ import annotations

from app.config import AppConfig
from app.services.rag import RagService


def build_rag_service(config: AppConfig) -> RagService:
    # Heavy ML and provider clients are imported only for the production container.
    # API/unit tests can inject deterministic providers without importing Torch.
    from app.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from app.generation.ollama import OllamaGenerator
    from app.generation.prompts import PromptTemplate
    from app.vectorstore.qdrant_store import QdrantVectorStore

    embedder = SentenceTransformerEmbedder(config.providers.embedding_model)
    store = QdrantVectorStore(
        config.providers.qdrant_url,
        config.providers.qdrant_collection,
        embedder.dimension,
    )
    generator = OllamaGenerator(
        config.providers.ollama_base_url,
        config.providers.llm_model,
        PromptTemplate(config.prompt.answer_file),
    )
    return RagService(
        embedder=embedder,
        store=store,
        generator=generator,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        dense_top_k=config.retrieval.dense_top_k,
        minimum_score=config.retrieval.minimum_score,
        sparse_top_k=config.retrieval.sparse_top_k,
        hybrid_candidate_count=config.retrieval.hybrid_candidate_count,
        rrf_k=config.retrieval.rrf_k,
    )
