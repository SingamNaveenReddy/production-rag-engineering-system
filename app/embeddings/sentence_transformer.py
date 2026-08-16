from __future__ import annotations

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report a vector dimension")
        return dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

