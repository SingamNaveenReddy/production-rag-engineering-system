from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import DocumentPage
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever, tokenize
from tests.fakes import FakeEmbedder, MemoryVectorStore


def _add_document(store, embedder, tmp_path, name: str, text: str, index: int) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    chunks = chunk_pages(
        [DocumentPage(text=text, page=None)],
        document_id=f"doc-{index}",
        path=path,
        chunk_size=50,
        chunk_overlap=5,
    )
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return chunks[0].metadata.chunk_id


def test_tokenizer_preserves_exact_technical_identifiers() -> None:
    assert tokenize("CVE-2026-1234 applies to GPT-5.6 in section 14.3") == [
        "cve-2026-1234",
        "applies",
        "to",
        "gpt-5.6",
        "in",
        "section",
        "14.3",
    ]


def test_hybrid_improves_exact_keyword_query_over_dense_only(tmp_path) -> None:
    embedder = FakeEmbedder()
    store = MemoryVectorStore()
    _add_document(store, embedder, tmp_path, "overview.txt", "General platform overview.", 1)
    expected = _add_document(
        store,
        embedder,
        tmp_path,
        "advisory.txt",
        "Patch guidance for CVE-2026-1234 is available.",
        2,
    )

    dense = DenseRetriever(embedder, store)
    dense_top = dense.retrieve("CVE-2026-1234", top_k=1)
    assert dense_top[0].chunk.metadata.chunk_id != expected

    hybrid = HybridRetriever(
        dense,
        SparseRetriever(store),
        dense_minimum_score=0.01,
    )
    hybrid_top = hybrid.retrieve(
        "CVE-2026-1234", dense_top_k=5, sparse_top_k=5, limit=1
    )
    assert hybrid_top[0].chunk.metadata.chunk_id == expected

