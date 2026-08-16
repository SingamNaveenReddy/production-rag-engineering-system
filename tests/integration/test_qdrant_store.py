from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import DocumentPage
from app.vectorstore.qdrant_store import QdrantVectorStore
from tests.fakes import FakeEmbedder


def test_qdrant_upsert_search_list_and_delete(tmp_path) -> None:
    embedder = FakeEmbedder()
    source = tmp_path / "policy.txt"
    source.write_text("Authentication requires a hardware key.", encoding="utf-8")
    chunks = chunk_pages(
        [DocumentPage(text=source.read_text(encoding="utf-8"), page=None)],
        document_id="doc-integration",
        path=source,
        chunk_size=20,
        chunk_overlap=5,
    )
    store = QdrantVectorStore(":memory:", "test-documents", embedder.dimension)
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))

    results = store.search(embedder.embed_query("authentication"), limit=5)
    assert results[0].chunk.metadata.chunk_id == "doc-integration-pna-c0001"
    assert store.list_documents()[0].chunk_count == 1
    assert store.delete_document("doc-integration") is True
    assert store.contains_document("doc-integration") is False
