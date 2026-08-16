from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.rag import RagService
from tests.fakes import FakeEmbedder, FakeGenerator, MemoryVectorStore


def test_phase_one_api(tmp_path: Path) -> None:
    store = MemoryVectorStore()
    service = RagService(
        embedder=FakeEmbedder(),
        store=store,
        generator=FakeGenerator(),
        chunk_size=20,
        chunk_overlap=5,
        dense_top_k=5,
        minimum_score=0.2,
    )
    app = create_app(service=service)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json() == {"status": "ok", "phase": "phase-5"}
        assert "/evaluate" in client.get("/openapi.json").json()["paths"]

        uploaded = client.post(
            "/documents/upload",
            files={
                "file": (
                    "policy.txt",
                    b"Authentication requires a hardware key.",
                    "text/plain",
                )
            },
        )
        assert uploaded.status_code == 201
        document_id = uploaded.json()["document_id"]

        queried = client.post("/query", json={"question": "What authentication is required?"})
        assert queried.status_code == 200
        assert queried.json()["answerable"] is True

        listed = client.get("/documents")
        assert listed.status_code == 200
        assert len(listed.json()) == 1

        deleted = client.delete(f"/documents/{document_id}")
        assert deleted.status_code == 204
