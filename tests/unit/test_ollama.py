from pathlib import Path

from app.generation.ollama import OllamaGenerator
from app.generation.prompts import PromptTemplate
from app.models.schemas import DocumentChunk, SourceMetadata


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "message": {
                "content": (
                    '{"answer":"A key is required.","answerable":true,'
                    '"supporting_chunk_ids":["doc-1-p1-c0001"]}'
                )
            }
        }


def test_ollama_uses_json_schema_and_validates_response(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "answer.yaml"
    prompt_path.write_text(
        "version: test-v1\nsystem: Ground answers.\ntemplate: '{question}\\n{context}'\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.generation.ollama.httpx.post", fake_post)
    chunk = DocumentChunk(
        text="Authentication requires a hardware key.",
        metadata=SourceMetadata(
            document_id="doc-1",
            filename="security.pdf",
            page=1,
            chunk_id="doc-1-p1-c0001",
            original_source="security.pdf",
        ),
    )
    generator = OllamaGenerator(
        "http://localhost:11434", "qwen3:4b", PromptTemplate(prompt_path)
    )

    result = generator.generate("What is required?", [chunk])

    assert result.supporting_chunk_ids == ["doc-1-p1-c0001"]
    assert captured["payload"]["format"]["required"] == [
        "answer",
        "answerable",
        "supporting_chunk_ids",
    ]
    assert captured["payload"]["options"] == {"temperature": 0}
