from __future__ import annotations

import httpx

from app.generation.prompts import PromptTemplate
from app.models.schemas import DocumentChunk


class OllamaGenerator:
    def __init__(
        self, base_url: str, model: str, prompt: PromptTemplate, timeout: float = 120
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._prompt = prompt
        self._timeout = timeout

    def generate(self, question: str, context: list[DocumentChunk]) -> str:
        rendered_context = "\n\n".join(
            f"[{chunk.metadata.chunk_id}] {chunk.text}" for chunk in context
        )
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": self._prompt.system},
                    {
                        "role": "user",
                        "content": self._prompt.render(question=question, context=rendered_context),
                    },
                ],
                "options": {"temperature": 0},
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"]).strip()
