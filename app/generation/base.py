from __future__ import annotations

from typing import Protocol

from app.models.schemas import DocumentChunk


class AnswerGenerator(Protocol):
    def generate(self, question: str, context: list[DocumentChunk]) -> str: ...

