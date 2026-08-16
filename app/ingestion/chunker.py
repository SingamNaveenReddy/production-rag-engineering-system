from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.loaders import DocumentPage
from app.ingestion.metadata import chunk_id_for
from app.models.schemas import DocumentChunk, SourceMetadata

TOKEN_PATTERN = re.compile(r"\S+")


def chunk_pages(
    pages: list[DocumentPage],
    *,
    document_id: str,
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    step = chunk_size - chunk_overlap
    for page in pages:
        tokens = TOKEN_PATTERN.findall(page.text)
        for start in range(0, len(tokens), step):
            selected = tokens[start : start + chunk_size]
            if not selected:
                continue
            chunk_index += 1
            text = " ".join(selected)
            chunk_id = chunk_id_for(document_id, page.page, chunk_index)
            metadata = SourceMetadata(
                document_id=document_id,
                filename=path.name,
                page=page.page,
                section=page.section,
                chunk_id=chunk_id,
                original_source=str(path),
            )
            chunks.append(DocumentChunk(text=text, metadata=metadata))
            if start + chunk_size >= len(tokens):
                break
    return chunks

