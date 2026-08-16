from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import DocumentLoadError, DocumentPage, load_document


def test_markdown_load_and_metadata_preserving_chunking(tmp_path: Path) -> None:
    source = tmp_path / "policy.md"
    source.write_text("# Authentication\n" + "control " * 12, encoding="utf-8")
    pages = load_document(source)
    chunks = chunk_pages(
        pages,
        document_id="doc-test",
        path=source,
        chunk_size=7,
        chunk_overlap=2,
    )
    assert pages[0].section == "Authentication"
    assert len(chunks) == 3
    assert chunks[0].metadata.filename == "policy.md"
    assert chunks[0].metadata.chunk_id == "doc-test-pna-c0001"
    assert chunks[1].text.split()[:2] == chunks[0].text.split()[-2:]


def test_empty_text_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="empty"):
        load_document(source)


def test_overlap_must_be_smaller_than_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smaller"):
        chunk_pages(
            [DocumentPage(text="one two", page=None)],
            document_id="doc-test",
            path=tmp_path / "a.txt",
            chunk_size=2,
            chunk_overlap=2,
        )

