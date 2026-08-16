from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


class DocumentLoadError(ValueError):
    """Raised when an input document cannot be safely parsed."""


@dataclass(frozen=True)
class DocumentPage:
    text: str
    page: int | None
    section: str | None = None


def load_document(path: Path) -> list[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return _load_text(path)
    raise DocumentLoadError(f"Unsupported document type: {suffix or '[none]'}")


def _load_pdf(path: Path) -> list[DocumentPage]:
    try:
        with pymupdf.open(path) as document:
            pages = [
                DocumentPage(text=page.get_text("text").strip(), page=index + 1)
                for index, page in enumerate(document)
                if page.get_text("text").strip()
            ]
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise DocumentLoadError(f"Could not parse PDF: {path.name}") from exc
    if not pages:
        raise DocumentLoadError(f"Document contains no extractable text: {path.name}")
    return pages


def _load_text(path: Path) -> list[DocumentPage]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"Document is not valid UTF-8: {path.name}") from exc
    if not text:
        raise DocumentLoadError(f"Document is empty: {path.name}")
    section = _first_markdown_heading(text) if path.suffix.lower() in {".md", ".markdown"} else None
    return [DocumentPage(text=text, page=None, section=section)]


def _first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            return line.lstrip("# ").strip() or None
    return None
