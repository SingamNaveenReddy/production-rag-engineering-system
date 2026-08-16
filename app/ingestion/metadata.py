from __future__ import annotations

import hashlib
from pathlib import Path


def document_id_for(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"doc-{digest[:20]}"


def chunk_id_for(document_id: str, page: int | None, index: int) -> str:
    page_label = f"p{page}" if page is not None else "pna"
    return f"{document_id}-{page_label}-c{index:04d}"

