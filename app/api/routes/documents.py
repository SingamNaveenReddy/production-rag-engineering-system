from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_rag_service
from app.ingestion.loaders import DocumentLoadError
from app.models.schemas import DocumentSummary, IngestPathRequest
from app.services.rag import DuplicateDocumentError, RagService

router = APIRouter(prefix="/documents", tags=["documents"])
ServiceDependency = Annotated[RagService, Depends(get_rag_service)]


def _ingest_or_400(service: RagService, path: Path) -> DocumentSummary:
    try:
        return service.ingest(path)
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (DocumentLoadError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/upload", response_model=DocumentSummary, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File()], service: ServiceDependency
) -> DocumentSummary:
    safe_name = Path(file.filename or "upload").name
    with TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / safe_name
        with path.open("wb") as destination:
            while content := await file.read(1024 * 1024):
                destination.write(content)
        return _ingest_or_400(service, path)


@router.post("/ingest", response_model=DocumentSummary, status_code=status.HTTP_201_CREATED)
def ingest_document(
    request: IngestPathRequest, service: ServiceDependency
) -> DocumentSummary:
    return _ingest_or_400(service, Path(request.path))


@router.get("", response_model=list[DocumentSummary])
def list_documents(service: ServiceDependency) -> list[DocumentSummary]:
    return service.list_documents()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, service: ServiceDependency) -> None:
    if not service.delete_document(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
