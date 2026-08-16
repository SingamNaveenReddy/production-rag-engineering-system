from fastapi import Request

from app.container import build_rag_service
from app.services.rag import RagService


def get_rag_service(request: Request) -> RagService:
    service: RagService | None = request.app.state.rag_service
    if service is None:
        service = build_rag_service(request.app.state.config)
        request.app.state.rag_service = service
    return service
