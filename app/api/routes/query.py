from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag_service
from app.models.schemas import QueryRequest, QueryResult
from app.services.rag import RagService

router = APIRouter(tags=["query"])
ServiceDependency = Annotated[RagService, Depends(get_rag_service)]


@router.post("/query", response_model=QueryResult)
def query(request: QueryRequest, service: ServiceDependency) -> QueryResult:
    return service.query(request.question, request.top_k)
