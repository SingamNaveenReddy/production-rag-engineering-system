from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_rag_service
from app.services.rag import RagService
from evaluation.dataset import load_golden_dataset
from evaluation.runner import run_evaluation
from evaluation.schemas import EvaluationReport

router = APIRouter(tags=["evaluation"])
ServiceDependency = Annotated[RagService, Depends(get_rag_service)]


@router.post("/evaluate", response_model=EvaluationReport)
def evaluate(request: Request, service: ServiceDependency) -> EvaluationReport:
    config = request.app.state.config
    dataset_path = config.evaluation.dataset_path
    return run_evaluation(
        service,
        load_golden_dataset(dataset_path),
        dataset_path=dataset_path,
        profile="api-production",
        retrieval_k=config.evaluation.retrieval_k,
        thresholds=config.evaluation.thresholds,
    )

