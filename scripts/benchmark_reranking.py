from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from app.models.schemas import DocumentChunk, ScoredChunk, SourceMetadata
from app.reranking.cross_encoder import CrossEncoderReranker

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _candidate(case_index: int, rank: int, text: str) -> ScoredChunk:
    chunk_id = f"reranking-{case_index}-c{rank}"
    return ScoredChunk(
        chunk=DocumentChunk(
            text=text,
            metadata=SourceMetadata(
                document_id=f"reranking-{case_index}",
                filename=f"case-{case_index}.txt",
                chunk_id=chunk_id,
                original_source="controlled-reranking-benchmark",
            ),
        ),
        score=1.0 - rank / 10,
    )


def benchmark_cases() -> list[dict[str, object]]:
    definitions = [
        (
            "What authentication device is mandatory for employees?",
            "Employee directory devices are cataloged for annual inventory.",
            "Employees must authenticate with a FIDO2 hardware security key.",
        ),
        (
            "How long must audit logs be retained?",
            "Audit logs retention is discussed in the terminology reference without a duration.",
            "Audit records must be retained for seven years.",
        ),
        (
            "What action is required after a critical vulnerability is found?",
            "Critical vulnerability reports are filed by the compliance office.",
            "Apply the security patch within 24 hours of finding a critical vulnerability.",
        ),
    ]
    cases: list[dict[str, object]] = []
    for index, (query, distractor, relevant) in enumerate(definitions, start=1):
        candidates = [
            _candidate(index, 1, distractor),
            _candidate(index, 2, relevant),
        ]
        cases.append(
            {
                "query": query,
                "candidates": candidates,
                "expected_chunk_id": candidates[1].chunk.metadata.chunk_id,
            }
        )
    return cases


def main() -> None:
    model_started = perf_counter()
    reranker = CrossEncoderReranker(MODEL_NAME)
    model_load_ms = int((perf_counter() - model_started) * 1000)
    details: list[dict[str, object]] = []
    hybrid_hits = 0
    reranked_hits = 0
    latencies: list[int] = []
    for case in benchmark_cases():
        candidates = case["candidates"]
        hybrid_id = candidates[0].chunk.metadata.chunk_id
        started = perf_counter()
        reranked = reranker.rerank(case["query"], candidates, top_k=1)
        latency_ms = int((perf_counter() - started) * 1000)
        latencies.append(latency_ms)
        reranked_id = reranked[0].chunk.metadata.chunk_id
        hybrid_hit = hybrid_id == case["expected_chunk_id"]
        reranked_hit = reranked_id == case["expected_chunk_id"]
        hybrid_hits += hybrid_hit
        reranked_hits += reranked_hit
        details.append(
            {
                "query": case["query"],
                "expected_chunk_id": case["expected_chunk_id"],
                "hybrid_chunk_id": hybrid_id,
                "reranked_chunk_id": reranked_id,
                "hybrid_hit": hybrid_hit,
                "reranked_hit": reranked_hit,
                "reranker_score": reranked[0].score,
                "reranking_latency_ms": latency_ms,
            }
        )

    count = len(details)
    result = {
        "benchmark": "phase3-controlled-cross-encoder-reranking",
        "model": MODEL_NAME,
        "query_count": count,
        "metric": "top1_accuracy",
        "hybrid_top1_accuracy": hybrid_hits / count,
        "hybrid_plus_reranker_top1_accuracy": reranked_hits / count,
        "absolute_improvement": (reranked_hits - hybrid_hits) / count,
        "model_load_ms": model_load_ms,
        "median_reranking_latency_ms": sorted(latencies)[count // 2],
        "cases": details,
    }
    output_directory = Path("evaluation/results")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "phase3_reranking.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    markdown = (
        "# Phase 3 controlled CrossEncoder reranking benchmark\n\n"
        f"Model: `{MODEL_NAME}`\n\n"
        f"Queries: {count}\n\n"
        f"- Hybrid top-1 accuracy: {result['hybrid_top1_accuracy']:.3f}\n"
        "- Hybrid + reranker top-1 accuracy: "
        f"{result['hybrid_plus_reranker_top1_accuracy']:.3f}\n"
        f"- Absolute improvement: {result['absolute_improvement']:.3f}\n"
        f"- Median reranking latency: {result['median_reranking_latency_ms']} ms\n\n"
        "This controlled benchmark tests whether the CrossEncoder can correct deliberately "
        "misordered candidate pairs. It is not an end-to-end production-corpus benchmark.\n"
    )
    (output_directory / "phase3_reranking.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
