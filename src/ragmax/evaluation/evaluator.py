"""High-level RAG evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragmax.evaluation.metrics import (
    chunk_attribution_score,
    chunk_utilization_score,
    context_adherence_score,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass
class EvalSample:
    """A single evaluation sample."""

    query: str
    relevant_ids: list[str] = field(default_factory=list)
    expected_answer: str | None = None


@dataclass
class EvalResult:
    """Results from evaluating a single sample."""

    query: str
    metrics: dict[str, float] = field(default_factory=dict)
    retrieved_ids: list[str] = field(default_factory=list)
    response: str = ""
    chunks: list[str] = field(default_factory=list)


class RAGEvaluator:
    """Evaluate a RAG pipeline across a test dataset.

    Computes all metrics from the book's Ch. 7:
    - Retrieval: recall@k, precision@k, NDCG@k, MRR
    - Generation: chunk attribution, utilization, context adherence

    Usage::

        evaluator = RAGEvaluator()
        samples = [EvalSample(query="...", relevant_ids=["id1", "id2"])]
        results = await evaluator.evaluate(rag_pipeline, samples)
        summary = evaluator.summarize(results)
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    async def evaluate_sample(
        self,
        retrieved_ids: list[str],
        response: str,
        chunks: list[str],
        sample: EvalSample,
    ) -> EvalResult:
        """Evaluate a single query result against ground truth."""
        metrics: dict[str, float] = {}

        # Retrieval metrics
        if sample.relevant_ids:
            metrics["recall@k"] = recall_at_k(retrieved_ids, sample.relevant_ids, self.k)
            metrics["precision@k"] = precision_at_k(retrieved_ids, sample.relevant_ids, self.k)
            metrics["ndcg@k"] = ndcg_at_k(retrieved_ids, sample.relevant_ids, self.k)
            metrics["mrr"] = mrr(retrieved_ids, sample.relevant_ids)

        # Generation metrics
        if response and chunks:
            metrics["chunk_attribution"] = chunk_attribution_score(response, chunks)
            metrics["chunk_utilization"] = chunk_utilization_score(response, chunks)
            metrics["context_adherence"] = context_adherence_score(response, chunks)

        return EvalResult(
            query=sample.query,
            metrics=metrics,
            retrieved_ids=retrieved_ids,
            response=response,
            chunks=chunks,
        )

    @staticmethod
    def summarize(results: list[EvalResult]) -> dict[str, float]:
        """Compute mean metrics across all evaluation samples."""
        if not results:
            return {}

        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for key, value in result.metrics.items():
                all_metrics.setdefault(key, []).append(value)

        return {
            key: sum(values) / len(values) for key, values in all_metrics.items()
        }
