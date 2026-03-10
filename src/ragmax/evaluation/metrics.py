"""RAG evaluation metrics from the book (Ch. 7).

Covers both retrieval quality metrics and generation quality metrics,
mapped to the 8 failure modes described in the book.
"""

from __future__ import annotations

import math
from typing import Any


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """Recall@K — fraction of relevant docs retrieved.

    Addresses: Missing Content, Missed Top-Ranked Documents.
    """
    if not relevant_ids:
        return 1.0
    top = retrieved_ids[:k] if k else retrieved_ids
    found = len(set(top) & set(relevant_ids))
    return found / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """Precision@K — fraction of retrieved docs that are relevant.

    Addresses: Not in Context (noisy retrieval).
    """
    top = retrieved_ids[:k] if k else retrieved_ids
    if not top:
        return 0.0
    found = len(set(top) & set(relevant_ids))
    return found / len(top)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    Measures ranking quality — rewards placing relevant docs higher.
    """
    top = retrieved_ids[:k] if k else retrieved_ids
    relevant_set = set(relevant_ids)

    dcg = 0.0
    for i, doc_id in enumerate(top):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1)=0

    # Ideal DCG
    ideal_length = min(len(relevant_ids), len(top))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_length))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant result."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def chunk_attribution_score(
    response: str,
    chunks: list[str],
) -> float:
    """Chunk attribution — what fraction of chunks contributed to the response.

    Addresses: Not Extracted failure mode.  A low score means too many
    irrelevant chunks were retrieved.
    """
    if not chunks:
        return 0.0
    response_lower = response.lower()
    response_tokens = set(response_lower.split())

    attributed = 0
    for chunk in chunks:
        chunk_tokens = set(chunk.lower().split())
        overlap = len(response_tokens & chunk_tokens)
        if overlap >= 3:  # At least 3 overlapping tokens
            attributed += 1

    return attributed / len(chunks)


def chunk_utilization_score(
    response: str,
    chunks: list[str],
) -> float:
    """Chunk utilization — how much of each chunk's content was used.

    High utilization means chunks are well-targeted.
    Low utilization suggests chunks are too large or off-topic.
    """
    if not chunks:
        return 0.0
    response_tokens = set(response.lower().split())

    total_utilization = 0.0
    for chunk in chunks:
        chunk_tokens = set(chunk.lower().split())
        if chunk_tokens:
            overlap = len(response_tokens & chunk_tokens) / len(chunk_tokens)
            total_utilization += overlap

    return total_utilization / len(chunks)


def context_adherence_score(
    response: str,
    chunks: list[str],
) -> float:
    """Context adherence — how much of the response is grounded in chunks.

    Addresses: Wrong Format, Incorrect Specificity failure modes.
    High adherence means the response stays close to the source material.
    """
    if not response.strip():
        return 1.0

    response_tokens = response.lower().split()
    if not response_tokens:
        return 1.0

    context_tokens = set()
    for chunk in chunks:
        context_tokens.update(chunk.lower().split())

    # Exclude common stopwords from the check
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "only", "own", "same", "than", "too", "very",
        "just", "because", "if", "when", "while", "where", "how", "what",
        "which", "who", "whom", "this", "that", "these", "those", "i",
        "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their",
    }

    content_tokens = [t for t in response_tokens if t not in stopwords]
    if not content_tokens:
        return 1.0

    grounded = sum(1 for t in content_tokens if t in context_tokens)
    return grounded / len(content_tokens)
