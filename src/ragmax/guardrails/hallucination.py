"""Hallucination detection guardrail for generated outputs."""

from __future__ import annotations

from ragmax.core.models import Chunk, GuardrailResult, QueryContext


class HallucinationGuardrail:
    """Check if a generated response is grounded in the provided chunks.

    Uses a simple token overlap heuristic by default.  For higher accuracy,
    integrate an LLM-based grounding check by subclassing and overriding
    :meth:`check`.

    Parameters
    ----------
    min_grounding_ratio : float
        Minimum fraction of response sentences that must have evidence
        in the context chunks.  Default is 0.5.
    """

    def __init__(self, min_grounding_ratio: float = 0.5) -> None:
        self.min_grounding_ratio = min_grounding_ratio

    async def check(
        self,
        response: str,
        context: QueryContext,
        chunks: list[Chunk],
    ) -> GuardrailResult:
        if not response.strip():
            return GuardrailResult(passed=True, guardrail_name="hallucination")

        context_text = " ".join(c.content.lower() for c in chunks)
        context_tokens = set(context_text.split())

        # Split response into sentences
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if not sentences:
            return GuardrailResult(passed=True, guardrail_name="hallucination")

        grounded = 0
        ungrounded_sentences: list[str] = []

        for sentence in sentences:
            tokens = set(sentence.lower().split())
            if not tokens:
                grounded += 1
                continue
            overlap = len(tokens & context_tokens) / len(tokens)
            if overlap >= 0.3:  # At least 30% token overlap
                grounded += 1
            else:
                ungrounded_sentences.append(sentence)

        ratio = grounded / len(sentences)
        passed = ratio >= self.min_grounding_ratio

        return GuardrailResult(
            passed=passed,
            guardrail_name="hallucination",
            message=f"Grounding ratio: {ratio:.2f} ({grounded}/{len(sentences)} sentences)",
            details={
                "grounding_ratio": ratio,
                "grounded_sentences": grounded,
                "total_sentences": len(sentences),
                "ungrounded_sentences": ungrounded_sentences[:5],
            },
        )
