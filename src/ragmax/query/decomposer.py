"""Query decomposition for complex multi-part questions."""

from __future__ import annotations

from ragmax.core.protocols import LLMProvider

DECOMPOSE_PROMPT = """Break down the following complex question into simpler sub-questions
that can each be answered independently. Return each sub-question on a new line.
Return ONLY the sub-questions, no numbering or extra text.

Complex question: {query}"""


class QueryDecomposer:
    """Decompose complex queries into simpler sub-queries.

    Useful for multi-hop questions that require information from
    multiple different documents or document sections.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def decompose(self, query: str) -> list[str]:
        """Break a complex query into simpler sub-queries."""
        prompt = DECOMPOSE_PROMPT.format(query=query)
        response = await self.llm.generate([{"role": "user", "content": prompt}])
        sub_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        return sub_queries if sub_queries else [query]
