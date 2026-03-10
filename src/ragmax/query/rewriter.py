"""LLM-based query rewriting for improved retrieval."""

from __future__ import annotations

from ragmax.core.models import QueryContext
from ragmax.core.protocols import LLMProvider

REWRITE_PROMPT = """Rewrite the following user query to be more specific and detailed
for document retrieval. The rewritten query should capture the same intent
but use more precise language. Return ONLY the rewritten query.

Original query: {query}"""

CONVERSATIONAL_PROMPT = """Given the conversation history and the latest user message,
rewrite the user's message as a standalone query that captures the full intent.
Return ONLY the standalone query.

Conversation history:
{history}

Latest message: {query}"""


class QueryRewriter:
    """Rewrite queries for improved retrieval.

    Supports:
    - Simple expansion: add detail/specificity to short queries
    - Conversational: resolve coreferences from conversation history
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def rewrite(self, context: QueryContext) -> str:
        """Rewrite the query, choosing strategy based on context."""
        if context.conversation_history:
            return await self._conversational_rewrite(context)
        return await self._simple_rewrite(context.query)

    async def _simple_rewrite(self, query: str) -> str:
        prompt = REWRITE_PROMPT.format(query=query)
        return (await self.llm.generate([{"role": "user", "content": prompt}])).strip()

    async def _conversational_rewrite(self, context: QueryContext) -> str:
        history = "\n".join(
            f"{m['role']}: {m['content']}" for m in context.conversation_history[-6:]
        )
        prompt = CONVERSATIONAL_PROMPT.format(history=history, query=context.query)
        return (await self.llm.generate([{"role": "user", "content": prompt}])).strip()
