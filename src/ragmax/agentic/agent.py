"""Agentic RAG — an LLM agent that iteratively retrieves and reasons.

Implements the Agentic RAG pattern from the book: the LLM decides
when to retrieve more information, refine its search, or generate
the final answer.  Supports multi-hop reasoning over the knowledge base.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ragmax.core.models import GenerationResult, QueryContext, SearchResult, Trace
from ragmax.core.protocols import LLMProvider, Retriever
from ragmax.core.tracing import TracingContext

logger = logging.getLogger("ragmax.agent")

AGENT_SYSTEM_PROMPT = """You are a research assistant with access to a knowledge base.
To answer the user's question, you can search the knowledge base by calling the search tool.

You MUST respond with a JSON object in one of these formats:

1. To search: {{"action": "search", "query": "your search query"}}
2. To give the final answer: {{"action": "answer", "answer": "your final answer"}}

Think step by step. If the first search doesn't give you enough information,
refine your query and search again. You have a maximum of {max_steps} search steps."""


class RAGAgent:
    """An agentic RAG system that iteratively retrieves and reasons.

    Parameters
    ----------
    retriever : Retriever
        The retriever to use for searching the knowledge base.
    llm : LLMProvider
        The LLM to use for reasoning and answer generation.
    max_steps : int
        Maximum number of retrieval steps before forcing an answer.
    top_k : int
        Number of results to retrieve per search step.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        max_steps: int = 5,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.top_k = top_k

    async def query(
        self,
        question: str,
        context: QueryContext | None = None,
        filters: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Run the agentic RAG loop."""
        tracer = TracingContext()
        all_chunks: list[SearchResult] = []
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT.format(max_steps=self.max_steps),
            },
            {"role": "user", "content": question},
        ]

        for step in range(self.max_steps):
            with tracer.span(f"agent_step_{step}"):
                # Get LLM decision
                with tracer.span("llm_reason"):
                    response = await self.llm.generate(messages)

                # Parse action
                action = self._parse_action(response)

                if action["action"] == "answer":
                    return GenerationResult(
                        answer=action.get("answer", response),
                        chunks_used=[r.chunk for r in all_chunks],
                        metadata={
                            "agent_steps": step + 1,
                            "total_chunks_retrieved": len(all_chunks),
                        },
                        trace=tracer.build(),
                    )

                if action["action"] == "search":
                    search_query = action.get("query", question)
                    with tracer.span("retrieve", query=search_query):
                        results = await self.retriever.retrieve(
                            search_query, top_k=self.top_k, filters=filters
                        )
                        all_chunks.extend(results)

                    # Feed results back to the agent
                    context_text = "\n\n".join(
                        f"[Result {i+1}]: {r.chunk.content}"
                        for i, r in enumerate(results)
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Search results:\n{context_text}\n\nBased on these results, "
                            f"either search again or provide your final answer.",
                        }
                    )

        # Force an answer after max steps
        messages.append(
            {
                "role": "user",
                "content": "You've reached the maximum number of search steps. "
                "Please provide your best answer based on all the information gathered.",
            }
        )
        with tracer.span("final_generation"):
            final = await self.llm.generate(messages)

        action = self._parse_action(final)
        answer = action.get("answer", final)

        return GenerationResult(
            answer=answer,
            chunks_used=[r.chunk for r in all_chunks],
            metadata={
                "agent_steps": self.max_steps,
                "total_chunks_retrieved": len(all_chunks),
                "forced_answer": True,
            },
            trace=tracer.build(),
        )

    def _parse_action(self, response: str) -> dict[str, Any]:
        """Parse the agent's JSON response."""
        import re

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        # If parsing fails, treat the whole response as an answer
        return {"action": "answer", "answer": response}
