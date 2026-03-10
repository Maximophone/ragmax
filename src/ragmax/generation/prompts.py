"""Default prompt templates for RAG generation."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer. If the context does not contain
enough information to answer the question, say so clearly — do not make up information.
Be concise and precise."""

DEFAULT_RAG_PROMPT = """Context:
{context}

Question: {query}

Answer based on the context above. If the information is not in the context, say "I don't have enough information to answer this question."
"""

MULTI_QUERY_SYSTEM = """You are a helpful assistant that generates search queries.
Given the user's question, generate diverse reformulations that capture
different aspects and phrasings of the same information need."""

HYDE_SYSTEM = """You are a helpful assistant. Write a short, factual passage
that would answer the given question. Write ONLY the passage content."""


def build_rag_messages(
    query: str,
    context_chunks: list[str],
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> list[dict[str, str]]:
    """Build the message list for a RAG generation call."""
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    template = user_template or DEFAULT_RAG_PROMPT

    context = "\n\n---\n\n".join(context_chunks)
    user_content = template.format(context=context, query=query)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
