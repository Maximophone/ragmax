"""Agentic chunker — LLM-driven intelligent document segmentation.

Uses an LLM to determine optimal chunk boundaries based on
semantic coherence, topic shifts, and information completeness.
The most accurate but also the most expensive chunking strategy.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ragmax.core.models import Chunk, Document
from ragmax.core.protocols import LLMProvider

AGENTIC_PROMPT = """You are a document segmentation expert. Analyze the following text
and identify the natural boundaries where it should be split into coherent chunks.

Each chunk should:
1. Cover a single topic or concept
2. Be self-contained enough to be useful in isolation
3. Be between 100 and 800 words

Return ONLY a JSON array of objects with "start_sentence" and "end_sentence" indices
(0-based), plus a "topic" label for each chunk.

Example: [{{"start_sentence": 0, "end_sentence": 4, "topic": "introduction"}}, ...]

Text (sentences numbered for reference):
{numbered_text}"""


class AgenticChunker:
    """Use an LLM to determine intelligent chunk boundaries.

    Falls back to recursive chunking if no LLM is provided.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        max_sentences_per_call: int = 100,
        **kwargs: Any,
    ) -> None:
        self.llm = llm
        self.max_sentences_per_call = max_sentences_per_call

    def chunk(self, document: Document) -> list[Chunk]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running() or self.llm is None:
            return self._fallback_chunk(document)
        return asyncio.run(self._chunk_async(document))

    async def _chunk_async(self, document: Document) -> list[Chunk]:
        import json
        import re

        sentences = self._split_sentences(document.content)
        if len(sentences) <= 3:
            return [
                Chunk(
                    content=document.content,
                    document_id=document.id,
                    index=0,
                    metadata={**document.metadata, "chunker": "agentic"},
                )
            ]

        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
        prompt = AGENTIC_PROMPT.format(numbered_text=numbered[:8000])

        try:
            response = await self.llm.generate([{"role": "user", "content": prompt}])
            # Extract JSON from response
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                boundaries = json.loads(match.group())
            else:
                return self._fallback_chunk(document)
        except Exception:
            return self._fallback_chunk(document)

        chunks: list[Chunk] = []
        for idx, boundary in enumerate(boundaries):
            start = boundary.get("start_sentence", 0)
            end = boundary.get("end_sentence", len(sentences) - 1)
            topic = boundary.get("topic", "")
            text = " ".join(sentences[start : end + 1])
            if text.strip():
                chunks.append(
                    Chunk(
                        content=text,
                        document_id=document.id,
                        index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "agentic",
                            "topic": topic,
                            "sentence_start": start,
                            "sentence_end": end,
                        },
                    )
                )
        return chunks if chunks else self._fallback_chunk(document)

    def _split_sentences(self, text: str) -> list[str]:
        import re

        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _fallback_chunk(self, document: Document) -> list[Chunk]:
        from ragmax.chunking.recursive import RecursiveChunker

        return RecursiveChunker().chunk(document)
