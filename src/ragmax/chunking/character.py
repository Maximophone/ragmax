"""Simple character-based chunker."""

from __future__ import annotations

from ragmax.core.models import Chunk, Document


class CharacterChunker:
    """Split text into fixed-size character chunks with overlap.

    This is the simplest chunking strategy — fast and deterministic,
    but may cut mid-word or mid-sentence.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separator: str = "",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end]
            if segment.strip():
                chunks.append(
                    Chunk(
                        content=segment,
                        document_id=document.id,
                        index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "character",
                            "char_start": start,
                            "char_end": min(end, len(text)),
                        },
                    )
                )
                idx += 1
            step = self.chunk_size - self.chunk_overlap
            start += max(step, 1)  # Guarantee forward progress

        return chunks
