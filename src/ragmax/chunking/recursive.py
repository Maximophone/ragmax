"""Recursive character text splitter — the workhorse chunker."""

from __future__ import annotations

from ragmax.core.models import Chunk, Document

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    """Split text recursively by trying separators from coarse to fine.

    Mirrors the LangChain RecursiveCharacterTextSplitter algorithm:
    try to split on the coarsest separator first; if any piece is
    still too large, recurse with the next separator.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.content, self.separators)
        merged = self._merge(pieces)
        return [
            Chunk(
                content=text,
                document_id=document.id,
                index=i,
                metadata={
                    **document.metadata,
                    "chunker": "recursive",
                },
            )
            for i, text in enumerate(merged)
            if text.strip()
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the first applicable separator."""
        if not text:
            return []

        final: list[str] = []

        # Find the appropriate separator
        separator = separators[-1]
        remaining_seps = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_seps = []
                break
            if sep in text:
                separator = sep
                remaining_seps = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good: list[str] = []
        for piece in splits:
            if len(piece) < self.chunk_size:
                good.append(piece)
            else:
                if good:
                    merged_text = separator.join(good)
                    final.append(merged_text)
                    good = []
                if remaining_seps:
                    final.extend(self._split(piece, remaining_seps))
                else:
                    final.append(piece)

        if good:
            final.append(separator.join(good))

        return final

    def _merge(self, pieces: list[str]) -> list[str]:
        """Merge small pieces into chunks respecting size and overlap."""
        merged: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len > self.chunk_size and current:
                merged.append(" ".join(current))
                # Keep overlap
                overlap: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > self.chunk_overlap:
                        break
                    overlap.insert(0, p)
                    overlap_len += len(p)
                current = overlap
                current_len = overlap_len

            current.append(piece)
            current_len += piece_len

        if current:
            merged.append(" ".join(current))

        return merged
