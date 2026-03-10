"""Sentence-aware chunker."""

from __future__ import annotations

import re

from ragmax.core.models import Chunk, Document

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker:
    """Split text on sentence boundaries and group into chunks.

    Uses regex-based sentence detection by default. If spaCy is
    installed and *use_spacy* is True, uses spaCy's sentence
    segmentation for much higher accuracy.
    """

    def __init__(
        self,
        max_sentences: int = 8,
        chunk_overlap_sentences: int = 1,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.max_sentences = max_sentences
        self.chunk_overlap_sentences = chunk_overlap_sentences
        self.use_spacy = use_spacy
        self._nlp = None
        if use_spacy:
            try:
                import spacy

                self._nlp = spacy.load(spacy_model)
            except (ImportError, OSError):
                self._nlp = None

    def _split_sentences(self, text: str) -> list[str]:
        if self._nlp is not None:
            doc = self._nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(sentences):
            end = min(start + self.max_sentences, len(sentences))
            group = sentences[start:end]
            text = " ".join(group)
            if text.strip():
                chunks.append(
                    Chunk(
                        content=text,
                        document_id=document.id,
                        index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "sentence",
                            "sentence_start": start,
                            "sentence_end": end,
                        },
                    )
                )
                idx += 1
            new_start = end - self.chunk_overlap_sentences
            if new_start <= start:
                new_start = end  # Guarantee forward progress
            start = new_start

        return chunks
