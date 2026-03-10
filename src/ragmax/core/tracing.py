"""Lightweight tracing / observability for the RAG pipeline."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from ragmax.core.models import Span, Trace

logger = logging.getLogger("ragmax")


class TracingContext:
    """Collects spans for a single pipeline execution.

    Usage::

        tracer = TracingContext()
        with tracer.span("embed_query"):
            embeddings = await embedder.embed([query])
        trace = tracer.build()
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._spans: list[Span] = []
        self._stack: list[Span] = []
        self._metadata: dict[str, Any] = {}

    @contextmanager
    def span(self, name: str, **meta: Any) -> Generator[Span, None, None]:
        """Time a named operation and record it as a span."""
        s = Span(name=name, start_time=time.time(), metadata=meta)
        if self._enabled:
            if self._stack:
                self._stack[-1].children.append(s)
            else:
                self._spans.append(s)
            self._stack.append(s)
        try:
            yield s
        finally:
            s.end_time = time.time()
            if self._enabled and self._stack:
                self._stack.pop()
            logger.debug(
                "span %s completed in %.1fms",
                name,
                s.duration_ms or 0,
            )

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def build(self) -> Trace:
        """Return the completed Trace."""
        return Trace(spans=list(self._spans), metadata=dict(self._metadata))
