"""Plain-text file parser."""

from __future__ import annotations

from ragmax.core.models import Document


class TextParser:
    """Parse plain text, markdown, CSV, and similar files."""

    def parse(self, path: str) -> Document:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return Document(content=content, source=path, metadata={"parser": "text"})

    def supports(self, path: str) -> bool:
        return path.lower().endswith((".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml"))
