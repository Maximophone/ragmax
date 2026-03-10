"""Auto-selecting parser registry."""

from __future__ import annotations

import os
from typing import Any

from ragmax.core.models import Document
from ragmax.core.protocols import Parser
from ragmax.core.exceptions import ParsingError


class ParserRegistry:
    """Routes files to the correct parser based on extension.

    Built-in parsers are registered automatically.  Users can register
    custom parsers with :meth:`register`.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        from ragmax.parsers.text import TextParser

        text = TextParser()
        for ext in (".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml"):
            self._parsers[ext] = text

        try:
            from ragmax.parsers.pdf import PDFParser

            self._parsers[".pdf"] = PDFParser()
        except ImportError:
            pass

        try:
            from ragmax.parsers.docx import DocxParser

            self._parsers[".docx"] = DocxParser()
        except ImportError:
            pass

        try:
            from ragmax.parsers.html import HTMLParser

            for ext in (".html", ".htm"):
                self._parsers[ext] = HTMLParser()
        except ImportError:
            pass

    def register(self, extension: str, parser: Parser) -> None:
        """Register a parser for the given file extension (including dot)."""
        self._parsers[extension.lower()] = parser

    def parse(self, path: str, **kwargs: Any) -> Document:
        """Parse a file, auto-selecting the parser by extension."""
        ext = os.path.splitext(path)[1].lower()
        parser = self._parsers.get(ext)
        if parser is None:
            raise ParsingError(
                f"No parser registered for extension '{ext}'. "
                f"Registered: {sorted(self._parsers.keys())}"
            )
        return parser.parse(path)

    def parse_directory(self, directory: str, recursive: bool = True) -> list[Document]:
        """Parse all supported files in a directory."""
        docs: list[Document] = []
        for root, _dirs, files in os.walk(directory):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self._parsers:
                    full = os.path.join(root, fname)
                    docs.append(self.parse(full))
            if not recursive:
                break
        return docs
