"""PDF file parser (requires pdfplumber)."""

from __future__ import annotations

from ragmax.core.models import Document
from ragmax.core.utils import require_dependency


class PDFParser:
    """Extract text from PDF files using pdfplumber."""

    def __init__(self) -> None:
        require_dependency("pdfplumber", "parsers")

    def parse(self, path: str) -> Document:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)

        return Document(
            content="\n\n".join(pages),
            source=path,
            metadata={
                "parser": "pdf",
                "num_pages": len(pages),
            },
        )

    def supports(self, path: str) -> bool:
        return path.lower().endswith(".pdf")
