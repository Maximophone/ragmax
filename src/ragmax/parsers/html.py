"""HTML file parser (requires beautifulsoup4)."""

from __future__ import annotations

from ragmax.core.models import Document
from ragmax.core.utils import require_dependency


class HTMLParser:
    """Extract text from HTML files using BeautifulSoup."""

    def __init__(self) -> None:
        require_dependency("bs4", "parsers")

    def parse(self, path: str) -> Document:
        from bs4 import BeautifulSoup

        with open(path, encoding="utf-8", errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return Document(
            content=text,
            source=path,
            metadata={"parser": "html"},
        )

    def supports(self, path: str) -> bool:
        return path.lower().endswith((".html", ".htm"))
