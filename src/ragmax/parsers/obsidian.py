"""Obsidian-aware markdown parser.

Handles all Obsidian-specific syntax:
- YAML frontmatter  → stripped, key fields extracted into metadata
- [[Wikilinks]]     → plain text (just the link label)
- ```dataview```    → stripped entirely (not real content)
- [field:: value]   → inline Obsidian metadata stripped
- #tags in content  → left as-is (informative)
"""

from __future__ import annotations

import re
from typing import Any

from ragmax.core.models import Document

# ── Regex patterns ────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_DATAVIEW_RE = re.compile(r"```dataview\b.*?```", re.DOTALL | re.IGNORECASE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_INLINE_META_RE = re.compile(r"\[[^\[\]]+::[^\[\]]*\]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and return (metadata_dict, remaining_text)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, Any] = {}
    body = text[match.end():]

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(match.group(1))
        if isinstance(raw, dict):
            meta = raw
    except Exception:
        # Fallback: simple line-by-line key: value parsing
        for line in match.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()

    return meta, body


def _wikilink_to_text(match: re.Match) -> str:
    """Replace [[Link|Display]] → Display, [[Link]] → Link."""
    display = match.group(2) or match.group(1)
    # Strip path prefixes like "Folder/Note Name" → "Note Name"
    display = display.split("/")[-1].strip()
    return display


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten and clean frontmatter dict for use as Chunk metadata."""
    clean: dict[str, Any] = {}

    for k, v in meta.items():
        if v is None or v == "":
            continue
        key = k.strip().lower().replace(" ", "_")

        if isinstance(v, list):
            # Unwrap nested lists and wikilinks
            items: list[str] = []
            for item in v:
                if isinstance(item, str):
                    item = _WIKILINK_RE.sub(_wikilink_to_text, item)
                    item = item.strip()
                    if item:
                        items.append(item)
                elif item is not None:
                    items.append(str(item))
            if items:
                clean[key] = items
        elif isinstance(v, str):
            v = _WIKILINK_RE.sub(_wikilink_to_text, v).strip()
            if v:
                clean[key] = v
        else:
            clean[key] = v

    return clean


def clean_obsidian_markdown(text: str) -> str:
    """Clean Obsidian-specific syntax from markdown text."""
    # Strip dataview blocks
    text = _DATAVIEW_RE.sub("", text)
    # Convert wikilinks to plain text
    text = _WIKILINK_RE.sub(_wikilink_to_text, text)
    # Strip inline metadata like [due:: 2024-06-12]
    text = _INLINE_META_RE.sub("", text)
    # Collapse excessive blank lines
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


class ObsidianParser:
    """Parse Obsidian markdown notes into ragmax Documents.

    Handles YAML frontmatter, wikilinks, dataview blocks, and inline
    Obsidian metadata.  Key frontmatter fields (date, tags, participants,
    title) are stored as Document metadata for downstream filtering.
    """

    def parse(self, path: str) -> Document:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()

        # Extract frontmatter
        meta_raw, body = _parse_frontmatter(raw)
        meta = _clean_metadata(meta_raw)

        # Clean body
        content = clean_obsidian_markdown(body)

        # Derive a title from filename if not in frontmatter
        import os
        filename = os.path.splitext(os.path.basename(path))[0]
        if "title" not in meta:
            meta["title"] = filename

        # Add source path as relative for readability
        meta["parser"] = "obsidian"
        meta["filename"] = filename

        return Document(
            content=content,
            source=path,
            metadata=meta,
        )

    def supports(self, path: str) -> bool:
        return path.lower().endswith(".md")
