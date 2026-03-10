"""PII detection guardrail for input queries."""

from __future__ import annotations

import re

from ragmax.core.models import GuardrailResult, QueryContext

# Common PII patterns
_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


class PIIGuardrail:
    """Detect and optionally redact PII in user queries.

    Parameters
    ----------
    block : bool
        If True, block queries containing PII.
        If False, redact PII and pass the modified query through.
    patterns : dict[str, re.Pattern] | None
        Custom regex patterns to check.  Merged with defaults.
    """

    def __init__(
        self,
        block: bool = False,
        patterns: dict[str, re.Pattern] | None = None,
    ) -> None:
        self.block = block
        self.patterns = {**_PATTERNS, **(patterns or {})}

    async def check(self, query: str, context: QueryContext) -> GuardrailResult:
        found: dict[str, list[str]] = {}
        for name, pattern in self.patterns.items():
            matches = pattern.findall(query)
            if matches:
                found[name] = matches

        if not found:
            return GuardrailResult(passed=True, guardrail_name="pii")

        if self.block:
            return GuardrailResult(
                passed=False,
                guardrail_name="pii",
                message=f"PII detected: {', '.join(found.keys())}",
                details={"matches": found},
            )

        # Redact mode
        redacted = query
        for name, pattern in self.patterns.items():
            redacted = pattern.sub(f"[{name.upper()}_REDACTED]", redacted)

        return GuardrailResult(
            passed=True,
            guardrail_name="pii",
            message="PII detected and redacted",
            details={"matches": found},
            modified_text=redacted,
        )
