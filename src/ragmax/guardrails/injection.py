"""Prompt injection detection guardrail."""

from __future__ import annotations

import re

from ragmax.core.models import GuardrailResult, QueryContext

# Patterns that commonly indicate prompt injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)\s+", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"\bDAN\b.*\bmode\b", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an)\s+(unrestricted|unfiltered)", re.IGNORECASE),
]


class InjectionGuardrail:
    """Detect common prompt injection patterns in user queries.

    Uses a combination of regex pattern matching and structural
    analysis to flag potentially malicious inputs.
    """

    def __init__(
        self,
        extra_patterns: list[re.Pattern] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.patterns = _INJECTION_PATTERNS + (extra_patterns or [])
        self.threshold = threshold

    async def check(self, query: str, context: QueryContext) -> GuardrailResult:
        matches: list[str] = []
        for pattern in self.patterns:
            if pattern.search(query):
                matches.append(pattern.pattern)

        # Score based on number of patterns matched
        if not matches:
            return GuardrailResult(passed=True, guardrail_name="injection")

        score = len(matches) / len(self.patterns)
        passed = score < self.threshold

        return GuardrailResult(
            passed=passed,
            guardrail_name="injection",
            message=f"Injection patterns detected ({len(matches)} matches, score={score:.2f})",
            details={"matched_patterns": matches, "score": score},
        )
