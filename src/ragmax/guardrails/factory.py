"""Factory for creating guardrail instances from names."""

from __future__ import annotations

from ragmax.core.protocols import InputGuardrail, OutputGuardrail


_INPUT_REGISTRY: dict[str, type] = {}
_OUTPUT_REGISTRY: dict[str, type] = {}


def _ensure_registries() -> None:
    if not _INPUT_REGISTRY:
        from ragmax.guardrails.pii import PIIGuardrail
        from ragmax.guardrails.injection import InjectionGuardrail

        _INPUT_REGISTRY["pii"] = PIIGuardrail
        _INPUT_REGISTRY["injection"] = InjectionGuardrail

    if not _OUTPUT_REGISTRY:
        from ragmax.guardrails.hallucination import HallucinationGuardrail
        from ragmax.guardrails.relevance import RelevanceGuardrail

        _OUTPUT_REGISTRY["hallucination"] = HallucinationGuardrail
        _OUTPUT_REGISTRY["relevance"] = RelevanceGuardrail


def create_input_guardrails(names: list[str]) -> list[InputGuardrail]:
    """Create input guardrail instances from a list of names."""
    _ensure_registries()
    guardrails: list[InputGuardrail] = []
    for name in names:
        cls = _INPUT_REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown input guardrail: {name!r}. "
                f"Available: {sorted(_INPUT_REGISTRY.keys())}"
            )
        guardrails.append(cls())
    return guardrails


def create_output_guardrails(names: list[str]) -> list[OutputGuardrail]:
    """Create output guardrail instances from a list of names."""
    _ensure_registries()
    guardrails: list[OutputGuardrail] = []
    for name in names:
        cls = _OUTPUT_REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown output guardrail: {name!r}. "
                f"Available: {sorted(_OUTPUT_REGISTRY.keys())}"
            )
        guardrails.append(cls())
    return guardrails
