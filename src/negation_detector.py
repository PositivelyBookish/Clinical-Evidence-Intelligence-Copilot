"""Negation and uncertainty detection helpers for clinical evidence snippets."""

from __future__ import annotations

import re


NEGATION_TRIGGERS = [
    "without evidence of",
    "no evidence of",
    "negative for",
    "not consistent with",
    "ruled out",
    "denies",
    "denied",
    "absent",
    "no",
]

UNCERTAINTY_TRIGGERS = [
    "pending confirmation",
    "concern for",
    "possible angina",
    "suspected",
    "rule out",
    "may have",
    "could be",
    "possible",
    "likely",
    "r/o",
]

WORD_PATTERN = re.compile(r"\b[\w/-]+\b")


def detect_negation_and_uncertainty(entity, evidence_sentence: str) -> dict:
    """Detect negation and uncertainty cues near the entity mention."""
    entity_text = _resolve_entity_text(entity)
    evidence_text = evidence_sentence or ""

    if not entity_text or not evidence_text.strip():
        return {
            "is_negated": False,
            "is_uncertain": False,
            "negation_trigger": None,
            "uncertainty_trigger": None,
            "status_modifier": "affirmed",
        }

    sentence_lower = evidence_text.lower()
    entity_lower = entity_text.lower()
    match = re.search(re.escape(entity_lower), sentence_lower)
    if not match:
        return {
            "is_negated": False,
            "is_uncertain": False,
            "negation_trigger": None,
            "uncertainty_trigger": None,
            "status_modifier": "affirmed",
        }

    preceding_text = sentence_lower[: match.start()]
    preceding_words = WORD_PATTERN.findall(preceding_text)
    if len(preceding_words) > 8:
        window_words = preceding_words[-8:]
        window_start_word = window_words[0]
        window_start_index = preceding_text.rfind(window_start_word)
        context_window = preceding_text[window_start_index:]
    else:
        context_window = preceding_text

    negation_trigger = _find_trigger(context_window, NEGATION_TRIGGERS)
    uncertainty_trigger = _find_trigger(context_window, UNCERTAINTY_TRIGGERS)

    is_negated = negation_trigger is not None
    is_uncertain = not is_negated and uncertainty_trigger is not None

    if is_negated:
        status_modifier = "negated"
    elif is_uncertain:
        status_modifier = "uncertain"
    else:
        status_modifier = "affirmed"

    return {
        "is_negated": is_negated,
        "is_uncertain": is_uncertain,
        "negation_trigger": negation_trigger,
        "uncertainty_trigger": uncertainty_trigger,
        "status_modifier": status_modifier,
    }


def _resolve_entity_text(entity) -> str:
    """Resolve an entity string from dict or raw text input."""
    if isinstance(entity, dict):
        return entity.get("text") or entity.get("normalized_text") or ""
    return str(entity or "")


def _find_trigger(context_window: str, triggers: list[str]) -> str | None:
    """Return the first matching trigger in the local context window."""
    lowered = context_window.lower()
    for trigger in triggers:
        pattern = re.compile(rf"\b{re.escape(trigger)}\b")
        if pattern.search(lowered):
            return trigger
    return None
