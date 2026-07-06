"""Tests for local negation and uncertainty detection."""

from __future__ import annotations

from src.negation_detector import detect_negation_and_uncertainty


def test_no_evidence_of_pneumonia_is_marked_negated() -> None:
    result = detect_negation_and_uncertainty(
        {"text": "pneumonia"},
        "There is no evidence of pneumonia on this exam.",
    )

    assert result["is_negated"] is True
    assert result["is_uncertain"] is False
    assert result["negation_trigger"] == "no evidence of"
    assert result["status_modifier"] == "negated"


def test_possible_angina_is_marked_uncertain() -> None:
    result = detect_negation_and_uncertainty(
        {"text": "angina"},
        "Possible angina is being considered at this time.",
    )

    assert result["is_negated"] is False
    assert result["is_uncertain"] is True
    assert result["uncertainty_trigger"] in {"possible", "possible angina"}
    assert result["status_modifier"] == "uncertain"
