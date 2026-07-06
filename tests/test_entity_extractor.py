"""Tests for layered entity extraction with graceful fallback."""

from __future__ import annotations

from pathlib import Path

from src.entity_extractor import extract_entities
from src.section_detector import detect_sections
from src.text_cleaner import clean_text


def _load_extraction_result(filename: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    raw_text = (root / "data" / "synthetic_documents" / filename).read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned["clean_text"])
    return extract_entities(cleaned["clean_text"], sections, use_advanced_models=False)


def test_clean_supported_claim_extracts_expected_entities() -> None:
    result = _load_extraction_result("clean_supported_claim.txt")

    assert any(entity["normalized_text"] == "Jordan Carter" for entity in result["patient"])
    assert any(entity["normalized_text"] == "Dr. Elena Morris, MD" for entity in result["provider"])
    assert any(entity["normalized_text"] == "2026-02-14" for entity in result["dates"])
    hypertension_entities = [
        entity for entity in result["diagnoses"] if entity["normalized_text"] == "Hypertension"
    ]
    assert hypertension_entities
    assert hypertension_entities[0]["status_modifier"] == "affirmed"
    assert any(entity["normalized_text"] == "ECG" for entity in result["procedures"])
    assert any(entity["normalized_text"] == "Aspirin" for entity in result["medications"])
    assert any(entity["normalized_text"] == "I10" for entity in result["codes"])
    assert any(entity["normalized_text"] == "93000" for entity in result["codes"])
    assert any(entity["normalized_text"] == "Dr. Elena Morris, MD" for entity in result["signatures"])


def test_negated_diagnosis_still_extracts_pneumonia_entity() -> None:
    result = _load_extraction_result("negated_diagnosis.txt")

    pneumonia_entities = [
        entity for entity in result["diagnoses"] if entity["normalized_text"] == "Pneumonia"
    ]
    assert pneumonia_entities
    assert pneumonia_entities[0]["status_modifier"] == "negated"
    assert pneumonia_entities[0]["is_negated"] is True
    assert pneumonia_entities[0]["trigger"] in {"no evidence of", "ruled out"}


def test_unclear_support_marks_angina_as_uncertain() -> None:
    result = _load_extraction_result("unclear_support.txt")

    angina_entities = [
        entity for entity in result["diagnoses"] if entity["normalized_text"] == "Angina"
    ]
    assert angina_entities
    assert angina_entities[0]["status_modifier"] == "uncertain"
    assert angina_entities[0]["is_uncertain"] is True
    assert angina_entities[0]["trigger"] in {"possible", "concern for", "possible angina"}
