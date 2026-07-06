"""Tests for claim support checking."""

from __future__ import annotations

from pathlib import Path

from src.claim_support_checker import check_claim_support
from src.entity_extractor import extract_entities
from src.section_detector import detect_sections
from src.text_cleaner import clean_text


def _build_context(filename: str) -> tuple[str, dict, dict]:
    root = Path(__file__).resolve().parents[1]
    raw_text = (root / "data" / "synthetic_documents" / filename).read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned["clean_text"])
    entities = extract_entities(cleaned["clean_text"], sections, use_advanced_models=False)
    return cleaned["clean_text"], sections, entities


def test_clean_supported_claim_returns_supported() -> None:
    clean_text_value, sections, entities = _build_context("clean_supported_claim.txt")

    result = check_claim_support(
        {
            "claimed_diagnosis": "Hypertension",
            "claimed_procedure": "ECG",
            "claimed_code": "I10",
        },
        entities,
        clean_text_value,
        sections,
    )

    assert result["overall_status"] == "Supported"
    assert all(item["status"] == "Supported" for item in result["items"])


def test_unsupported_procedure_returns_not_supported() -> None:
    clean_text_value, sections, entities = _build_context("unsupported_procedure.txt")

    result = check_claim_support(
        {
            "claimed_diagnosis": "",
            "claimed_procedure": "ECG",
            "claimed_code": "",
        },
        entities,
        clean_text_value,
        sections,
    )

    assert result["overall_status"] == "Not Supported"
    assert result["items"][0]["status"] == "Not Supported"


def test_negated_pneumonia_returns_not_supported() -> None:
    clean_text_value, sections, entities = _build_context("negated_diagnosis.txt")

    result = check_claim_support(
        {
            "claimed_diagnosis": "Pneumonia",
            "claimed_procedure": "",
            "claimed_code": "",
        },
        entities,
        clean_text_value,
        sections,
    )

    assert result["overall_status"] == "Not Supported"
    assert result["items"][0]["status"] == "Not Supported"


def test_uncertain_angina_returns_unclear() -> None:
    clean_text_value, sections, entities = _build_context("unclear_support.txt")

    result = check_claim_support(
        {
            "claimed_diagnosis": "Angina",
            "claimed_procedure": "",
            "claimed_code": "",
        },
        entities,
        clean_text_value,
        sections,
    )

    assert result["overall_status"] == "Unclear"
    assert result["items"][0]["status"] == "Unclear"
