"""Tests for documentation completeness checking."""

from __future__ import annotations

from pathlib import Path

from src.documentation_checker import check_documentation_completeness
from src.entity_extractor import extract_entities
from src.section_detector import detect_sections
from src.text_cleaner import clean_text


def _build_context(filename: str):
    root = Path(__file__).resolve().parents[1]
    raw_text = (root / "data" / "synthetic_documents" / filename).read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned["clean_text"])
    entities = extract_entities(cleaned["clean_text"], sections, use_advanced_models=False)
    result = check_documentation_completeness(entities, sections, cleaned["clean_text"])
    return result


def test_missing_signature_flags_signature_missing() -> None:
    result = _build_context("missing_signature.txt")

    signature_checks = [check for check in result["checks"] if check["field"] == "Provider signature"]
    assert signature_checks
    assert signature_checks[0]["status"] == "Missing"
    assert "Provider signature" in result["missing_fields"]


def test_incomplete_documentation_has_low_score() -> None:
    result = _build_context("incomplete_documentation.txt")

    assert result["score_label"] == "Low"
    assert result["completeness_score"] < 0.5


def test_clean_supported_claim_has_high_score() -> None:
    result = _build_context("clean_supported_claim.txt")

    assert result["score_label"] == "High"
    assert result["completeness_score"] >= 0.8
