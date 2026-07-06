"""Tests for grounded reviewer summarization."""

from __future__ import annotations

from pathlib import Path

from src.claim_support_checker import check_claim_support
from src.documentation_checker import check_documentation_completeness
from src.entity_extractor import extract_entities
from src.section_detector import detect_sections
from src.summarizer import generate_reviewer_summary
from src.text_cleaner import clean_text


def _build_summary_context(filename: str) -> tuple[str, dict, dict]:
    root = Path(__file__).resolve().parents[1]
    raw_text = (root / "data" / "synthetic_documents" / filename).read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned["clean_text"])
    entities = extract_entities(cleaned["clean_text"], sections, use_advanced_models=False)
    entities["document_metadata"] = {"document_type": "Synthetic clinical note"}
    documentation_result = check_documentation_completeness(
        entities,
        sections,
        cleaned["clean_text"],
    )
    return cleaned["clean_text"], entities, documentation_result


def test_supported_document_summary_is_grounded_and_cautious() -> None:
    clean_text_value, entities, documentation_result = _build_summary_context(
        "clean_supported_claim.txt"
    )
    claim_support_result = check_claim_support(
        {
            "claimed_diagnosis": "Hypertension",
            "claimed_procedure": "ECG",
            "claimed_code": "I10",
        },
        entities,
        clean_text_value,
        detect_sections(clean_text_value),
    )

    result = generate_reviewer_summary(
        entities=entities,
        claim_support_result=claim_support_result,
        documentation_result=documentation_result,
    )

    assert "Document type: Synthetic clinical note." in result["short_summary"]
    assert "Hypertension" in result["short_summary"]
    assert "ECG" in result["short_summary"]
    assert "Supported" in result["short_summary"]
    assert "Aspirin" in result["clinical_evidence_summary"]
    assert "Human review required" in result["human_review_note"]


def test_incomplete_document_summary_acknowledges_missing_information() -> None:
    _, entities, documentation_result = _build_summary_context("incomplete_documentation.txt")

    result = generate_reviewer_summary(
        entities=entities,
        claim_support_result=None,
        documentation_result=documentation_result,
    )

    assert "Document type: Synthetic clinical note." in result["short_summary"]
    assert "Claim support has not been checked yet." in result["claim_support_summary"]
    assert "Documentation completeness is Low" in result["documentation_summary"]
    assert result["risk_flags"]
    assert any("Missing documentation" in flag for flag in result["risk_flags"])
