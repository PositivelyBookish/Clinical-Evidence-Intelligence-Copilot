"""Tests for exportable reviewer report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.documentation_checker import check_documentation_completeness
from src.entity_extractor import extract_entities
from src.export_report import (
    build_documentation_csv,
    build_entities_csv,
    build_review_report,
    build_summary_text,
)
from src.section_detector import detect_sections
from src.summarizer import generate_reviewer_summary
from src.text_cleaner import clean_text


def _build_session_data(filename: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    raw_text = (root / "data" / "synthetic_documents" / filename).read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned["clean_text"])
    entities = extract_entities(cleaned["clean_text"], sections, use_advanced_models=False)
    documentation_result = check_documentation_completeness(
        entities,
        sections,
        cleaned["clean_text"],
    )
    summary = generate_reviewer_summary(
        entities={**entities, "document_metadata": {"document_type": "Claim Review Note"}},
        claim_support_result=None,
        documentation_result=documentation_result,
    )
    return {
        "current_document_name": filename,
        "document_source": "Sample synthetic document",
        "processing_status": f"Loaded bundled synthetic document: {filename}",
        "current_document_result": {
            "file_name": filename,
            "file_type": "txt",
            "success": True,
            "extraction_method": "utf-8 text",
            "text_quality_score": 1.0,
            "warnings": [],
            "errors": [],
        },
        "current_cleaning_result": cleaned,
        "current_section_result": sections,
        "current_entity_result": entities,
        "current_claim_support_result": None,
        "current_documentation_result": documentation_result,
        "current_summary_result": summary,
        "qa_history": [],
    }


def test_build_review_report_handles_missing_optional_steps() -> None:
    session_data = _build_session_data("clean_supported_claim.txt")

    report = build_review_report(session_data)

    assert report["app_name"] == "Clinical Evidence Intelligence Copilot"
    assert report["document_name"] == "clean_supported_claim.txt"
    assert report["claim_support"] is None
    assert report["qa_history"] == []
    assert report["governance"]["synthetic_data_only"] is True
    assert "assessment" in report["detected_sections"]
    assert report["entities"]["all_entities"]


def test_export_helpers_generate_readable_json_csv_and_text() -> None:
    session_data = _build_session_data("missing_signature.txt")
    report = build_review_report(session_data)

    report_json = json.dumps(report, indent=2)
    entities_csv = build_entities_csv(report["entities"])
    documentation_csv = build_documentation_csv(report["documentation_completeness"])
    summary_text = build_summary_text(report)

    assert '"app_name": "Clinical Evidence Intelligence Copilot"' in report_json
    assert "entity,type,section,evidence,confidence,status_modifier,trigger,sources,start_char,end_char" in entities_csv
    assert "field,status,evidence,importance,reviewer_action" in documentation_csv
    assert "Clinical Evidence Intelligence Copilot" in summary_text
    assert "Human Review Note:" in summary_text
