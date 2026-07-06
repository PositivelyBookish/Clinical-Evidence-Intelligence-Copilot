"""Export helpers for reviewer report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


APP_NAME = "Clinical Evidence Intelligence Copilot"


def build_review_report(session_data: dict) -> dict:
    """Build a complete exportable reviewer report from Streamlit session data."""
    document_result = session_data.get("current_document_result", {}) or {}
    section_result = session_data.get("current_section_result", {}) or {}
    entity_result = session_data.get("current_entity_result", {}) or {}
    claim_support = session_data.get("current_claim_support_result")
    documentation = session_data.get("current_documentation_result") or {}
    qa_history = session_data.get("qa_history", []) or []
    reviewer_summary = session_data.get("current_summary_result") or {}

    extraction_metadata = {
        "document_source": session_data.get("document_source", "None"),
        "processing_status": session_data.get("processing_status", "Unknown"),
        "file_name": document_result.get("file_name", session_data.get("current_document_name", "")),
        "file_type": document_result.get("file_type", ""),
        "success": document_result.get("success", False),
        "extraction_method": document_result.get("extraction_method", "none"),
        "text_quality_score": document_result.get("text_quality_score", 0.0),
        "warnings": document_result.get("warnings", []),
        "errors": document_result.get("errors", []),
        "normalization_notes": (session_data.get("current_cleaning_result", {}) or {}).get(
            "normalization_notes", []
        ),
    }

    return {
        "app_name": APP_NAME,
        "document_name": session_data.get("current_document_name", "No document loaded"),
        "extraction_metadata": extraction_metadata,
        "detected_sections": section_result.get("sections", {}),
        "entities": entity_result,
        "claim_support": claim_support,
        "documentation_completeness": documentation,
        "qa_history": qa_history,
        "reviewer_summary": reviewer_summary,
        "governance": {
            "synthetic_data_only": True,
            "human_review_required": True,
            "not_a_medical_diagnosis_tool": True,
            "not_a_final_payment_decision_tool": True,
        },
    }


def build_export_payload(
    document_name: str,
    summary: dict,
    claim_support_result: dict | None,
    documentation_result: dict | None,
) -> dict:
    """Build a compact grounded JSON export payload."""
    return {
        "document_name": document_name,
        "reviewer_summary": summary,
        "claim_support_result": claim_support_result,
        "documentation_result": documentation_result,
    }


def build_entities_csv(entities: dict) -> str:
    """Flatten extracted entities into CSV text."""
    rows = []
    for entity in (entities or {}).get("all_entities", []):
        rows.append(
            {
                "entity": entity.get("normalized_text", entity.get("text", "")),
                "type": entity.get("entity_type", ""),
                "section": entity.get("section", ""),
                "evidence": entity.get("evidence", ""),
                "confidence": entity.get("confidence", ""),
                "status_modifier": entity.get("status_modifier", "affirmed"),
                "trigger": entity.get("trigger", ""),
                "sources": ", ".join(entity.get("sources", [entity.get("source", "")])),
                "start_char": entity.get("start_char", ""),
                "end_char": entity.get("end_char", ""),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def build_documentation_csv(documentation_result: dict) -> str:
    """Flatten documentation checklist into CSV text."""
    rows = (documentation_result or {}).get("checks", [])
    return pd.DataFrame(rows).to_csv(index=False)


def build_summary_text(report: dict) -> str:
    """Build a plain-text reviewer summary download."""
    summary = report.get("reviewer_summary", {}) or {}
    extraction = report.get("extraction_metadata", {}) or {}
    lines = [
        APP_NAME,
        f"Document: {report.get('document_name', 'No document loaded')}",
        f"Status: {extraction.get('processing_status', 'Unknown')}",
        "",
        "Short Summary:",
        summary.get("short_summary", "No document summary available yet."),
        "",
        "Clinical Evidence Summary:",
        summary.get("clinical_evidence_summary", "Insufficient evidence found."),
        "",
        "Claim Support Summary:",
        summary.get("claim_support_summary", "Claim support has not been checked yet."),
        "",
        "Documentation Summary:",
        summary.get("documentation_summary", "Documentation completeness has not been evaluated yet."),
        "",
        "Human Review Note:",
        summary.get("human_review_note", "Human review required."),
    ]

    risk_flags = summary.get("risk_flags", [])
    if risk_flags:
        lines.extend(["", "Risk Flags:"])
        lines.extend([f"- {flag}" for flag in risk_flags])
    return "\n".join(lines)


def export_placeholder_report(output_path: str | Path, content: str) -> Path:
    """Write a simple text report as a temporary export placeholder."""
    path = Path(output_path)
    path.write_text(content, encoding="utf-8")
    return path


def export_json_report(output_path: str | Path, payload: dict) -> Path:
    """Write a JSON reviewer report payload to disk."""
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
