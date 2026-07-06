"""Template-based grounded summarization for reviewer-facing outputs."""

from __future__ import annotations


def generate_reviewer_summary(
    entities: dict,
    claim_support_result: dict | None,
    documentation_result: dict | None,
) -> dict:
    """Generate a safe reviewer summary from extracted evidence only."""
    metadata = entities.get("document_metadata", {}) if isinstance(entities, dict) else {}
    document_type = metadata.get("document_type")

    diagnoses = _unique_entity_names(entities.get("diagnoses", []), status="affirmed")
    symptoms = _unique_entity_names(entities.get("symptoms", []), status="affirmed")
    procedures = _unique_entity_names(entities.get("procedures", []), status="affirmed")
    medications = _unique_entity_names(entities.get("medications", []), status="affirmed")
    codes = _unique_entity_names(entities.get("codes", []), status="affirmed")

    risk_flags = []
    if any(entity.get("status_modifier") == "negated" for entity in entities.get("diagnoses", [])):
        risk_flags.append("Negated diagnosis evidence present.")
    if any(entity.get("status_modifier") == "uncertain" for entity in entities.get("diagnoses", [])):
        risk_flags.append("Uncertain diagnosis evidence present.")

    if claim_support_result:
        overall_support = claim_support_result.get("overall_status", "Unclear")
        if overall_support == "Unclear":
            risk_flags.append("Claim support is unclear based on available document evidence.")
        elif overall_support == "Not Supported":
            risk_flags.append("Claim item support is not established in the current document.")
    else:
        overall_support = None

    missing_fields = documentation_result.get("missing_fields", []) if documentation_result else []
    if missing_fields:
        risk_flags.append("Missing documentation fields require reviewer follow-up.")

    short_summary_parts = []
    if document_type:
        short_summary_parts.append(f"Document type: {document_type}.")
    short_summary_parts.append(
        f"Key diagnoses: {_format_list(diagnoses, fallback='No clear diagnosis was extracted.')}."
    )
    if procedures:
        short_summary_parts.append(f"Key procedures: {_format_list(procedures)}.")
    if overall_support:
        short_summary_parts.append(f"Claim support status: {overall_support}.")
    short_summary = " ".join(short_summary_parts)

    clinical_evidence_summary = (
        f"Diagnoses: {_format_list(diagnoses, fallback='No clear diagnosis was extracted')}. "
        f"Symptoms: {_format_list(symptoms, fallback='No clear symptom evidence was extracted')}. "
        f"Procedures: {_format_list(procedures, fallback='No procedure evidence was extracted')}. "
        f"Medications: {_format_list(medications, fallback='No medication evidence was extracted')}. "
        f"Codes: {_format_list(codes, fallback='No coding references were extracted')}."
    )

    if claim_support_result:
        claim_support_summary = _build_claim_support_summary(claim_support_result)
    else:
        claim_support_summary = "Claim support has not been checked yet."

    if documentation_result:
        documentation_summary = _build_documentation_summary(documentation_result)
    else:
        documentation_summary = "Documentation completeness has not been evaluated yet."

    human_review_note = (
        "Human review required before any clinical interpretation, coding validation, "
        "or payment decision."
    )

    return {
        "short_summary": short_summary,
        "clinical_evidence_summary": clinical_evidence_summary,
        "claim_support_summary": claim_support_summary,
        "documentation_summary": documentation_summary,
        "risk_flags": risk_flags,
        "human_review_note": human_review_note,
    }


def _unique_entity_names(entities: list[dict], status: str | None = None) -> list[str]:
    names: list[str] = []
    seen = set()
    for entity in entities:
        if status and entity.get("status_modifier") != status:
            continue
        normalized = entity.get("normalized_text", entity.get("text", "")).strip()
        if normalized and normalized not in seen:
            names.append(normalized)
            seen.add(normalized)
    return names


def _format_list(items: list[str], fallback: str = "Insufficient evidence found") -> str:
    if not items:
        return fallback
    return ", ".join(items)


def _build_claim_support_summary(claim_support_result: dict) -> str:
    overall_status = claim_support_result.get("overall_status", "Unclear")
    items = claim_support_result.get("items", [])
    if not items:
        return "Support is unclear based on available document evidence."

    item_parts = [
        f"{item['claim_type'].title()} '{item['claim_value']}' -> {item['status']}"
        for item in items
    ]
    if overall_status == "Unclear":
        lead = "Support is unclear based on available document evidence."
    elif overall_status == "Not Supported":
        lead = "Claim support is not established in the current document evidence."
    elif overall_status == "Supported":
        lead = "Claim support appears present in the current document evidence."
    else:
        lead = "Claim support is partially established and still requires reviewer judgment."
    return f"{lead} {'; '.join(item_parts)}."


def _build_documentation_summary(documentation_result: dict) -> str:
    score = documentation_result.get("completeness_score", 0.0)
    label = documentation_result.get("score_label", "Low")
    missing_fields = documentation_result.get("missing_fields", [])
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        return (
            f"Documentation completeness is {label} ({score:.2f}). "
            f"Potentially missing fields: {missing_text}."
        )
    return f"Documentation completeness is {label} ({score:.2f}) with no obvious missing core fields detected."
