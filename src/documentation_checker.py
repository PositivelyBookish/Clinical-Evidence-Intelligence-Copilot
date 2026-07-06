"""Documentation completeness checks for reviewer-facing clinical review."""

from __future__ import annotations

from typing import Any


def check_documentation_completeness(entities: dict, sections: dict, clean_text: str) -> dict:
    """Assess whether the document includes core review fields."""
    del clean_text  # Reserved for future richer sentence-level checks.

    section_map = sections.get("sections", sections) if isinstance(sections, dict) else {}
    section_names = set(section_map.keys()) if isinstance(section_map, dict) else set()

    checks = [
        _build_check(
            field="Patient name",
            status=_presence_status(entities.get("patient", [])),
            evidence=_first_evidence(entities.get("patient", [])),
            importance="High",
            reviewer_action="Confirm the patient identity aligns with the reviewed claim.",
        ),
        _build_check(
            field="Date of Service",
            status=_presence_status(entities.get("dates", [])),
            evidence=_first_evidence(entities.get("dates", [])),
            importance="High",
            reviewer_action="Verify date matches claim.",
        ),
        _build_check(
            field="Provider name",
            status=_presence_status(entities.get("provider", [])),
            evidence=_first_evidence(entities.get("provider", [])),
            importance="High",
            reviewer_action="Confirm the treating or documenting provider is identified.",
        ),
        _build_check(
            field="Provider signature",
            status=_presence_status(entities.get("signatures", [])),
            evidence=_first_evidence(entities.get("signatures", [])),
            importance="High",
            reviewer_action="Confirm the note is signed or appropriately authenticated.",
        ),
        _build_check(
            field="Diagnosis or assessment",
            status=_diagnosis_assessment_status(entities, section_names),
            evidence=_diagnosis_assessment_evidence(entities, section_map),
            importance="High",
            reviewer_action="Verify the diagnosis or assessment supports the review target.",
        ),
        _build_check(
            field="Procedure/test support",
            status=_procedure_support_status(entities, section_names),
            evidence=_first_evidence(entities.get("procedures", [])),
            importance="High",
            reviewer_action="Verify the procedure or test is documented when a procedural claim is under review.",
        ),
        _build_check(
            field="Medication evidence",
            status=_presence_status(entities.get("medications", [])),
            evidence=_first_evidence(entities.get("medications", [])),
            importance="Medium",
            reviewer_action="Check whether medications add supporting clinical context.",
        ),
        _build_check(
            field="ICD-10-like code",
            status=_code_presence_status(entities.get("codes", []), code_type="icd"),
            evidence=_code_evidence(entities.get("codes", []), code_type="icd"),
            importance="Medium",
            reviewer_action="Confirm diagnosis-linked coding references if this is a coding validation note.",
        ),
        _build_check(
            field="CPT-like code",
            status=_code_presence_status(entities.get("codes", []), code_type="cpt"),
            evidence=_code_evidence(entities.get("codes", []), code_type="cpt"),
            importance="Medium",
            reviewer_action="Confirm procedural coding references if a service or procedure is being reviewed.",
        ),
        _build_check(
            field="Clinical evidence sentence",
            status=_clinical_evidence_status(entities),
            evidence=_clinical_evidence_text(entities),
            importance="High",
            reviewer_action="Review the strongest evidence sentence before making any support determination.",
        ),
    ]

    total_score = sum(_status_score(check["status"]) for check in checks)
    completeness_score = total_score / len(checks) if checks else 0.0
    score_label = _score_label(completeness_score)
    missing_fields = [check["field"] for check in checks if check["status"] == "Missing"]

    reviewer_recommendations = []
    for check in checks:
        if check["status"] != "Present":
            reviewer_recommendations.append(f"{check['field']}: {check['reviewer_action']}")

    return {
        "completeness_score": round(completeness_score, 2),
        "score_label": score_label,
        "checks": checks,
        "missing_fields": missing_fields,
        "reviewer_recommendations": reviewer_recommendations,
    }


def _build_check(
    field: str,
    status: str,
    evidence: str,
    importance: str,
    reviewer_action: str,
) -> dict[str, Any]:
    return {
        "field": field,
        "status": status,
        "evidence": evidence or "No direct evidence found.",
        "importance": importance,
        "reviewer_action": reviewer_action,
    }


def _presence_status(entities: list[dict]) -> str:
    if not entities:
        return "Missing"
    if any(entity.get("status_modifier") == "affirmed" for entity in entities):
        return "Present"
    if any(entity.get("status_modifier") == "uncertain" for entity in entities):
        return "Unclear"
    return "Missing"


def _diagnosis_assessment_status(entities: dict, section_names: set[str]) -> str:
    diagnoses = entities.get("diagnoses", [])
    if any(entity.get("status_modifier") == "affirmed" for entity in diagnoses):
        return "Present"
    if diagnoses or "assessment" in section_names or "diagnosis" in section_names:
        return "Unclear"
    return "Missing"


def _diagnosis_assessment_evidence(entities: dict, section_map: dict) -> str:
    diagnoses = entities.get("diagnoses", [])
    evidence = _first_evidence(diagnoses)
    if evidence:
        return evidence
    for section_name in ["assessment", "diagnosis"]:
        if section_name in section_map and section_map[section_name].strip():
            return section_map[section_name].split("\n")[0]
    return ""


def _procedure_support_status(entities: dict, section_names: set[str]) -> str:
    procedures = entities.get("procedures", [])
    if any(entity.get("status_modifier") == "affirmed" for entity in procedures):
        return "Present"
    if procedures or "procedures" in section_names:
        return "Unclear"
    return "Missing"


def _code_presence_status(code_entities: list[dict], code_type: str) -> str:
    if not code_entities:
        return "Missing"
    for entity in code_entities:
        normalized = entity.get("normalized_text", "")
        if code_type == "icd" and _is_icd_code(normalized) and entity.get("status_modifier") == "affirmed":
            return "Present"
        if code_type == "cpt" and _is_cpt_code(normalized) and entity.get("status_modifier") == "affirmed":
            return "Present"
    for entity in code_entities:
        normalized = entity.get("normalized_text", "")
        if code_type == "icd" and _is_icd_code(normalized):
            return "Unclear"
        if code_type == "cpt" and _is_cpt_code(normalized):
            return "Unclear"
    return "Missing"


def _code_evidence(code_entities: list[dict], code_type: str) -> str:
    for entity in code_entities:
        normalized = entity.get("normalized_text", "")
        if code_type == "icd" and _is_icd_code(normalized):
            return entity.get("evidence", "")
        if code_type == "cpt" and _is_cpt_code(normalized):
            return entity.get("evidence", "")
    return ""


def _clinical_evidence_status(entities: dict) -> str:
    evidence_candidates = []
    for bucket_name in ["diagnoses", "procedures", "medications", "codes"]:
        evidence_candidates.extend(entities.get(bucket_name, []))
    affirmed_candidates = [entity for entity in evidence_candidates if entity.get("status_modifier") == "affirmed"]
    uncertain_candidates = [entity for entity in evidence_candidates if entity.get("status_modifier") == "uncertain"]
    if affirmed_candidates:
        return "Present"
    if uncertain_candidates or evidence_candidates:
        return "Unclear"
    return "Missing"


def _clinical_evidence_text(entities: dict) -> str:
    evidence_candidates = []
    for bucket_name in ["diagnoses", "procedures", "medications", "codes"]:
        evidence_candidates.extend(entities.get(bucket_name, []))
    if not evidence_candidates:
        return ""
    best = sorted(
        evidence_candidates,
        key=lambda entity: (
            entity.get("status_modifier") == "affirmed",
            entity.get("confidence") == "High",
            len(entity.get("evidence", "")),
        ),
        reverse=True,
    )[0]
    return best.get("evidence", "")


def _first_evidence(entities: list[dict]) -> str:
    if not entities:
        return ""
    return entities[0].get("evidence", "")


def _status_score(status: str) -> float:
    return {"Present": 1.0, "Unclear": 0.5, "Missing": 0.0}.get(status, 0.0)


def _score_label(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def _is_icd_code(value: str) -> bool:
    value = (value or "").strip().upper()
    return len(value) >= 3 and value[0].isalpha() and value[1:3].isdigit()


def _is_cpt_code(value: str) -> bool:
    value = (value or "").strip()
    return len(value) == 5 and value.isdigit()
