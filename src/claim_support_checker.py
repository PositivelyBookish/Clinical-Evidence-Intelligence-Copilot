"""Claim support classification for reviewer-facing clinical evidence checks."""

from __future__ import annotations

from typing import Any


DIAGNOSIS_CODE_MAP = {
    "I10": "hypertension",
    "E11.9": "diabetes",
    "J18.9": "pneumonia",
    "R07.9": "chest pain",
}
PROCEDURE_CODE_MAP = {
    "93000": ["ecg", "ekg", "electrocardiogram"],
    "71046": ["chest x-ray", "x-ray"],
    "82947": ["glucose test"],
}
DIAGNOSIS_STRONG_SECTIONS = {"assessment", "diagnosis", "codes"}
PROCEDURE_STRONG_SECTIONS = {"procedures", "codes"}
CODE_STRONG_SECTIONS = {"codes", "procedures", "assessment", "diagnosis"}
NEGATIVE_SUPPORT_PHRASES = [
    "not mentioned",
    "not documented",
    "not present",
    "no procedure performed",
    "does not support",
]


def check_claim_support(claim: dict, entities: dict, clean_text: str, sections: dict) -> dict:
    """Classify whether a claimed diagnosis, procedure, or code is supported."""
    del clean_text  # Reserved for future richer evidence retrieval.
    del sections

    warnings: list[str] = []
    items: list[dict[str, Any]] = []

    claimed_diagnosis = (claim.get("claimed_diagnosis") or "").strip()
    claimed_procedure = (claim.get("claimed_procedure") or "").strip()
    claimed_code = (claim.get("claimed_code") or "").strip().upper()

    if claimed_diagnosis:
        items.append(
            _evaluate_entity_claim(
                claim_type="diagnosis",
                claim_value=claimed_diagnosis,
                entity_candidates=entities.get("diagnoses", []),
                strong_sections=DIAGNOSIS_STRONG_SECTIONS,
            )
        )

    if claimed_procedure:
        items.append(
            _evaluate_entity_claim(
                claim_type="procedure",
                claim_value=claimed_procedure,
                entity_candidates=entities.get("procedures", []),
                strong_sections=PROCEDURE_STRONG_SECTIONS,
                aliases=_procedure_aliases(claimed_procedure),
            )
        )

    if claimed_code:
        code_result = _evaluate_code_claim(
            claimed_code=claimed_code,
            entities=entities,
        )
        items.append(code_result)
        if "reviewer validation" in code_result["reason"].lower():
            warnings.append(
                "Clinical evidence can suggest code support, but final coding correctness still requires human review."
            )

    if not items:
        warnings.append("No claim items were entered for review.")
        return {
            "overall_status": "Unclear",
            "items": [],
            "reviewer_summary": "No diagnosis, procedure, or code claim was provided for review.",
            "warnings": warnings,
        }

    if any("conflicting" in item["reason"].lower() for item in items):
        warnings.append("Conflicting evidence was detected for at least one claim item.")

    overall_status = _determine_overall_status(items)
    reviewer_summary = _build_reviewer_summary(overall_status, items)

    return {
        "overall_status": overall_status,
        "items": items,
        "reviewer_summary": reviewer_summary,
        "warnings": warnings,
    }


def _evaluate_entity_claim(
    claim_type: str,
    claim_value: str,
    entity_candidates: list[dict],
    strong_sections: set[str],
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate support for a diagnosis or procedure claim."""
    aliases = aliases or []
    matches = [
        entity
        for entity in entity_candidates
        if _entity_matches_claim(entity, claim_value, aliases)
    ]

    if not matches:
        return _build_item(
            claim_type=claim_type,
            claim_value=claim_value,
            status="Not Supported",
            confidence="Low",
            evidence="",
            reason=f"No matching {claim_type} mention was found in the current document.",
        )

    affirmed_matches = [entity for entity in matches if entity.get("status_modifier") == "affirmed"]
    uncertain_matches = [entity for entity in matches if entity.get("status_modifier") == "uncertain"]
    negated_matches = [entity for entity in matches if entity.get("status_modifier") == "negated"]

    strong_affirmed = [entity for entity in affirmed_matches if entity.get("section") in strong_sections]

    if strong_affirmed and (uncertain_matches or negated_matches):
        best_match = _best_entity(strong_affirmed)
        return _build_item(
            claim_type=claim_type,
            claim_value=claim_value,
            status="Unclear",
            confidence="Medium",
            evidence=best_match.get("evidence", ""),
            reason=(
                f"{claim_type.title()} evidence is present, but conflicting negated or uncertain mentions "
                "mean the item still needs reviewer judgment."
            ),
        )

    if strong_affirmed:
        best_match = _best_entity(strong_affirmed)
        return _build_item(
            claim_type=claim_type,
            claim_value=claim_value,
            status="Supported",
            confidence="High",
            evidence=best_match.get("evidence", ""),
            reason=(
                f"{claim_type.title()} found in {best_match.get('section', 'a relevant section')} "
                "and is not negated."
            ),
        )

    if affirmed_matches and all(_is_negative_support_context(entity.get("evidence", "")) for entity in affirmed_matches):
        best_match = _best_entity(affirmed_matches)
        return _build_item(
            claim_type=claim_type,
            claim_value=claim_value,
            status="Not Supported",
            confidence="Low",
            evidence=best_match.get("evidence", ""),
            reason=(
                f"{claim_type.title()} appears only in non-supportive commentary and is not documented as performed or confirmed."
            ),
        )

    if affirmed_matches:
        best_match = _best_entity(affirmed_matches)
        return _build_item(
            claim_type=claim_type,
            claim_value=claim_value,
            status="Unclear",
            confidence="Medium" if best_match.get("confidence") != "Low" else "Low",
            evidence=best_match.get("evidence", ""),
            reason=(
                f"{claim_type.title()} was found, but only in weaker or less structured context. "
                "Support should be confirmed by a reviewer."
            ),
        )

    if uncertain_matches:
        best_match = _best_entity(uncertain_matches)
        return _build_item(
            claim_type=claim_type,
            claim_value=claim_value,
            status="Unclear",
            confidence="Low",
            evidence=best_match.get("evidence", ""),
            reason=(
                f"{claim_type.title()} appears in uncertain context and should not be treated as fully supported."
            ),
        )

    best_match = _best_entity(negated_matches or matches)
    return _build_item(
        claim_type=claim_type,
        claim_value=claim_value,
        status="Not Supported",
        confidence="Low",
        evidence=best_match.get("evidence", ""),
        reason=f"{claim_type.title()} appears only in negated context and does not support the claim.",
    )


def _evaluate_code_claim(claimed_code: str, entities: dict) -> dict[str, Any]:
    """Evaluate support for a code claim using direct matches and cautious inference."""
    code_entities = entities.get("codes", [])
    direct_matches = [
        entity for entity in code_entities if entity.get("normalized_text", "").upper() == claimed_code
    ]
    affirmed_direct = [
        entity for entity in direct_matches if entity.get("status_modifier") == "affirmed"
    ]

    if affirmed_direct:
        best_match = _best_entity(affirmed_direct)
        return _build_item(
            claim_type="code",
            claim_value=claimed_code,
            status="Supported",
            confidence="High",
            evidence=best_match.get("evidence", ""),
            reason="Claimed code appears directly in the document evidence.",
        )

    inferred_support = _infer_code_support_from_entities(claimed_code, entities)
    if inferred_support:
        return inferred_support

    if direct_matches:
        best_match = _best_entity(direct_matches)
        if best_match.get("status_modifier") == "uncertain":
            return _build_item(
                claim_type="code",
                claim_value=claimed_code,
                status="Unclear",
                confidence="Low",
                evidence=best_match.get("evidence", ""),
                reason="Claimed code appears only in uncertain context and needs reviewer validation.",
            )
        return _build_item(
            claim_type="code",
            claim_value=claimed_code,
            status="Not Supported",
            confidence="Low",
            evidence=best_match.get("evidence", ""),
            reason="Claimed code appears only in non-supporting context.",
        )

    return _build_item(
        claim_type="code",
        claim_value=claimed_code,
        status="Not Supported",
        confidence="Low",
        evidence="",
        reason="No direct or clinically inferred support was found for the claimed code.",
    )


def _infer_code_support_from_entities(claimed_code: str, entities: dict) -> dict[str, Any] | None:
    """Infer cautious code support from aligned diagnosis or procedure evidence."""
    diagnosis_target = DIAGNOSIS_CODE_MAP.get(claimed_code)
    if diagnosis_target:
        diagnosis_entities = [
            entity
            for entity in entities.get("diagnoses", [])
            if diagnosis_target == entity.get("normalized_text", "").lower()
        ]
        supported_entities = [
            entity for entity in diagnosis_entities if entity.get("status_modifier") == "affirmed"
        ]
        uncertain_entities = [
            entity for entity in diagnosis_entities if entity.get("status_modifier") == "uncertain"
        ]
        negated_entities = [
            entity for entity in diagnosis_entities if entity.get("status_modifier") == "negated"
        ]

        if supported_entities:
            best_match = _best_entity(supported_entities)
            return _build_item(
                claim_type="code",
                claim_value=claimed_code,
                status="Supported",
                confidence="Medium",
                evidence=best_match.get("evidence", ""),
                reason=(
                    f"Clinically supported by diagnosis mention of {best_match.get('normalized_text')}, "
                    "but code match still requires reviewer validation."
                ),
            )
        if uncertain_entities:
            best_match = _best_entity(uncertain_entities)
            return _build_item(
                claim_type="code",
                claim_value=claimed_code,
                status="Unclear",
                confidence="Low",
                evidence=best_match.get("evidence", ""),
                reason="Related diagnosis evidence is uncertain, so the code cannot be treated as fully supported.",
            )
        if negated_entities:
            best_match = _best_entity(negated_entities)
            return _build_item(
                claim_type="code",
                claim_value=claimed_code,
                status="Not Supported",
                confidence="Low",
                evidence=best_match.get("evidence", ""),
                reason="Related diagnosis evidence is negated, so the code is not supported.",
            )

    procedure_targets = PROCEDURE_CODE_MAP.get(claimed_code, [])
    if procedure_targets:
        procedure_entities = [
            entity
            for entity in entities.get("procedures", [])
            if any(target in entity.get("normalized_text", "").lower() for target in procedure_targets)
        ]
        supported_entities = [
            entity for entity in procedure_entities if entity.get("status_modifier") == "affirmed"
        ]
        uncertain_entities = [
            entity for entity in procedure_entities if entity.get("status_modifier") == "uncertain"
        ]
        negated_entities = [
            entity for entity in procedure_entities if entity.get("status_modifier") == "negated"
        ]

        if supported_entities:
            best_match = _best_entity(supported_entities)
            return _build_item(
                claim_type="code",
                claim_value=claimed_code,
                status="Supported",
                confidence="High",
                evidence=best_match.get("evidence", ""),
                reason=(
                    f"Procedure evidence for {best_match.get('normalized_text')} is present. "
                    "Claimed code appears clinically supported, but reviewer validation is still required."
                ),
            )
        if uncertain_entities:
            best_match = _best_entity(uncertain_entities)
            return _build_item(
                claim_type="code",
                claim_value=claimed_code,
                status="Unclear",
                confidence="Low",
                evidence=best_match.get("evidence", ""),
                reason="Related procedure evidence is uncertain, so code support is unclear.",
            )
        if negated_entities:
            best_match = _best_entity(negated_entities)
            return _build_item(
                claim_type="code",
                claim_value=claimed_code,
                status="Not Supported",
                confidence="Low",
                evidence=best_match.get("evidence", ""),
                reason="Related procedure evidence is negated, so the code is not supported.",
            )
    return None


def _entity_matches_claim(entity: dict, claim_value: str, aliases: list[str] | None = None) -> bool:
    """Check whether an extracted entity matches the claimed value."""
    aliases = aliases or []
    claim_normalized = _normalize_text(claim_value)
    entity_normalized = _normalize_text(entity.get("normalized_text", entity.get("text", "")))

    candidates = {claim_normalized, entity_normalized}
    candidates.update(_normalize_text(alias) for alias in aliases)

    if claim_normalized == entity_normalized:
        return True
    if claim_normalized in entity_normalized or entity_normalized in claim_normalized:
        return True
    return entity_normalized in candidates or claim_normalized in candidates


def _procedure_aliases(claimed_procedure: str) -> list[str]:
    """Return normalized aliases for common equivalent procedure names."""
    normalized = _normalize_text(claimed_procedure)
    if normalized in {"ecg", "ekg", "electrocardiogram"}:
        return ["ECG", "EKG", "electrocardiogram"]
    return [claimed_procedure]


def _best_entity(entities: list[dict]) -> dict:
    """Pick the strongest evidence entity from a match set."""
    return sorted(
        entities,
        key=lambda entity: (
            _status_rank(entity.get("status_modifier", "affirmed")),
            _confidence_rank(entity.get("confidence", "Low")),
            1 if entity.get("section") in CODE_STRONG_SECTIONS else 0,
            len(entity.get("evidence", "")),
        ),
        reverse=True,
    )[0]


def _determine_overall_status(items: list[dict[str, Any]]) -> str:
    """Roll up item statuses into a document-level judgment."""
    statuses = [item["status"] for item in items]
    if statuses and all(status == "Supported" for status in statuses):
        return "Supported"
    if any(status == "Supported" for status in statuses):
        return "Partially Supported"
    if any(status == "Unclear" for status in statuses):
        return "Unclear"
    return "Not Supported"


def _build_reviewer_summary(overall_status: str, items: list[dict[str, Any]]) -> str:
    """Summarize reviewer takeaways in cautious language."""
    fragments = [
        f"{item['claim_type'].title()} '{item['claim_value']}' -> {item['status']}"
        for item in items
    ]
    return (
        f"Overall claim review status: {overall_status}. "
        f"Item summary: {'; '.join(fragments)}. "
        "Human review remains required before any final coding or payment decision."
    )


def _build_item(
    claim_type: str,
    claim_value: str,
    status: str,
    confidence: str,
    evidence: str,
    reason: str,
) -> dict[str, Any]:
    """Construct a reviewer-facing item result."""
    return {
        "claim_type": claim_type,
        "claim_value": claim_value,
        "status": status,
        "confidence": confidence,
        "evidence": evidence,
        "reason": reason,
        "human_review_required": True,
    }


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _is_negative_support_context(evidence: str) -> bool:
    lowered = evidence.lower()
    return any(phrase in lowered for phrase in NEGATIVE_SUPPORT_PHRASES)


def _confidence_rank(confidence: str) -> int:
    return {"Low": 1, "Medium": 2, "High": 3}.get(confidence, 0)


def _status_rank(status: str) -> int:
    return {"affirmed": 3, "uncertain": 2, "negated": 1}.get(status, 0)
