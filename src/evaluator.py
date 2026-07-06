"""Model-aware synthetic evaluation harness for the clinical evidence POC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

from src.claim_support_checker import check_claim_support
from src.documentation_checker import check_documentation_completeness
from src.document_loader import load_document
from src.entity_extractor import extract_entities, load_nlp_models
from src.evidence_retriever import (
    _load_sentence_transformer_model,
    answer_question,
    build_retrieval_index,
    chunk_document,
    retrieve_evidence,
)
from src.section_detector import detect_sections
from src.text_cleaner import clean_text


NOT_EVALUATED = "Not evaluated — missing labels"
CANONICAL_TERM_MAP = {
    "ekg": "ecg",
    "electrocardiogram": "ecg",
    "ecg": "ecg",
    "chest xray": "chest x-ray",
    "x ray": "x-ray",
    "aspirin 81 mg": "aspirin",
    "acute myocardial infarction": "acute myocardial infarction",
}
MISSING_FIELD_MAP = {
    "provider_signature": "provider_signature",
    "provider signature": "provider_signature",
    "date_of_service": "date_of_service",
    "date of service": "date_of_service",
    "provider_name": "provider_name",
    "provider name": "provider_name",
}


def run_evaluation(
    data_dir: str,
    use_scispacy: bool = True,
    use_sentence_transformers: bool = True,
) -> dict:
    """Run synthetic evaluation across bundled documents and labels."""
    data_path = Path(data_dir)
    docs_dir = data_path / "synthetic_documents"
    claims_path = data_path / "sample_claims.csv"
    labels_path = data_path / "expected_labels.json"

    sample_claims_df = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    expected_labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    document_paths = sorted(docs_dir.glob("*.txt"))

    model_registry = load_nlp_models()
    sentence_model, _sentence_warning = (
        _load_sentence_transformer_model() if use_sentence_transformers else (None, None)
    )
    scispacy_available = model_registry.get("scispacy_model") is not None
    bc5cdr_available = model_registry.get("bc5cdr_model") is not None
    bioclinicalbert_available = bool(model_registry.get("bioclinicalbert_available"))
    sentence_transformers_available = sentence_model is not None

    models = {
        "rule_based": "available",
        "scispacy_en_core_sci_sm": "available" if scispacy_available else "unavailable",
        "scispacy_en_ner_bc5cdr_md": "available" if bc5cdr_available else "unavailable",
        "sentence_transformers_all_MiniLM_L6_v2": (
            "available" if sentence_transformers_available else "unavailable"
        ),
        "Bio_ClinicalBERT_optional": "available" if bioclinicalbert_available else "unavailable",
    }

    extraction_scores = {
        "rule_based": {"diagnosis": [], "procedure": [], "medication": [], "code": []},
        "scispacy_enhanced": {"diagnosis": [], "procedure": [], "medication": [], "code": []},
    }
    claim_scores = {"diagnosis": [], "procedure": [], "code": [], "overall": []}
    negation_scores: list[bool] = []
    uncertainty_scores: list[bool] = []
    missing_signature_scores: list[bool] = []
    missing_date_scores: list[bool] = []
    missing_provider_scores: list[bool] = []
    completeness_scores: list[float] = []
    retrieval_hits: list[bool] = []
    top_scores: list[float] = []
    insufficient_evidence_scores: list[bool] = []
    per_document_results: list[dict[str, Any]] = []
    top_level_notes = [
        "This evaluation is designed to demonstrate research discipline and model-aware validation, not production accuracy.",
        "Synthetic data only. Metrics below are proof-of-concept checks, not deployment claims.",
        "Negation and uncertainty metrics are computed as document-level exact-set matches against available synthetic labels.",
    ]

    for document_path in document_paths:
        document_file = document_path.name
        labels = expected_labels.get(document_file)
        document_notes: list[str] = []

        loaded = load_document(document_path)
        raw_text = loaded.get("raw_text", "")
        cleaned = clean_text(raw_text)
        clean_document_text = cleaned.get("clean_text", "")
        sections = detect_sections(clean_document_text)

        rule_based_entities = _extract_rule_based_entities(clean_document_text, sections)
        scispacy_entities = None
        if use_scispacy and scispacy_available:
            scispacy_entities = extract_entities(
                clean_text=clean_document_text,
                sections=sections,
                use_advanced_models=False,
            )
        else:
            document_notes.append(
                "scispaCy comparison not available in this environment; rule-based extraction used as the active pipeline."
            )

        active_entities = scispacy_entities if scispacy_entities is not None else rule_based_entities

        documentation_result = check_documentation_completeness(
            entities=active_entities,
            sections=sections,
            clean_text=clean_document_text,
        )
        completeness_scores.append(float(documentation_result.get("completeness_score", 0.0)))

        retrieval_method = "sentence_transformers" if use_sentence_transformers else "tfidf"
        chunks = chunk_document(clean_document_text, sections)
        retrieval_index = build_retrieval_index(chunks, method=retrieval_method)
        document_notes.extend(retrieval_index.get("warnings", []))

        claim_row = _get_claim_row(sample_claims_df, document_file)
        claim_result = None
        claim_expected = NOT_EVALUATED
        claim_predicted = NOT_EVALUATED
        if claim_row:
            claim_payload = {
                "claimed_diagnosis": claim_row.get("claimed_diagnosis", ""),
                "claimed_procedure": claim_row.get("claimed_procedure", ""),
                "claimed_code": claim_row.get("claimed_code", ""),
            }
            claim_result = check_claim_support(
                claim=claim_payload,
                entities=active_entities,
                clean_text=clean_document_text,
                sections=sections,
            )
            claim_expected = _expected_overall_claim_status(claim_row)
            claim_predicted = claim_result.get("overall_status", "Unclear")
            if claim_expected != NOT_EVALUATED:
                claim_scores["overall"].append(
                    _normalize_overall_status(claim_expected)
                    == _normalize_overall_status(claim_predicted)
                )
            _update_claim_type_scores(claim_scores, claim_row, claim_result)
        else:
            document_notes.append("Claim support was not evaluated because no sample_claims.csv row was found.")

        retrieval_questions = _build_retrieval_questions(claim_row, labels)
        primary_retrieval_hit = False
        if not retrieval_questions:
            document_notes.append("No retrieval probe was created for this document.")
        for question_record in retrieval_questions:
            retrieved = retrieve_evidence(question_record["question"], retrieval_index, top_k=3)
            top_scores.append(float(retrieved[0]["score"]) if retrieved else 0.0)
            hit = _retrieval_hit(question_record["aliases"], retrieved)
            retrieval_hits.append(hit)
            primary_retrieval_hit = primary_retrieval_hit or hit

        unsupported_probe = "Does this document support appendicitis?"
        unsupported_retrieved = retrieve_evidence(unsupported_probe, retrieval_index, top_k=3)
        unsupported_answer = answer_question(
            question=unsupported_probe,
            retrieved_chunks=unsupported_retrieved,
            entities=active_entities,
            documentation_result=documentation_result,
        )
        insufficient_evidence_scores.append(
            unsupported_answer.get("answer_status") == "Insufficient Evidence"
        )

        if labels:
            _update_extraction_scores(extraction_scores["rule_based"], labels, rule_based_entities)
            if scispacy_entities is not None:
                _update_extraction_scores(extraction_scores["scispacy_enhanced"], labels, scispacy_entities)

            predicted_negated = _entity_names_by_status(active_entities, "diagnoses", "negated")
            predicted_uncertain = _entity_names_by_status(active_entities, "diagnoses", "uncertain")
            expected_negated = {_canonicalize_value(value) for value in labels.get("negated_entities", [])}
            expected_uncertain = {_canonicalize_value(value) for value in labels.get("uncertain_entities", [])}
            negation_scores.append(predicted_negated == expected_negated)
            uncertainty_scores.append(predicted_uncertain == expected_uncertain)

            predicted_missing = {
                _normalize_missing_field(value)
                for value in documentation_result.get("missing_fields", [])
                if _normalize_missing_field(value)
            }
            expected_missing = {
                _normalize_missing_field(value)
                for value in labels.get("missing_fields", [])
                if _normalize_missing_field(value)
            }
            missing_signature_scores.append(
                ("provider_signature" in predicted_missing) == ("provider_signature" in expected_missing)
            )
            missing_date_scores.append(
                ("date_of_service" in predicted_missing) == ("date_of_service" in expected_missing)
            )
            missing_provider_scores.append(
                ("provider_name" in predicted_missing) == ("provider_name" in expected_missing)
            )
        else:
            document_notes.append("Expected labels are missing for this document.")

        per_document_results.append(
            {
                "document_file": document_file,
                "expected_diagnoses": labels.get("diagnoses", []) if labels else NOT_EVALUATED,
                "predicted_diagnoses": _sorted_display_names(active_entities, "diagnoses"),
                "predicted_diagnoses_rule_based": _sorted_display_names(rule_based_entities, "diagnoses"),
                "predicted_diagnoses_scispacy": (
                    _sorted_display_names(scispacy_entities, "diagnoses")
                    if scispacy_entities is not None
                    else "Not available"
                ),
                "expected_procedures": labels.get("procedures", []) if labels else NOT_EVALUATED,
                "predicted_procedures": _sorted_display_names(active_entities, "procedures"),
                "claim_expected": claim_expected,
                "claim_predicted": claim_predicted,
                "missing_fields_expected": labels.get("missing_fields", []) if labels else NOT_EVALUATED,
                "missing_fields_predicted": documentation_result.get("missing_fields", []),
                "retrieval_hit": primary_retrieval_hit,
                "notes": " | ".join(document_notes) if document_notes else "",
            }
        )

    retrieval_methods_used = {
        result.get("notes", "")
        for result in per_document_results
        if "Sentence-transformers retrieval is unavailable" in result.get("notes", "")
    }
    retrieval_method_label = (
        retrieval_index.get("method", "tfidf") if document_paths else "tfidf"
    )
    if retrieval_methods_used:
        retrieval_method_label = "tfidf"

    return {
        "models": models,
        "dataset_metrics": {
            "documents_tested": len(document_paths),
            "synthea_derived_documents_tested": sum(
                1 for path in document_paths if path.name.startswith("synthetic_")
            ),
            "controlled_edge_case_documents_tested": sum(
                1 for path in document_paths if not path.name.startswith("synthetic_")
            ),
        },
        "entity_extraction_metrics": {
            "rule_based": _format_extraction_metrics(extraction_scores["rule_based"]),
            "scispacy_enhanced": (
                _format_extraction_metrics(extraction_scores["scispacy_enhanced"])
                if use_scispacy and scispacy_available
                else {
                    "diagnosis_exact_match_rate": "Not available",
                    "procedure_exact_match_rate": "Not available",
                    "medication_exact_match_rate": "Not available",
                    "code_exact_match_rate": "Not available",
                }
            ),
        },
        "claim_support_metrics": {
            "diagnosis_support_accuracy": _safe_rate(claim_scores["diagnosis"]),
            "procedure_support_accuracy": _safe_rate(claim_scores["procedure"]),
            "code_support_accuracy": _safe_rate(claim_scores["code"]),
            "overall_claim_status_accuracy": _safe_rate(claim_scores["overall"]),
        },
        "negation_uncertainty_metrics": {
            "negation_detection_accuracy": _safe_rate(negation_scores),
            "uncertainty_detection_accuracy": _safe_rate(uncertainty_scores),
        },
        "documentation_metrics": {
            "missing_signature_detection_accuracy": _safe_rate(missing_signature_scores),
            "missing_date_detection_accuracy": _safe_rate(missing_date_scores),
            "missing_provider_detection_accuracy": _safe_rate(missing_provider_scores),
            "average_completeness_score": round(
                sum(completeness_scores) / len(completeness_scores), 2
            )
            if completeness_scores
            else NOT_EVALUATED,
        },
        "retrieval_metrics": {
            "retrieval_method_used": retrieval_method_label,
            "evidence_retrieval_hit_rate": _safe_rate(retrieval_hits),
            "average_top_score": round(sum(top_scores) / len(top_scores), 3) if top_scores else NOT_EVALUATED,
            "insufficient_evidence_correct_rate": _safe_rate(insufficient_evidence_scores),
        },
        "per_document_results": per_document_results,
        "notes": top_level_notes,
    }


def _extract_rule_based_entities(clean_text_value: str, sections: dict) -> dict:
    """Force rule-based extraction only for baseline comparison."""
    model_stub = {
        "spacy_available": False,
        "scispacy_model": None,
        "bc5cdr_model": None,
        "bioclinicalbert_available": False,
        "warnings": [],
    }
    with patch("src.entity_extractor.load_nlp_models", return_value=model_stub):
        return extract_entities(
            clean_text=clean_text_value,
            sections=sections,
            use_advanced_models=False,
        )


def _get_claim_row(sample_claims_df: pd.DataFrame, document_file: str) -> dict[str, Any] | None:
    if sample_claims_df.empty or "document_file" not in sample_claims_df.columns:
        return None
    matches = sample_claims_df[sample_claims_df["document_file"] == document_file]
    if matches.empty:
        return None
    return matches.iloc[0].fillna("").to_dict()


def _update_extraction_scores(scores: dict[str, list[bool]], labels: dict, entities: dict) -> None:
    expected_diagnoses = {_canonicalize_value(value) for value in labels.get("diagnoses", [])}
    expected_procedures = {_canonicalize_value(value) for value in labels.get("procedures", [])}
    expected_medications = {_canonicalize_value(value) for value in labels.get("medications", [])}
    expected_codes = {_canonicalize_value(value) for value in labels.get("codes", [])}

    scores["diagnosis"].append(_entity_name_set(entities, "diagnoses") == expected_diagnoses)
    scores["procedure"].append(_entity_name_set(entities, "procedures") == expected_procedures)
    scores["medication"].append(_entity_name_set(entities, "medications") == expected_medications)
    scores["code"].append(_entity_name_set(entities, "codes") == expected_codes)


def _entity_name_set(entities: dict, bucket: str) -> set[str]:
    return {
        _canonicalize_value(entity.get("normalized_text", entity.get("text", "")))
        for entity in entities.get(bucket, [])
        if entity.get("status_modifier") == "affirmed"
    }


def _entity_names_by_status(entities: dict, bucket: str, status: str) -> set[str]:
    return {
        _canonicalize_value(entity.get("normalized_text", entity.get("text", "")))
        for entity in entities.get(bucket, [])
        if entity.get("status_modifier") == status
    }


def _sorted_display_names(entities: dict | None, bucket: str) -> list[str]:
    if not entities:
        return []
    names = []
    seen = set()
    for entity in entities.get(bucket, []):
        if entity.get("status_modifier") != "affirmed":
            continue
        normalized = entity.get("normalized_text", entity.get("text", ""))
        if normalized not in seen:
            names.append(normalized)
            seen.add(normalized)
    return sorted(names)


def _format_extraction_metrics(scores: dict[str, list[bool]]) -> dict[str, Any]:
    return {
        "diagnosis_exact_match_rate": _safe_rate(scores["diagnosis"]),
        "procedure_exact_match_rate": _safe_rate(scores["procedure"]),
        "medication_exact_match_rate": _safe_rate(scores["medication"]),
        "code_exact_match_rate": _safe_rate(scores["code"]),
    }


def _safe_rate(values: list[bool]) -> float | str:
    if not values:
        return NOT_EVALUATED
    return round(sum(1 for value in values if value) / len(values), 2)


def _canonicalize_value(value: str) -> str:
    normalized = " ".join((value or "").strip().lower().split())
    normalized = normalized.replace("_", " ")
    normalized = CANONICAL_TERM_MAP.get(normalized, normalized)
    return normalized


def _normalize_missing_field(value: str) -> str | None:
    normalized = _canonicalize_value(value)
    return MISSING_FIELD_MAP.get(normalized)


def _normalize_expected_status(status: str) -> str:
    status = (status or "").strip()
    mapping = {
        "Supported": "Supported",
        "Not Supported": "Not Supported",
        "Negated": "Not Supported",
        "Missing Evidence": "Not Supported",
        "Unclear": "Unclear",
        "Partially Supported": "Partially Supported",
    }
    return mapping.get(status, status or NOT_EVALUATED)


def _normalize_overall_status(status: str) -> str:
    status = _normalize_expected_status(status)
    if status == "Partially Supported":
        return "Partially Supported"
    return status


def _expected_overall_claim_status(claim_row: dict[str, Any]) -> str:
    statuses = []
    if str(claim_row.get("claimed_diagnosis", "")).strip():
        statuses.append(_normalize_expected_status(str(claim_row.get("expected_diagnosis_status", ""))))
    if str(claim_row.get("claimed_procedure", "")).strip():
        statuses.append(_normalize_expected_status(str(claim_row.get("expected_procedure_status", ""))))
    if str(claim_row.get("claimed_code", "")).strip():
        statuses.append(_normalize_expected_status(str(claim_row.get("expected_code_status", ""))))
    if not statuses:
        return NOT_EVALUATED
    if all(status == "Supported" for status in statuses):
        return "Supported"
    if all(status == "Not Supported" for status in statuses):
        return "Not Supported"
    if "Unclear" in statuses and all(status in {"Supported", "Unclear"} for status in statuses):
        return "Partially Supported" if "Supported" in statuses else "Unclear"
    if any(status == "Supported" for status in statuses):
        return "Partially Supported"
    if any(status == "Unclear" for status in statuses):
        return "Unclear"
    return "Not Supported"


def _update_claim_type_scores(
    claim_scores: dict[str, list[bool]],
    claim_row: dict[str, Any],
    claim_result: dict,
) -> None:
    item_by_type = {item["claim_type"]: item for item in claim_result.get("items", [])}
    claim_type_specs = [
        ("diagnosis", "claimed_diagnosis", "expected_diagnosis_status"),
        ("procedure", "claimed_procedure", "expected_procedure_status"),
        ("code", "claimed_code", "expected_code_status"),
    ]
    for claim_type, value_key, expected_key in claim_type_specs:
        if not str(claim_row.get(value_key, "")).strip():
            continue
        expected_status = _normalize_expected_status(str(claim_row.get(expected_key, "")))
        item = item_by_type.get(claim_type)
        if expected_status == NOT_EVALUATED or item is None:
            continue
        predicted_status = _normalize_expected_status(item.get("status", ""))
        claim_scores[claim_type].append(expected_status == predicted_status)


def _build_retrieval_questions(claim_row: dict[str, Any] | None, labels: dict | None) -> list[dict[str, Any]]:
    questions = []
    if claim_row:
        diagnosis = str(claim_row.get("claimed_diagnosis", "")).strip()
        procedure = str(claim_row.get("claimed_procedure", "")).strip()
        code = str(claim_row.get("claimed_code", "")).strip()
        if diagnosis:
            questions.append(
                {
                    "question": f"Does this document support {diagnosis}?",
                    "aliases": _aliases_for_target(diagnosis),
                }
            )
        if procedure:
            questions.append(
                {
                    "question": f"Does this document support {procedure}?",
                    "aliases": _aliases_for_target(procedure),
                }
            )
        if code:
            questions.append(
                {
                    "question": f"Does this document support {code}?",
                    "aliases": _aliases_for_target(code),
                }
            )
    elif labels and labels.get("diagnoses"):
        first_diagnosis = labels["diagnoses"][0]
        questions.append(
            {
                "question": f"Does this document support {first_diagnosis}?",
                "aliases": _aliases_for_target(first_diagnosis),
            }
        )
    return questions


def _aliases_for_target(value: str) -> set[str]:
    canonical = _canonicalize_value(value)
    aliases = {canonical}
    if canonical == "ecg":
        aliases.update({"ekg", "electrocardiogram"})
    if canonical == "chest x-ray":
        aliases.update({"x-ray", "chest xray"})
    return aliases


def _retrieval_hit(aliases: set[str], retrieved_chunks: list[dict]) -> bool:
    if not retrieved_chunks:
        return False
    texts = [" ".join(chunk.get("text", "").lower().split()) for chunk in retrieved_chunks]
    for alias in aliases:
        normalized_alias = alias.lower()
        if any(normalized_alias in text for text in texts):
            return True
    return False
