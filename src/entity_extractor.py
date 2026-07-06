"""Layered clinical entity extraction with graceful model fallbacks."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from src.negation_detector import detect_negation_and_uncertainty


FALLBACK_WARNING = "Advanced clinical NLP model not installed. Falling back to rule-based extraction."
BIOCLINICALBERT_NAME = "emilyalsentzer/Bio_ClinicalBERT"
STRONG_SECTIONS = {
    "assessment",
    "diagnosis",
    "procedures",
    "medications",
    "codes",
    "provider_signature",
}
OCR_NOISE_MARKERS = [
    "patlent",
    "provlder",
    "medicatlon",
    "dlagnosis",
    "slgnature",
    "cllnic",
    "hypertenslon",
    "dally",
]

ENTITY_CATEGORY_ORDER = [
    "patient",
    "provider",
    "dates",
    "diagnoses",
    "symptoms",
    "procedures",
    "medications",
    "codes",
    "signatures",
]

ENTITY_SPECS = {
    "diagnosis": [
        ("chronic obstructive pulmonary disease", "COPD"),
        ("chest pain syndrome", "Chest pain syndrome"),
        ("type 2 diabetes", "Type 2 diabetes"),
        ("heart failure", "Heart failure"),
        ("hypertension", "Hypertension"),
        ("diabetes", "Diabetes"),
        ("pneumonia", "Pneumonia"),
        ("angina", "Angina"),
        ("viral cough", "Viral cough"),
        ("asthma", "Asthma"),
        ("copd", "COPD"),
    ],
    "symptom": [
        ("shortness of breath", "Shortness of breath"),
        ("elevated blood pressure", "Elevated blood pressure"),
        ("chest pain", "Chest pain"),
        ("dizziness", "Dizziness"),
        ("headache", "Headache"),
        ("fatigue", "Fatigue"),
        ("nausea", "Nausea"),
        ("cough", "Cough"),
        ("fever", "Fever"),
    ],
    "procedure": [
        ("blood pressure check", "Blood pressure check"),
        ("electrocardiogram", "ECG"),
        ("chest x-ray", "Chest X-ray"),
        ("glucose test", "Glucose test"),
        ("lipid panel", "Lipid panel"),
        ("blood draw", "Blood draw"),
        ("ct scan", "CT scan"),
        ("lab test", "Lab test"),
        ("x-ray", "X-ray"),
        ("ekg", "ECG"),
        ("ecg", "ECG"),
        ("mri", "MRI"),
    ],
    "medication": [
        ("aspirin 81 mg", "Aspirin"),
        ("atorvastatin", "Atorvastatin"),
        ("metformin", "Metformin"),
        ("lisinopril", "Lisinopril"),
        ("albuterol", "Albuterol"),
        ("ibuprofen", "Ibuprofen"),
        ("amlodipine", "Amlodipine"),
        ("aspirin", "Aspirin"),
    ],
}

DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"),
]
ICD_PATTERN = re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?\b")
CPT_PATTERN = re.compile(r"\b\d{5}\b")


@lru_cache(maxsize=1)
def load_nlp_models() -> dict:
    """Attempt to load optional NLP components without crashing."""
    warnings: list[str] = []
    spacy_available = False
    scispacy_model = None
    bc5cdr_model = None
    bioclinicalbert_available = False

    try:
        import spacy  # type: ignore

        spacy_available = True
    except ImportError:
        warnings.append(FALLBACK_WARNING)
        return {
            "spacy_available": False,
            "scispacy_model": None,
            "bc5cdr_model": None,
            "bioclinicalbert_available": False,
            "warnings": warnings,
        }

    try:
        scispacy_model = spacy.load("en_core_sci_sm")
    except Exception:
        warnings.append(FALLBACK_WARNING)

    try:
        bc5cdr_model = spacy.load("en_ner_bc5cdr_md")
    except Exception:
        warnings.append("Optional scispaCy BC5CDR model is unavailable.")

    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        _ = torch
        _ = AutoModel
        _ = AutoTokenizer
        bioclinicalbert_available = True
    except ImportError:
        warnings.append("Optional Bio_ClinicalBERT dependencies are unavailable.")

    return {
        "spacy_available": spacy_available,
        "scispacy_model": scispacy_model,
        "bc5cdr_model": bc5cdr_model,
        "bioclinicalbert_available": bioclinicalbert_available,
        "warnings": warnings,
    }


def extract_entities(
    clean_text: str,
    sections: dict,
    use_advanced_models: bool = False,
) -> dict:
    """Extract clinical entities using rules first, then optional local models."""
    section_map = sections.get("sections", sections) if isinstance(sections, dict) else {}
    if not isinstance(section_map, dict):
        section_map = {}

    model_registry = load_nlp_models()
    model_info = {
        "rule_based": True,
        "scispacy_used": False,
        "bc5cdr_used": False,
        "bioclinicalbert_used": False,
        "scispacy_available": model_registry["scispacy_model"] is not None,
        "bc5cdr_available": model_registry["bc5cdr_model"] is not None,
        "bioclinicalbert_available": model_registry["bioclinicalbert_available"],
        "warnings": list(model_registry["warnings"]),
    }

    result = {
        "model_info": model_info,
        "patient": [],
        "provider": [],
        "dates": [],
        "diagnoses": [],
        "symptoms": [],
        "procedures": [],
        "medications": [],
        "codes": [],
        "signatures": [],
        "all_entities": [],
    }

    if not clean_text.strip():
        return result

    merged_entities: dict[tuple[str, str], dict] = {}

    for entity in _extract_rule_based_entities(clean_text, section_map):
        _merge_entity(merged_entities, entity)

    if model_registry["scispacy_model"] is not None:
        model_info["scispacy_used"] = True
        for entity in _extract_scispacy_entities(
            clean_text,
            section_map,
            model_registry["scispacy_model"],
            source_name="scispacy",
        ):
            _merge_entity(merged_entities, entity)

    if model_registry["bc5cdr_model"] is not None:
        model_info["bc5cdr_used"] = True
        for entity in _extract_scispacy_entities(
            clean_text,
            section_map,
            model_registry["bc5cdr_model"],
            source_name="bc5cdr",
        ):
            _merge_entity(merged_entities, entity)

    if use_advanced_models:
        semantic_enhancer, biobert_warnings = _load_bioclinicalbert()
        if biobert_warnings:
            model_info["warnings"].extend(biobert_warnings)
        if semantic_enhancer is not None:
            model_info["bioclinicalbert_used"] = True
            _apply_bioclinicalbert_semantic_scoring(merged_entities, semantic_enhancer)
        else:
            model_info["warnings"].append(
                "Bio_ClinicalBERT advanced mode was requested but is unavailable locally. "
                "Continuing without semantic enhancement."
            )

    all_entities = sorted(
        merged_entities.values(),
        key=lambda entity: (
            ENTITY_CATEGORY_ORDER.index(_bucket_to_category(entity["entity_type"]))
            if _bucket_to_category(entity["entity_type"]) in ENTITY_CATEGORY_ORDER
            else len(ENTITY_CATEGORY_ORDER),
            entity.get("start_char", math.inf),
            entity["normalized_text"].lower(),
        ),
    )

    for entity in all_entities:
        bucket = _bucket_to_category(entity["entity_type"])
        result[bucket].append(entity)
    result["all_entities"] = all_entities
    return result


def _extract_rule_based_entities(clean_text: str, sections: dict[str, str]) -> list[dict]:
    """Extract entities using regexes and curated dictionaries."""
    entities: list[dict] = []
    entities.extend(_extract_patient_entities(clean_text, sections))
    entities.extend(_extract_provider_entities(clean_text, sections))
    entities.extend(_extract_date_entities(clean_text, sections))
    entities.extend(_extract_signature_entities(clean_text, sections))
    entities.extend(_extract_code_entities(clean_text, sections))

    for entity_type, alias_specs in ENTITY_SPECS.items():
        entities.extend(
            _extract_dictionary_entities(
                clean_text=clean_text,
                sections=sections,
                entity_type=entity_type,
                alias_specs=alias_specs,
            )
        )
    return entities


def _extract_patient_entities(clean_text: str, sections: dict[str, str]) -> list[dict]:
    pattern = re.compile(r"Patient Name:[ \t]*([^\n]+)", re.IGNORECASE)
    return _extract_regex_entities(
        clean_text,
        sections,
        pattern,
        entity_type="patient",
        source="regex",
        normalized_transform=lambda value: value.strip(),
    )


def _extract_provider_entities(clean_text: str, sections: dict[str, str]) -> list[dict]:
    entities = _extract_regex_entities(
        clean_text,
        sections,
        re.compile(r"(?:Provider Name|Provider):[ \t]*([^\n]+)", re.IGNORECASE),
        entity_type="provider",
        source="regex",
        normalized_transform=lambda value: value.strip(),
    )

    doctor_pattern = re.compile(
        r"\b(Dr\.\s+[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3}(?:,\s*(?:MD|DO|NP|PA))?)"
    )
    entities.extend(
        _extract_regex_entities(
            clean_text,
            sections,
            doctor_pattern,
            entity_type="provider",
            source="regex",
            normalized_transform=lambda value: value.strip(),
        )
    )
    return entities


def _extract_date_entities(clean_text: str, sections: dict[str, str]) -> list[dict]:
    entities: list[dict] = []
    labeled_date_pattern = re.compile(
        r"(?:Date of Service):[ \t]*(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})",
        re.IGNORECASE,
    )
    entities.extend(
        _extract_regex_entities(
            clean_text,
            sections,
            labeled_date_pattern,
            entity_type="date",
            source="regex",
            normalized_transform=lambda value: value.strip(),
        )
    )

    for pattern in DATE_PATTERNS:
        entities.extend(
            _extract_regex_entities(
                clean_text,
                sections,
                pattern,
                entity_type="date",
                source="regex",
                normalized_transform=lambda value: value.strip(),
            )
        )
    return entities


def _extract_signature_entities(clean_text: str, sections: dict[str, str]) -> list[dict]:
    pattern = re.compile(r"(?:Provider Signature|Signature):[ \t]*([^\n]+)", re.IGNORECASE)
    entities = _extract_regex_entities(
        clean_text,
        sections,
        pattern,
        entity_type="signature",
        source="regex",
        normalized_transform=lambda value: value.strip(),
    )
    section_signature = (sections.get("provider_signature") or "").strip()
    if section_signature:
        start_char = clean_text.find(section_signature)
        if start_char != -1:
            entity = _make_entity(
                clean_text=clean_text,
                sections=sections,
                entity_type="signature",
                text=section_signature,
                normalized_text=section_signature,
                source="regex",
                start_char=start_char,
                end_char=start_char + len(section_signature),
                evidence_override=section_signature,
            )
            if entity:
                entities.append(entity)
    return entities


def _extract_code_entities(clean_text: str, sections: dict[str, str]) -> list[dict]:
    entities: list[dict] = []
    for match in re.finditer(r"(ICD-10:\s*([A-TV-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?))", clean_text, re.IGNORECASE):
        code = match.group(2).upper()
        entity = _make_entity(
            clean_text=clean_text,
            sections=sections,
            entity_type="code",
            text=code,
            normalized_text=code,
            source="regex",
            start_char=match.start(2),
            end_char=match.end(2),
            evidence_override=match.group(1),
        )
        if entity:
            entities.append(entity)

    for match in re.finditer(r"(CPT:\s*(\d{5}))", clean_text, re.IGNORECASE):
        code = match.group(2)
        entity = _make_entity(
            clean_text=clean_text,
            sections=sections,
            entity_type="code",
            text=code,
            normalized_text=code,
            source="regex",
            start_char=match.start(2),
            end_char=match.end(2),
            evidence_override=match.group(1),
        )
        if entity:
            entities.append(entity)

    for pattern in [ICD_PATTERN, CPT_PATTERN]:
        for match in pattern.finditer(clean_text):
            code = match.group(0)
            entity = _make_entity(
                clean_text=clean_text,
                sections=sections,
                entity_type="code",
                text=code,
                normalized_text=code.upper(),
                source="regex",
                start_char=match.start(),
                end_char=match.end(),
            )
            if entity:
                entities.append(entity)
    return entities


def _extract_dictionary_entities(
    clean_text: str,
    sections: dict[str, str],
    entity_type: str,
    alias_specs: list[tuple[str, str]],
) -> list[dict]:
    entities: list[dict] = []
    sorted_specs = sorted(alias_specs, key=lambda item: len(item[0]), reverse=True)
    for alias, normalized in sorted_specs:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(clean_text):
            entity = _make_entity(
                clean_text=clean_text,
                sections=sections,
                entity_type=entity_type,
                text=match.group(0),
                normalized_text=normalized,
                source="dictionary",
                start_char=match.start(),
                end_char=match.end(),
            )
            if entity:
                entities.append(entity)
    return entities


def _extract_regex_entities(
    clean_text: str,
    sections: dict[str, str],
    pattern: re.Pattern,
    entity_type: str,
    source: str,
    normalized_transform,
) -> list[dict]:
    entities: list[dict] = []
    for match in pattern.finditer(clean_text):
        capture_group = match.lastindex or 0
        if capture_group:
            entity_text = match.group(capture_group)
            start_char = match.start(capture_group)
            end_char = match.end(capture_group)
        else:
            entity_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
        entity = _make_entity(
            clean_text=clean_text,
            sections=sections,
            entity_type=entity_type,
            text=entity_text,
            normalized_text=normalized_transform(entity_text),
            source=source,
            start_char=start_char,
            end_char=end_char,
            evidence_override=match.group(0),
        )
        if entity:
            entities.append(entity)
    return entities


def _extract_scispacy_entities(
    clean_text: str,
    sections: dict[str, str],
    nlp_model,
    source_name: str,
) -> list[dict]:
    entities: list[dict] = []
    try:
        doc = nlp_model(clean_text)
    except Exception:
        return entities

    for span in getattr(doc, "ents", []):
        mapped_type = _categorize_free_text(span.text)
        if not mapped_type:
            continue
        if len(span.text.strip()) < 3:
            continue
        entity = _make_entity(
            clean_text=clean_text,
            sections=sections,
            entity_type=mapped_type,
            text=span.text,
            normalized_text=_normalize_free_text(span.text),
            source=source_name,
            start_char=span.start_char,
            end_char=span.end_char,
        )
        if entity:
            entities.append(entity)
    return entities


@lru_cache(maxsize=1)
def _load_bioclinicalbert():
    """Load Bio_ClinicalBERT locally if the weights are already available."""
    warnings: list[str] = []
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            BIOCLINICALBERT_NAME,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            BIOCLINICALBERT_NAME,
            local_files_only=True,
        )
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "torch": torch}, warnings
    except Exception:
        warnings.append(
            "Optional Bio_ClinicalBERT model weights are not available locally. "
            "Skipping semantic enhancement."
        )
        return None, warnings


def _apply_bioclinicalbert_semantic_scoring(
    merged_entities: dict[tuple[str, str], dict],
    semantic_enhancer: dict,
) -> None:
    """Use Bio_ClinicalBERT embeddings to softly adjust confidence on extracted entities."""
    tokenizer = semantic_enhancer["tokenizer"]
    model = semantic_enhancer["model"]
    torch = semantic_enhancer["torch"]

    for entity in merged_entities.values():
        evidence = entity.get("evidence", "")
        if not evidence:
            continue
        similarity = _semantic_similarity(
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            text_a=entity["normalized_text"],
            text_b=evidence,
        )
        if similarity >= 0.70:
            entity["sources"] = sorted(set(entity.get("sources", [])) | {"bioclinicalbert"})
            entity["source"] = entity["source"] if entity["source"] != "bioclinicalbert" else entity["source"]
            if entity["confidence"] == "Low":
                entity["confidence"] = "Medium"
            elif entity["confidence"] == "Medium" and entity["status_modifier"] == "affirmed":
                entity["confidence"] = "High"


def _semantic_similarity(tokenizer, model, torch, text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using mean pooled hidden states."""
    with torch.no_grad():
        encoded_a = tokenizer(text_a, return_tensors="pt", truncation=True, max_length=128)
        encoded_b = tokenizer(text_b, return_tensors="pt", truncation=True, max_length=128)
        output_a = model(**encoded_a)
        output_b = model(**encoded_b)
        embedding_a = output_a.last_hidden_state.mean(dim=1).squeeze(0)
        embedding_b = output_b.last_hidden_state.mean(dim=1).squeeze(0)
        similarity = torch.nn.functional.cosine_similarity(
            embedding_a.unsqueeze(0),
            embedding_b.unsqueeze(0),
        ).item()
    return float(similarity)


def _make_entity(
    clean_text: str,
    sections: dict[str, str],
    entity_type: str,
    text: str,
    normalized_text: str,
    source: str,
    start_char: int,
    end_char: int,
    evidence_override: str | None = None,
) -> dict | None:
    """Construct a section-aware entity record."""
    evidence = evidence_override or _find_evidence_for_span(clean_text, start_char, end_char)
    if not evidence:
        return None

    section_name = _infer_section(evidence, normalized_text or text, sections)
    status_result = detect_negation_and_uncertainty(
        {"text": text, "normalized_text": normalized_text},
        evidence,
    )
    status_modifier = status_result["status_modifier"]
    trigger = status_result["negation_trigger"] or status_result["uncertainty_trigger"]
    confidence = _score_confidence(
        source=source,
        section_name=section_name,
        evidence=evidence,
        status_modifier=status_modifier,
    )

    return {
        "entity_type": entity_type,
        "text": text.strip(),
        "normalized_text": normalized_text.strip(),
        "source": source,
        "sources": [source],
        "section": section_name,
        "evidence": evidence.strip(),
        "confidence": confidence,
        "start_char": start_char,
        "end_char": end_char,
        "status_modifier": status_modifier,
        "is_negated": status_result["is_negated"],
        "is_uncertain": status_result["is_uncertain"],
        "trigger": trigger,
    }


def _find_evidence_for_span(clean_text: str, start_char: int, end_char: int) -> str:
    """Return the closest line or sentence containing the entity span."""
    for line, line_start, line_end in _iter_lines(clean_text):
        if not line.strip():
            continue
        if start_char >= line_start and end_char <= line_end:
            return line.strip()

    window_start = max(0, start_char - 120)
    window_end = min(len(clean_text), end_char + 120)
    snippet = clean_text[window_start:window_end].strip()
    if snippet:
        return " ".join(snippet.split())
    return ""


def _iter_lines(text: str):
    """Yield text lines with character offsets."""
    cursor = 0
    for line in text.splitlines():
        start = cursor
        end = start + len(line)
        yield line, start, end
        cursor = end + 1


def _infer_section(evidence: str, entity_text: str, sections: dict[str, str]) -> str:
    """Assign the most likely section based on evidence membership."""
    evidence_lower = evidence.lower()
    entity_lower = entity_text.lower()
    for section_name, section_text in sections.items():
        lowered_section = section_text.lower()
        if evidence_lower and evidence_lower in lowered_section:
            return section_name
        if entity_lower and entity_lower in lowered_section and evidence_lower[:60] in lowered_section:
            return section_name
    for section_name, section_text in sections.items():
        if entity_lower and entity_lower in section_text.lower():
            return section_name
    return "unknown"


def _score_confidence(
    source: str,
    section_name: str,
    evidence: str,
    status_modifier: str,
) -> str:
    """Compute a simple confidence level from source strength and context."""
    if source in {"dictionary", "regex"} and section_name in STRONG_SECTIONS:
        confidence = "High"
    elif source in {"dictionary", "regex"}:
        confidence = "Medium"
    elif section_name in STRONG_SECTIONS:
        confidence = "Medium"
    else:
        confidence = "Low"

    if status_modifier != "affirmed":
        confidence = _downgrade_confidence(confidence)

    if any(marker in evidence.lower() for marker in OCR_NOISE_MARKERS):
        confidence = _downgrade_confidence(confidence)
    return confidence


def _downgrade_confidence(confidence: str) -> str:
    if confidence == "High":
        return "Medium"
    if confidence == "Medium":
        return "Low"
    return "Low"


def _merge_entity(merged_entities: dict[tuple[str, str], dict], entity: dict) -> None:
    """Merge duplicate entities while preserving strongest evidence and all sources."""
    key = (entity["entity_type"], entity["normalized_text"].lower())
    existing = merged_entities.get(key)
    if existing is None:
        merged_entities[key] = entity
        return

    existing["sources"] = sorted(set(existing.get("sources", [])) | set(entity.get("sources", [])))
    if existing["source"] == "dictionary":
        pass
    elif entity["source"] == "dictionary":
        existing["source"] = "dictionary"

    if _status_rank(entity["status_modifier"]) > _status_rank(existing["status_modifier"]):
        preserved_sources = existing["sources"]
        merged_entities[key] = entity
        merged_entities[key]["sources"] = preserved_sources
        return
    if _status_rank(entity["status_modifier"]) < _status_rank(existing["status_modifier"]):
        return

    if _confidence_rank(entity["confidence"]) > _confidence_rank(existing["confidence"]):
        preserved_sources = existing["sources"]
        merged_entities[key] = entity
        merged_entities[key]["sources"] = preserved_sources
    elif (
        _status_rank(entity["status_modifier"]) == _status_rank(existing["status_modifier"])
        and _confidence_rank(entity["confidence"]) == _confidence_rank(existing["confidence"])
        and entity.get("trigger")
        and not existing.get("trigger")
    ):
        existing["evidence"] = entity["evidence"]
        existing["section"] = entity["section"]
        existing["start_char"] = entity["start_char"]
        existing["end_char"] = entity["end_char"]
        existing["status_modifier"] = entity["status_modifier"]
        existing["is_negated"] = entity["is_negated"]
        existing["is_uncertain"] = entity["is_uncertain"]
        existing["trigger"] = entity["trigger"]
    elif (
        _confidence_rank(entity["confidence"]) == _confidence_rank(existing["confidence"])
        and len(entity.get("evidence", "")) > len(existing.get("evidence", ""))
    ):
        existing["evidence"] = entity["evidence"]
        existing["section"] = entity["section"]
        existing["start_char"] = entity["start_char"]
        existing["end_char"] = entity["end_char"]
        existing["status_modifier"] = entity["status_modifier"]
        existing["is_negated"] = entity["is_negated"]
        existing["is_uncertain"] = entity["is_uncertain"]
        existing["trigger"] = entity["trigger"]


def _confidence_rank(confidence: str) -> int:
    return {"Low": 1, "Medium": 2, "High": 3}.get(confidence, 0)


def _status_rank(status_modifier: str) -> int:
    return {"affirmed": 1, "uncertain": 2, "negated": 3}.get(status_modifier, 0)


def _categorize_free_text(text: str) -> str | None:
    lowered = text.lower().strip()
    for entity_type, alias_specs in ENTITY_SPECS.items():
        for alias, _normalized in alias_specs:
            if alias in lowered or lowered in alias:
                return entity_type

    if ICD_PATTERN.fullmatch(text) or CPT_PATTERN.fullmatch(text):
        return "code"
    if lowered.startswith("dr."):
        return "provider"
    return None


def _normalize_free_text(text: str) -> str:
    lowered = text.lower().strip()
    for alias_specs in ENTITY_SPECS.values():
        for alias, normalized in alias_specs:
            if alias == lowered or alias in lowered:
                return normalized
    return " ".join(part.capitalize() for part in text.strip().split())


def _bucket_to_category(entity_type: str) -> str:
    mapping = {
        "patient": "patient",
        "provider": "provider",
        "date": "dates",
        "diagnosis": "diagnoses",
        "symptom": "symptoms",
        "procedure": "procedures",
        "medication": "medications",
        "code": "codes",
        "signature": "signatures",
    }
    return mapping.get(entity_type, "all_entities")
