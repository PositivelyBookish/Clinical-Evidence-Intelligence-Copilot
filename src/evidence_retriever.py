"""Local RAG-style evidence retrieval for clinical note review."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.claim_support_checker import check_claim_support


SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_THRESHOLD = 0.35
TFIDF_THRESHOLD = 0.12
SUPPORTED_QUESTION_PATTERNS = [
    "does this document support",
    "what diagnoses are documented",
    "what procedures are documented",
    "what medications are mentioned",
    "is provider signature present",
    "what documentation is missing",
    "what evidence supports the claim",
    "is",
]


def chunk_document(clean_text: str, sections: dict | None = None) -> list[dict]:
    """Split a cleaned document into retrieval-friendly chunks."""
    if not clean_text.strip():
        return []

    section_map = {}
    if isinstance(sections, dict):
        section_map = sections.get("sections", sections) if isinstance(sections.get("sections", sections), dict) else {}

    chunk_records: list[dict] = []
    chunk_counter = 1
    cursor = 0

    section_items = list(section_map.items()) if section_map else [("unknown", clean_text)]
    for section_name, section_text in section_items:
        section_text = (section_text or "").strip()
        if not section_text:
            continue
        absolute_start = clean_text.find(section_text, cursor)
        if absolute_start == -1:
            absolute_start = clean_text.find(section_text)
        if absolute_start == -1:
            absolute_start = cursor
        cursor = absolute_start + len(section_text)

        for relative_start, relative_end, chunk_text in _split_section_text(section_text):
            chunk_records.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "section": section_name,
                    "text": chunk_text,
                    "start_char": absolute_start + relative_start,
                    "end_char": absolute_start + relative_end,
                }
            )
            chunk_counter += 1

    if not chunk_records:
        chunk_records.append(
            {
                "chunk_id": "chunk_001",
                "section": "unknown",
                "text": clean_text.strip(),
                "start_char": 0,
                "end_char": len(clean_text.strip()),
            }
        )
    return chunk_records


def build_retrieval_index(chunks: list[dict], method: str = "auto") -> dict:
    """Build a local retrieval index with sentence-transformers or TF-IDF fallback."""
    warnings: list[str] = []
    texts = [chunk.get("text", "") for chunk in chunks]

    if not chunks:
        return {
            "method": "tfidf",
            "chunks": [],
            "model_name": "TfidfVectorizer",
            "index_object": None,
            "warnings": ["No chunks were available for retrieval indexing."],
        }

    chosen_method = method
    if method == "auto":
        chosen_method = "sentence_transformers"

    if chosen_method == "sentence_transformers":
        sentence_model, model_warning = _load_sentence_transformer_model()
        if model_warning:
            warnings.append(model_warning)
        if sentence_model is not None:
            try:
                embeddings = sentence_model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                return {
                    "method": "sentence_transformers",
                    "chunks": chunks,
                    "model_name": SENTENCE_TRANSFORMER_MODEL,
                    "index_object": {
                        "model": sentence_model,
                        "embeddings": embeddings,
                    },
                    "warnings": warnings,
                }
            except Exception as exc:
                warnings.append(
                    f"Sentence-transformer retrieval failed and TF-IDF fallback will be used: {exc}"
                )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return {
        "method": "tfidf",
        "chunks": chunks,
        "model_name": "TfidfVectorizer",
        "index_object": {
            "vectorizer": vectorizer,
            "matrix": matrix,
        },
        "warnings": warnings,
    }


def retrieve_evidence(question: str, index: dict, top_k: int = 3) -> list[dict]:
    """Retrieve top-k document chunks for a question."""
    if not question.strip() or not index or not index.get("chunks"):
        return []

    method = index.get("method", "tfidf")
    chunks = index["chunks"]
    scored_chunks: list[dict] = []

    if method == "sentence_transformers":
        index_object = index.get("index_object") or {}
        model = index_object.get("model")
        embeddings = index_object.get("embeddings")
        if model is None or embeddings is None:
            return []
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        scores = np.dot(embeddings, query_embedding)
        threshold = SENTENCE_TRANSFORMER_THRESHOLD
    else:
        index_object = index.get("index_object") or {}
        vectorizer = index_object.get("vectorizer")
        matrix = index_object.get("matrix")
        if vectorizer is None or matrix is None:
            return []
        query_vector = vectorizer.transform([question])
        scores = cosine_similarity(query_vector, matrix).flatten()
        threshold = TFIDF_THRESHOLD

    ranked_indexes = np.argsort(scores)[::-1][:top_k]
    top_score = float(scores[ranked_indexes[0]]) if len(ranked_indexes) else 0.0
    weak_retrieval = top_score < threshold

    for rank, chunk_index in enumerate(ranked_indexes, start=1):
        chunk = chunks[int(chunk_index)]
        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "score": float(scores[int(chunk_index)]),
                "rank": rank,
                "weak_retrieval": weak_retrieval,
                "retrieval_method": method,
            }
        )
    return scored_chunks


def answer_question(
    question: str,
    retrieved_chunks: list[dict],
    entities: dict,
    documentation_result: dict | None = None,
) -> dict:
    """Answer supported clinical review questions using only retrieved evidence."""
    retrieval_method = retrieved_chunks[0]["retrieval_method"] if retrieved_chunks else "unavailable"
    weak_retrieval = retrieved_chunks[0]["weak_retrieval"] if retrieved_chunks else True
    evidence = retrieved_chunks

    if not _is_supported_question(question):
        return _build_answer(
            question=question,
            answer="Insufficient evidence found in the uploaded document for this question type.",
            answer_status="Insufficient Evidence",
            confidence="Low",
            retrieval_method=retrieval_method,
            evidence=evidence,
        )

    question_lower = question.strip().lower()

    if question_lower.startswith("does this document support "):
        target = question.strip()[len("Does this document support ") :].rstrip(" ?")
        return _answer_support_question(target, retrieved_chunks, entities, retrieval_method, weak_retrieval)

    if question_lower == "what diagnoses are documented?":
        return _answer_list_question("diagnoses", retrieved_chunks, entities, retrieval_method, weak_retrieval)

    if question_lower == "what procedures are documented?":
        return _answer_list_question("procedures", retrieved_chunks, entities, retrieval_method, weak_retrieval)

    if question_lower == "what medications are mentioned?":
        return _answer_list_question("medications", retrieved_chunks, entities, retrieval_method, weak_retrieval)

    if question_lower == "is provider signature present?":
        signatures = entities.get("signatures", [])
        if signatures:
            best = signatures[0]
            return _build_answer(
                question=question,
                answer="Yes. A provider signature appears to be present in the uploaded document.",
                answer_status="Answered",
                confidence="High",
                retrieval_method=retrieval_method,
                evidence=_ensure_evidence(best, retrieved_chunks),
                extra_note="Human review required before any clinical, coding, or payment decision.",
            )
        return _build_answer(
            question=question,
            answer="Insufficient evidence found in the uploaded document for a provider signature.",
            answer_status="Insufficient Evidence",
            confidence="Low",
            retrieval_method=retrieval_method,
            evidence=evidence,
        )

    if question_lower == "what documentation is missing?":
        return _answer_missing_documentation_question(
            question,
            retrieved_chunks,
            entities,
            documentation_result,
            retrieval_method,
            weak_retrieval,
        )

    if question_lower == "what evidence supports the claim?":
        if not retrieved_chunks:
            return _build_answer(
                question=question,
                answer="Insufficient evidence found in the uploaded document.",
                answer_status="Insufficient Evidence",
                confidence="Low",
                retrieval_method=retrieval_method,
                evidence=evidence,
            )
        answer = "Top retrieved evidence snippets are shown below. Review them to confirm whether they support the claim."
        return _build_answer(
            question=question,
            answer=answer,
            answer_status="Answered",
            confidence="Medium" if not weak_retrieval else "Low",
            retrieval_method=retrieval_method,
            evidence=evidence,
        )

    if question_lower.startswith("is ") and question_lower.endswith(" ruled out?"):
        target = question.strip()[3:-10].strip()
        return _answer_status_question(
            question=question,
            target=target,
            expected_status="negated",
            positive_answer=f"Yes. The document suggests {target} is ruled out or negated.",
            negative_answer=f"Insufficient evidence was found to conclude that {target} is ruled out.",
            retrieved_chunks=retrieved_chunks,
            entities=entities,
            retrieval_method=retrieval_method,
            weak_retrieval=weak_retrieval,
        )

    if question_lower.startswith("is ") and question_lower.endswith(" uncertain?"):
        target = question.strip()[3:-11].strip()
        return _answer_status_question(
            question=question,
            target=target,
            expected_status="uncertain",
            positive_answer=f"Yes. The document suggests {target} appears in uncertain context.",
            negative_answer=f"Insufficient evidence was found to conclude that {target} is uncertain.",
            retrieved_chunks=retrieved_chunks,
            entities=entities,
            retrieval_method=retrieval_method,
            weak_retrieval=weak_retrieval,
        )

    return _build_answer(
        question=question,
        answer="Insufficient evidence found in the uploaded document for this question type.",
        answer_status="Insufficient Evidence",
        confidence="Low",
        retrieval_method=retrieval_method,
        evidence=evidence,
    )


def _split_section_text(section_text: str) -> list[tuple[int, int, str]]:
    """Split a section into chunks near sentence boundaries."""
    max_length = 600
    min_length = 400
    text = section_text.strip()
    if len(text) <= max_length:
        return [(0, len(text), text)]

    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        remaining = len(text) - start
        if remaining <= max_length:
            chunk_text = text[start:].strip()
            if chunk_text:
                chunks.append((start, len(text), chunk_text))
            break

        window_end = min(len(text), start + max_length)
        window_text = text[start:window_end]
        split_index = _find_split_boundary(window_text, min_length=min_length)
        absolute_end = start + split_index
        chunk_text = text[start:absolute_end].strip()
        if not chunk_text:
            break
        chunks.append((start, absolute_end, chunk_text))
        start = absolute_end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def _find_split_boundary(window_text: str, min_length: int) -> int:
    """Choose a chunk boundary close to the desired range."""
    candidates = []
    for match in re.finditer(r"[.!?]\s+|\n", window_text):
        boundary = match.end()
        if boundary >= min_length:
            candidates.append(boundary)
    if candidates:
        return candidates[-1]
    return len(window_text)


@lru_cache(maxsize=1)
def _load_sentence_transformer_model():
    """Load the sentence-transformer model if available."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        return model, None
    except Exception:
        return None, (
            "Sentence-transformers retrieval is unavailable. Falling back to TF-IDF retrieval."
        )


def _is_supported_question(question: str) -> bool:
    lowered = question.strip().lower()
    return any(lowered.startswith(pattern) for pattern in SUPPORTED_QUESTION_PATTERNS)


def _answer_support_question(
    target: str,
    retrieved_chunks: list[dict],
    entities: dict,
    retrieval_method: str,
    weak_retrieval: bool,
) -> dict:
    """Answer whether the document supports a diagnosis, procedure, or code."""
    matches = _find_matching_entities(target, entities)
    if not matches:
        return _build_answer(
            question=f"Does this document support {target}?",
            answer="Insufficient evidence found in the uploaded document.",
            answer_status="Insufficient Evidence",
            confidence="Low",
            retrieval_method=retrieval_method,
            evidence=retrieved_chunks,
        )

    claim_payload = _infer_claim_payload(target, matches)
    claim_result = check_claim_support(
        claim=claim_payload,
        entities=entities,
        clean_text="",
        sections={},
    )
    item_result = claim_result["items"][0] if claim_result["items"] else None
    if item_result is None:
        return _build_answer(
            question=f"Does this document support {target}?",
            answer="Insufficient evidence found in the uploaded document.",
            answer_status="Insufficient Evidence",
            confidence="Low",
            retrieval_method=retrieval_method,
            evidence=retrieved_chunks,
        )

    if item_result["status"] == "Supported":
        answer = (
            f"Supported. The document contains affirmed evidence for {target} "
            "in a clinically relevant section."
        )
    elif item_result["status"] == "Not Supported":
        answer = (
            f"Not Supported. {item_result['reason']}"
        )
    elif item_result["status"] == "Unclear":
        answer = (
            f"Unclear. {item_result['reason']}"
        )
    else:
        answer = (
            f"Partially Supported. {item_result['reason']}"
        )

    confidence = item_result["confidence"]
    if weak_retrieval and confidence == "High":
        confidence = "Medium"
    elif weak_retrieval and confidence == "Medium":
        confidence = "Low"

    return _build_answer(
        question=f"Does this document support {target}?",
        answer=answer,
        answer_status="Answered",
        confidence=confidence,
        retrieval_method=retrieval_method,
        evidence=retrieved_chunks or [_item_to_evidence(item_result)],
    )


def _answer_list_question(
    category: str,
    retrieved_chunks: list[dict],
    entities: dict,
    retrieval_method: str,
    weak_retrieval: bool,
) -> dict:
    """Answer questions that ask for a list of documented items."""
    relevant_entities = [
        entity
        for entity in entities.get(category, [])
        if entity.get("status_modifier") == "affirmed"
    ]
    if not relevant_entities:
        return _build_answer(
            question=f"What {category} are documented?",
            answer="Insufficient evidence found in the uploaded document.",
            answer_status="Insufficient Evidence",
            confidence="Low",
            retrieval_method=retrieval_method,
            evidence=retrieved_chunks,
        )

    unique_names = []
    seen = set()
    for entity in relevant_entities:
        normalized = entity["normalized_text"]
        if normalized not in seen:
            unique_names.append(normalized)
            seen.add(normalized)
    answer = f"Documented {category}: {', '.join(unique_names)}."
    return _build_answer(
        question=f"What {category} are documented?",
        answer=answer,
        answer_status="Answered",
        confidence="Medium" if weak_retrieval else "High",
        retrieval_method=retrieval_method,
        evidence=retrieved_chunks or [_entity_to_evidence(entity) for entity in relevant_entities[:3]],
    )


def _answer_missing_documentation_question(
    question: str,
    retrieved_chunks: list[dict],
    entities: dict,
    documentation_result: dict | None,
    retrieval_method: str,
    weak_retrieval: bool,
) -> dict:
    """Answer a question about missing documentation fields."""
    missing_fields: list[str] = []
    if documentation_result and isinstance(documentation_result, dict):
        missing_fields = documentation_result.get("missing_fields", [])
    else:
        if not entities.get("patient"):
            missing_fields.append("patient identification")
        if not entities.get("provider"):
            missing_fields.append("provider name")
        if not entities.get("dates"):
            missing_fields.append("date of service")
        if not entities.get("signatures"):
            missing_fields.append("provider signature")

    if not missing_fields:
        answer = "No obvious missing core documentation fields were detected from the current extraction results."
        return _build_answer(
            question=question,
            answer=answer,
            answer_status="Answered",
            confidence="Medium" if weak_retrieval else "High",
            retrieval_method=retrieval_method,
            evidence=retrieved_chunks,
        )

    answer = f"Potentially missing documentation: {', '.join(missing_fields)}."
    return _build_answer(
        question=question,
        answer=answer,
        answer_status="Answered",
        confidence="Low" if weak_retrieval else "Medium",
        retrieval_method=retrieval_method,
        evidence=retrieved_chunks,
    )


def _answer_status_question(
    question: str,
    target: str,
    expected_status: str,
    positive_answer: str,
    negative_answer: str,
    retrieved_chunks: list[dict],
    entities: dict,
    retrieval_method: str,
    weak_retrieval: bool,
) -> dict:
    """Answer ruled-out or uncertainty status questions for a target condition."""
    matches = _find_matching_entities(target, entities)
    status_matches = [entity for entity in matches if entity.get("status_modifier") == expected_status]
    if not status_matches:
        return _build_answer(
            question=question,
            answer=negative_answer,
            answer_status="Insufficient Evidence",
            confidence="Low",
            retrieval_method=retrieval_method,
            evidence=retrieved_chunks,
        )
    best_match = _select_best_match(status_matches)
    return _build_answer(
        question=question,
        answer=positive_answer,
        answer_status="Answered",
        confidence="Medium" if weak_retrieval else "High",
        retrieval_method=retrieval_method,
        evidence=_ensure_evidence(best_match, retrieved_chunks),
    )


def _find_matching_entities(target: str, entities: dict) -> list[dict]:
    """Find extracted entities that match the question target."""
    target_normalized = _normalize_text(target)
    matches: list[dict] = []
    for bucket_name in [
        "diagnoses",
        "symptoms",
        "procedures",
        "medications",
        "codes",
        "signatures",
        "provider",
        "patient",
    ]:
        for entity in entities.get(bucket_name, []):
            entity_normalized = _normalize_text(entity.get("normalized_text", entity.get("text", "")))
            if target_normalized == entity_normalized:
                matches.append(entity)
            elif target_normalized in entity_normalized or entity_normalized in target_normalized:
                matches.append(entity)
    return matches


def _select_best_match(matches: list[dict]) -> dict:
    """Select the strongest entity match for answering."""
    return sorted(
        matches,
        key=lambda entity: (
            _status_rank(entity.get("status_modifier", "affirmed")),
            _confidence_rank(entity.get("confidence", "Low")),
            len(entity.get("evidence", "")),
        ),
        reverse=True,
    )[0]


def _ensure_evidence(best_entity: dict, retrieved_chunks: list[dict]) -> list[dict]:
    """Ensure evidence snippets are always visible."""
    if retrieved_chunks:
        return retrieved_chunks
    return [_entity_to_evidence(best_entity)]


def _entity_to_evidence(entity: dict) -> dict:
    """Convert an extracted entity into an evidence-like snippet."""
    return {
        "chunk_id": f"entity_{entity.get('entity_type', 'item')}",
        "section": entity.get("section", "unknown"),
        "text": entity.get("evidence", ""),
        "score": 1.0,
        "rank": 1,
        "weak_retrieval": False,
        "retrieval_method": "entity_context",
    }


def _item_to_evidence(item: dict) -> dict:
    """Convert a claim support item into an evidence-like snippet."""
    return {
        "chunk_id": f"claim_{item.get('claim_type', 'item')}",
        "section": "claim_support",
        "text": item.get("evidence", ""),
        "score": 1.0,
        "rank": 1,
        "weak_retrieval": False,
        "retrieval_method": "claim_support_checker",
    }


def _build_answer(
    question: str,
    answer: str,
    answer_status: str,
    confidence: str,
    retrieval_method: str,
    evidence: list[dict],
    extra_note: str | None = None,
) -> dict:
    """Construct a grounded answer payload."""
    note = "Answer generated only from retrieved document evidence."
    if extra_note:
        note = f"{note} {extra_note}"
    return {
        "question": question,
        "answer": answer,
        "answer_status": answer_status,
        "confidence": confidence,
        "retrieval_method": retrieval_method,
        "evidence": evidence,
        "grounding_note": note,
        "human_review_required": True,
    }


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _confidence_rank(confidence: str) -> int:
    return {"Low": 1, "Medium": 2, "High": 3}.get(confidence, 0)


def _status_rank(status: str) -> int:
    return {"affirmed": 3, "uncertain": 2, "negated": 1}.get(status, 0)


def _infer_claim_payload(target: str, matches: list[dict]) -> dict:
    """Infer which claim slot a support question target should populate."""
    normalized = _normalize_text(target)
    if re.fullmatch(r"[a-tv-z][0-9]{2}(?:\.[a-z0-9]{1,4})?", normalized, re.IGNORECASE) or re.fullmatch(
        r"\d{5}",
        normalized,
    ):
        return {"claimed_diagnosis": "", "claimed_procedure": "", "claimed_code": target}

    entity_types = {entity.get("entity_type") for entity in matches}
    if "procedure" in entity_types:
        return {"claimed_diagnosis": "", "claimed_procedure": target, "claimed_code": ""}
    if "code" in entity_types:
        return {"claimed_diagnosis": "", "claimed_procedure": "", "claimed_code": target}
    return {"claimed_diagnosis": target, "claimed_procedure": "", "claimed_code": ""}
