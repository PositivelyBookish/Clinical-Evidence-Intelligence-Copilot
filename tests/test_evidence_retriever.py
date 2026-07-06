"""Tests for evidence retrieval and grounded Q&A."""

from __future__ import annotations

from pathlib import Path

from src.entity_extractor import extract_entities
from src.evidence_retriever import (
    answer_question,
    build_retrieval_index,
    chunk_document,
    retrieve_evidence,
)
from src.section_detector import detect_sections
from src.text_cleaner import clean_text


def _build_context(filename: str):
    root = Path(__file__).resolve().parents[1]
    raw_text = (root / "data" / "synthetic_documents" / filename).read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned["clean_text"])
    chunks = chunk_document(cleaned["clean_text"], sections)
    entities = extract_entities(cleaned["clean_text"], sections, use_advanced_models=False)
    return cleaned["clean_text"], sections, chunks, entities


def test_chunk_document_prefers_section_chunks() -> None:
    clean_text_value, sections, chunks, _entities = _build_context("clean_supported_claim.txt")

    assert clean_text_value
    assert chunks
    assert all("chunk_id" in chunk for chunk in chunks)
    assert any(chunk["section"] == "assessment" for chunk in chunks)


def test_build_retrieval_index_auto_uses_available_method() -> None:
    _clean_text_value, sections, chunks, _entities = _build_context("clean_supported_claim.txt")
    index = build_retrieval_index(chunks, method="auto")

    assert index["method"] in {"sentence_transformers", "tfidf"}
    assert index["chunks"] == chunks


def test_retrieve_and_answer_grounded_question() -> None:
    clean_text_value, sections, chunks, entities = _build_context("clean_supported_claim.txt")
    index = build_retrieval_index(chunks, method="tfidf")
    retrieved = retrieve_evidence("Does this document support hypertension?", index, top_k=3)
    answer = answer_question(
        "Does this document support hypertension?",
        retrieved,
        entities,
    )

    assert retrieved
    assert answer["answer_status"] == "Answered"
    assert "Supported" in answer["answer"]
    assert answer["evidence"]


def test_hypertension_question_retrieves_assessment_evidence() -> None:
    _clean_text_value, _sections, chunks, _entities = _build_context("clean_supported_claim.txt")
    index = build_retrieval_index(chunks, method="tfidf")
    retrieved = retrieve_evidence("Does this document support hypertension?", index, top_k=3)

    assert retrieved
    assert any(chunk["section"] == "assessment" for chunk in retrieved)
    assert any("Hypertension" in chunk["text"] for chunk in retrieved)


def test_unsupported_question_returns_insufficient_evidence() -> None:
    _clean_text_value, _sections, chunks, entities = _build_context("clean_supported_claim.txt")
    index = build_retrieval_index(chunks, method="tfidf")
    retrieved = retrieve_evidence("What is the best treatment plan in cardiology guidelines?", index, top_k=3)
    answer = answer_question(
        "What is the best treatment plan in cardiology guidelines?",
        retrieved,
        entities,
    )

    assert answer["answer_status"] == "Insufficient Evidence"
