"""Tests for the model-aware evaluation harness."""

from __future__ import annotations

from pathlib import Path

from src.evaluator import run_evaluation


def test_run_evaluation_returns_honest_structured_results() -> None:
    root = Path(__file__).resolve().parents[1]

    result = run_evaluation(
        data_dir=str(root / "data"),
        use_scispacy=False,
        use_sentence_transformers=False,
    )

    assert result["models"]["rule_based"] == "available"
    assert result["dataset_metrics"]["documents_tested"] == 9
    assert result["dataset_metrics"]["synthea_derived_documents_tested"] == 2
    assert result["dataset_metrics"]["controlled_edge_case_documents_tested"] == 7
    assert result["retrieval_metrics"]["retrieval_method_used"] == "tfidf"
    assert len(result["per_document_results"]) == 9
    assert "diagnosis_exact_match_rate" in result["entity_extraction_metrics"]["rule_based"]
    assert result["entity_extraction_metrics"]["scispacy_enhanced"]["diagnosis_exact_match_rate"] == "Not available"


def test_run_evaluation_includes_document_level_claim_and_missing_field_comparisons() -> None:
    root = Path(__file__).resolve().parents[1]
    result = run_evaluation(
        data_dir=str(root / "data"),
        use_scispacy=False,
        use_sentence_transformers=False,
    )

    by_file = {row["document_file"]: row for row in result["per_document_results"]}
    clean_supported = by_file["clean_supported_claim.txt"]
    missing_signature = by_file["missing_signature.txt"]

    assert clean_supported["claim_expected"] == "Supported"
    assert clean_supported["predicted_diagnoses"] == ["Hypertension"]
    assert missing_signature["missing_fields_expected"] == ["provider_signature"]
    assert "Provider signature" in missing_signature["missing_fields_predicted"]
