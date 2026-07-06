"""Smoke tests for the initial project scaffold."""

from pathlib import Path


def test_expected_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_paths = [
        root / "app.py",
        root / "requirements.txt",
        root / "README.md",
        root / "data" / "sample_claims.csv",
        root / "data" / "expected_labels.json",
        root / "data" / "synthea_raw" / "fhir",
        root / "data" / "synthea_raw" / "csv",
        root / "src" / "document_loader.py",
        root / "src" / "claim_support_checker.py",
        root / "scripts" / "setup_synthea_instructions.md",
        root / "scripts" / "generate_synthea_data.py",
        root / "scripts" / "convert_synthea_to_notes.py",
        root / "scripts" / "create_fallback_synthetic_notes.py",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing expected path: {path}"


def test_expected_edge_case_documents_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    document_dir = root / "data" / "synthetic_documents"
    expected_documents = [
        "clean_supported_claim.txt",
        "missing_signature.txt",
        "unsupported_procedure.txt",
        "negated_diagnosis.txt",
        "unclear_support.txt",
        "incomplete_documentation.txt",
        "noisy_ocr_style_note.txt",
    ]
    for filename in expected_documents:
        assert (document_dir / filename).exists(), f"Missing synthetic document: {filename}"
