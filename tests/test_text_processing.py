"""Tests for text cleaning and section detection."""

from __future__ import annotations

from pathlib import Path

from src.section_detector import detect_sections
from src.text_cleaner import clean_text


def test_clean_text_fixes_requested_ocr_tokens() -> None:
    raw_text = (
        "Patlent Name: Jane Doe\n"
        "Provlder: Dr. Smith\n"
        "Medicatlon: Aspirin\n"
        "Dlagnosis: Hypertension\n"
        "Slgnature: Dr. Smith\n"
        "ICD 10: I10\n"
        "CPT Code: 93000\n"
    )

    result = clean_text(raw_text)

    assert "Patient Name" in result["clean_text"]
    assert "Provider: Dr. Smith" in result["clean_text"]
    assert "Medication: Aspirin" in result["clean_text"]
    assert "Diagnosis: Hypertension" in result["clean_text"]
    assert "Signature: Dr. Smith" in result["clean_text"]
    assert "ICD-10: I10" in result["clean_text"]
    assert "CPT: 93000" in result["clean_text"]


def test_clean_text_normalizes_whitespace_and_blank_lines() -> None:
    raw_text = (
        "Patient Name:   Jane Doe\n\n\n"
        "Assessment:    Hypertension\n"
        "Procedures:      ECG completed.\n\n"
        "Medication:   Aspirin 81 mg daily\n"
    )

    result = clean_text(raw_text)

    assert "Patient Name: Jane Doe" in result["clean_text"]
    assert "Assessment: Hypertension" in result["clean_text"]
    assert "Procedures: ECG completed." in result["clean_text"]
    assert "\n\n\n" not in result["clean_text"]


def test_detect_sections_finds_assessment_procedure_medication_and_signature() -> None:
    clean_note = (
        "Patient Name: Jane Doe\n"
        "Assessment:\n"
        "Hypertension\n\n"
        "Procedure:\n"
        "ECG completed in office.\n\n"
        "Medication:\n"
        "Aspirin 81 mg daily\n\n"
        "Signature:\n"
        "Dr. Smith, MD\n"
    )

    result = detect_sections(clean_note)

    assert "assessment" in result["detected_section_names"]
    assert "procedures" in result["detected_section_names"]
    assert "medications" in result["detected_section_names"]
    assert "provider_signature" in result["detected_section_names"]


def test_detect_sections_handles_noisy_ocr_style_note() -> None:
    root = Path(__file__).resolve().parents[1]
    noisy_note_path = root / "data" / "synthetic_documents" / "noisy_ocr_style_note.txt"
    raw_text = noisy_note_path.read_text(encoding="utf-8")

    cleaned = clean_text(raw_text)
    result = detect_sections(cleaned["clean_text"])

    assert "History" in cleaned["clean_text"]
    assert "patient_info" in result["detected_section_names"]
    assert "chief_complaint" in result["detected_section_names"]
    assert "history" in result["detected_section_names"]
    assert "diagnosis" in result["detected_section_names"]
    assert "procedures" in result["detected_section_names"]
    assert "provider_signature" in result["detected_section_names"]
