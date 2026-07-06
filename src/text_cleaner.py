"""Text normalization helpers for synthetic clinical notes."""

from __future__ import annotations

import re


OCR_REPLACEMENTS = {
    "Patlent": "Patient",
    "Provlder": "Provider",
    "Medicatlon": "Medication",
    "Dlagnosis": "Diagnosis",
    "Slgnature": "Signature",
    "Hlstory": "History",
}

FORMAT_REPLACEMENTS = {
    r"\bICD[\s-]*10\b": "ICD-10",
    r"\bCPT\s+Code\b": "CPT",
}


def clean_text(raw_text: str) -> dict:
    """Normalize noisy clinical text while preserving readable structure."""
    normalization_notes: list[str] = []
    cleaned_text = raw_text or ""

    if not cleaned_text:
        return {
            "clean_text": "",
            "normalization_notes": ["Input text was empty."],
        }

    original_text = cleaned_text

    # Normalize newlines first so the rest of the cleanup behaves consistently.
    cleaned_text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n")

    for incorrect, corrected in OCR_REPLACEMENTS.items():
        if incorrect in cleaned_text:
            cleaned_text = cleaned_text.replace(incorrect, corrected)
            normalization_notes.append(f"Corrected OCR-like token: {incorrect} -> {corrected}")

    for pattern, replacement in FORMAT_REPLACEMENTS.items():
        updated_text, replacement_count = re.subn(
            pattern,
            replacement,
            cleaned_text,
            flags=re.IGNORECASE,
        )
        if replacement_count:
            cleaned_text = updated_text
            normalization_notes.append(f"Normalized code formatting to {replacement}.")

    normalized_lines = []
    whitespace_changed = False
    for line in cleaned_text.split("\n"):
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        if normalized_line != line:
            whitespace_changed = True
        normalized_lines.append(normalized_line)
    cleaned_text = "\n".join(normalized_lines)
    if whitespace_changed:
        normalization_notes.append("Normalized whitespace within lines.")

    collapsed_text, blank_line_replacements = re.subn(r"\n{3,}", "\n\n", cleaned_text)
    if blank_line_replacements:
        cleaned_text = collapsed_text
        normalization_notes.append("Removed repeated blank lines.")

    cleaned_text = cleaned_text.strip()
    if cleaned_text != original_text.strip() and not normalization_notes:
        normalization_notes.append("Applied general text cleanup.")

    return {
        "clean_text": cleaned_text,
        "normalization_notes": normalization_notes,
    }
