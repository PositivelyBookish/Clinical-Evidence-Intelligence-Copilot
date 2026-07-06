"""Rule-based section detection for clinical note-style text."""

from __future__ import annotations

import re


SECTION_PATTERNS = [
    (
        "chief_complaint",
        re.compile(r"^chief complaint(?:\s*:\s*(.*))?$", re.IGNORECASE),
    ),
    (
        "history",
        re.compile(
            r"^(history(?: of present illness)?|hpi)(?:\s*:\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    ("assessment", re.compile(r"^assessment(?:\s*:\s*(.*))?$", re.IGNORECASE)),
    ("diagnosis", re.compile(r"^diagnosis(?:\s*:\s*(.*))?$", re.IGNORECASE)),
    ("procedures", re.compile(r"^procedures?(?:\s*:\s*(.*))?$", re.IGNORECASE)),
    ("medications", re.compile(r"^medications?(?:\s*:\s*(.*))?$", re.IGNORECASE)),
    ("plan", re.compile(r"^plan(?:\s*:\s*(.*))?$", re.IGNORECASE)),
    ("codes", re.compile(r"^(icd-10|cpt)(?:\s*:\s*(.*))?$", re.IGNORECASE)),
    (
        "provider_signature",
        re.compile(
            r"^(provider signature|signature)(?:\s*:\s*(.*))?$",
            re.IGNORECASE,
        ),
    ),
    (
        "reviewer_note",
        re.compile(r"^reviewer note(?:\s*:\s*(.*))?$", re.IGNORECASE),
    ),
    (
        "patient_info",
        re.compile(r"^patient (name|information)(?:\s*:\s*(.*))?$", re.IGNORECASE),
    ),
]

PATIENT_INFO_PREFIXES = (
    "patient name",
    "date of service",
    "birth date",
    "gender",
    "document type",
    "document id",
    "provider name",
)


def detect_sections(text: str) -> dict:
    """Split cleaned clinical text into simple semantic sections."""
    lines = text.splitlines() if text else []
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    unknown_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_section and sections.get(current_section):
                sections[current_section].append("")
            continue

        matched_section, inline_content = _match_section_heading(line)
        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
            if inline_content:
                sections[current_section].append(inline_content)
            continue

        if _is_patient_info_line(line):
            current_section = "patient_info"
            sections.setdefault(current_section, [])
            sections[current_section].append(line)
            continue

        if current_section:
            sections.setdefault(current_section, [])
            sections[current_section].append(line)
        else:
            unknown_lines.append(line)

    if unknown_lines:
        sections["unknown"] = unknown_lines
    if not sections and text.strip():
        sections["unknown"] = [text.strip()]

    collapsed_sections = {
        name: _collapse_lines(values)
        for name, values in sections.items()
        if _collapse_lines(values)
    }
    detected_names = list(collapsed_sections.keys())

    return {
        "sections": collapsed_sections,
        "detected_section_names": detected_names,
        "section_count": len(detected_names),
    }


def _match_section_heading(line: str) -> tuple[str | None, str]:
    """Return detected section name and same-line content if present."""
    for section_name, pattern in SECTION_PATTERNS:
        match = pattern.match(line)
        if not match:
            continue

        inline_groups = match.groups()
        inline_content = ""
        if section_name == "codes":
            heading_name = inline_groups[0].upper()
            inline_tail = (inline_groups[1] or "").strip()
            inline_content = f"{heading_name}: {inline_tail}".strip(": ").strip()
        elif section_name == "patient_info":
            label = match.group(1).strip().title()
            value = (match.group(2) or "").strip()
            inline_content = f"Patient {label}: {value}".strip(": ").strip()
        elif inline_groups:
            inline_content = (inline_groups[-1] or "").strip()
        return section_name, inline_content
    return None, ""


def _is_patient_info_line(line: str) -> bool:
    """Heuristic for patient and document metadata lines before the clinical body."""
    lowered = line.lower()
    return any(lowered.startswith(prefix) for prefix in PATIENT_INFO_PREFIXES)


def _collapse_lines(lines: list[str]) -> str:
    """Collapse section lines while preserving readable spacing."""
    joined = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()
