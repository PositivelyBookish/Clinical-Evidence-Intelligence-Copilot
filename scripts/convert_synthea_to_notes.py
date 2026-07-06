"""Convert Synthea FHIR JSON records into readable clinical note text files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FHIR_DIR = REPO_ROOT / "data" / "synthea_raw" / "fhir"
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_documents"
EXPECTED_LABELS_PATH = REPO_ROOT / "data" / "expected_labels.json"
SAMPLE_CLAIMS_PATH = REPO_ROOT / "data" / "sample_claims.csv"

NOTE_TYPES = [
    "discharge_summary",
    "progress_note",
    "coding_validation_note",
    "claim_review_note",
]

DIAGNOSIS_CODE_MAP = {
    "hypertension": "I10",
    "diabetes": "E11.9",
    "pneumonia": "J18.9",
    "chest pain": "R07.9",
}

PROCEDURE_CODE_MAP = {
    "ecg": "93000",
    "ekg": "93000",
    "chest x-ray": "71046",
    "glucose test": "82947",
}


@dataclass
class PatientSummary:
    patient_id: str
    patient_name: str
    birth_date: str = "Unknown"
    gender: str = "Unknown"
    provider_name: str = "Dr. Synthetic Provider, MD"
    encounters: list[dict] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Synthea FHIR JSON files into clinical note text documents."
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=10,
        help="Minimum number of note files to create when FHIR data is available.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_existing_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_existing_claims(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resource_iter(payload: dict) -> list[dict]:
    if payload.get("resourceType") == "Bundle":
        return [entry["resource"] for entry in payload.get("entry", []) if "resource" in entry]
    return [payload]


def extract_reference_id(reference: str | None) -> str | None:
    if not reference:
        return None
    return reference.split("/")[-1]


def get_code_display(resource: dict) -> str:
    code = resource.get("code", {})
    if code.get("text"):
        return str(code["text"])
    for coding in code.get("coding", []):
        if coding.get("display"):
            return str(coding["display"])
    return "Not documented"


def get_medication_display(resource: dict) -> str:
    med = resource.get("medicationCodeableConcept", {})
    if med.get("text"):
        return str(med["text"])
    for coding in med.get("coding", []):
        if coding.get("display"):
            return str(coding["display"])
    return "Medication not specified"


def get_observation_summary(resource: dict) -> str:
    label = get_code_display(resource)
    value = (
        resource.get("valueString")
        or resource.get("valueCodeableConcept", {}).get("text")
        or resource.get("valueQuantity", {}).get("value")
        or resource.get("valueQuantity", {}).get("unit")
        or resource.get("valueBoolean")
    )
    if value in (None, ""):
        return label
    return f"{label}: {value}"


def get_patient_name(resource: dict) -> str:
    names = resource.get("name", [])
    if not names:
        return "Synthetic Patient"
    first = names[0]
    given = " ".join(first.get("given", []))
    family = first.get("family", "")
    full_name = " ".join(part for part in [given, family] if part).strip()
    return full_name or "Synthetic Patient"


def extract_provider_name(resource: dict) -> str | None:
    participants = resource.get("participant", [])
    for participant in participants:
        individual = participant.get("individual", {})
        if individual.get("display"):
            return str(individual["display"])
    service_provider = resource.get("serviceProvider", {})
    if service_provider.get("display"):
        return str(service_provider["display"])
    return None


def normalize_date(value: str | None) -> str:
    if not value:
        return "Unknown"
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned).strftime("%Y-%m-%d")
    except ValueError:
        return value[:10]


def map_diagnosis_code(diagnosis: str) -> str:
    diagnosis_lower = diagnosis.lower()
    for key, code in DIAGNOSIS_CODE_MAP.items():
        if key in diagnosis_lower:
            return code
    return "R69"


def map_procedure_code(procedure: str) -> str:
    procedure_lower = procedure.lower()
    for key, code in PROCEDURE_CODE_MAP.items():
        if key in procedure_lower:
            return code
    return "00000"


def collect_patient_summaries() -> dict[str, PatientSummary]:
    patients: dict[str, PatientSummary] = {}
    grouped_resources: dict[str, list[dict]] = defaultdict(list)

    json_files = sorted(FHIR_DIR.glob("*.json"))
    for json_file in json_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Skipping unreadable JSON file: {json_file}")
            continue

        for resource in resource_iter(payload):
            resource_type = resource.get("resourceType")
            if not resource_type:
                continue
            if resource_type == "Patient":
                patient_id = resource.get("id", json_file.stem)
                summary = PatientSummary(
                    patient_id=patient_id,
                    patient_name=get_patient_name(resource),
                    birth_date=resource.get("birthDate", "Unknown"),
                    gender=resource.get("gender", "Unknown"),
                )
                patients[patient_id] = summary
            else:
                grouped_resources[resource_type].append(resource)

    if not patients:
        return {}

    for encounter in grouped_resources.get("Encounter", []):
        patient_id = extract_reference_id(encounter.get("subject", {}).get("reference"))
        if patient_id not in patients:
            continue
        patients[patient_id].encounters.append(encounter)
        provider_name = extract_provider_name(encounter)
        if provider_name:
            patients[patient_id].provider_name = provider_name

    for condition in grouped_resources.get("Condition", []):
        patient_id = extract_reference_id(condition.get("subject", {}).get("reference"))
        if patient_id in patients:
            patients[patient_id].conditions.append(get_code_display(condition))

    for procedure in grouped_resources.get("Procedure", []):
        patient_id = extract_reference_id(procedure.get("subject", {}).get("reference"))
        if patient_id in patients:
            patients[patient_id].procedures.append(get_code_display(procedure))

    for medication in grouped_resources.get("MedicationRequest", []):
        patient_id = extract_reference_id(medication.get("subject", {}).get("reference"))
        if patient_id in patients:
            patients[patient_id].medications.append(get_medication_display(medication))

    for observation in grouped_resources.get("Observation", []):
        patient_id = extract_reference_id(observation.get("subject", {}).get("reference"))
        if patient_id in patients:
            patients[patient_id].observations.append(get_observation_summary(observation))

    return patients


def get_date_of_service(summary: PatientSummary) -> str:
    for encounter in summary.encounters:
        period = encounter.get("period", {})
        if period.get("start"):
            return normalize_date(period["start"])
    return "2026-01-01"


def format_list(items: list[str], fallback: str) -> str:
    filtered = [item for item in items if item]
    return "\n".join(f"- {item}" for item in filtered[:5]) if filtered else f"- {fallback}"


def build_note(summary: PatientSummary, note_type: str, reviewer_note: str) -> str:
    date_of_service = get_date_of_service(summary)
    chief_complaint = summary.conditions[0] if summary.conditions else "General follow-up review"
    primary_diagnosis = summary.conditions[0] if summary.conditions else "Condition not specified"
    primary_procedure = summary.procedures[0] if summary.procedures else "No procedure documented"
    icd_code = map_diagnosis_code(primary_diagnosis)
    cpt_code = map_procedure_code(primary_procedure) if summary.procedures else "Not applicable"

    return (
        "SYNTHETIC CLINICAL DOCUMENT\n\n"
        f"Document Type: {note_type.replace('_', ' ').title()}\n"
        f"Patient Name: {summary.patient_name}\n"
        f"Birth Date: {summary.birth_date}\n"
        f"Gender: {summary.gender}\n"
        f"Date of Service: {date_of_service}\n"
        f"Provider Name: {summary.provider_name}\n\n"
        f"Chief Complaint:\n{chief_complaint}\n\n"
        "History:\n"
        f"The synthetic patient record was converted from structured Synthea data. "
        f"Relevant prior conditions include {', '.join(summary.conditions[:3]) or 'no major conditions captured'}.\n\n"
        "Assessment:\n"
        f"Primary diagnosis under review: {primary_diagnosis}\n"
        f"ICD-10-like Code: {icd_code}\n\n"
        "Procedures:\n"
        f"{format_list(summary.procedures, 'No procedures documented in the synthetic record.')}\n"
        f"CPT-like Code: {cpt_code}\n\n"
        "Medications:\n"
        f"{format_list(summary.medications, 'No active medications listed.')}\n\n"
        "Observations:\n"
        f"{format_list(summary.observations, 'No structured observations extracted.')}\n\n"
        "Provider Signature:\n"
        f"{summary.provider_name}\n\n"
        "Reviewer Note:\n"
        f"{reviewer_note}\n\n"
        "Safety Notice:\n"
        "This is a synthetic note derived from Synthea output for research and demonstration use only.\n"
    )


def write_note_file(filename: str, content: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")


def update_expected_labels(filename: str, summary: PatientSummary) -> None:
    existing = load_existing_json(EXPECTED_LABELS_PATH)
    diagnoses = [condition.lower() for condition in summary.conditions[:3]]
    procedures = summary.procedures[:3]
    medications = summary.medications[:3]
    codes = []
    if summary.conditions:
        codes.append(map_diagnosis_code(summary.conditions[0]))
    if summary.procedures:
        codes.append(map_procedure_code(summary.procedures[0]))

    existing[filename] = {
        "diagnoses": diagnoses,
        "procedures": procedures,
        "medications": medications,
        "codes": codes,
        "missing_fields": [],
        "negated_entities": [],
        "uncertain_entities": [],
        "completeness": "High",
    }
    EXPECTED_LABELS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def update_sample_claims(filename: str, summary: PatientSummary) -> None:
    existing_rows = load_existing_claims(SAMPLE_CLAIMS_PATH)
    key = (filename, summary.conditions[0] if summary.conditions else "", summary.procedures[0] if summary.procedures else "")
    seen = {
        (
            row.get("document_file", ""),
            row.get("claimed_diagnosis", ""),
            row.get("claimed_procedure", ""),
        )
        for row in existing_rows
    }
    if key in seen:
        return

    existing_rows.append(
        {
            "document_file": filename,
            "claimed_diagnosis": summary.conditions[0] if summary.conditions else "",
            "claimed_procedure": summary.procedures[0] if summary.procedures else "",
            "claimed_code": map_diagnosis_code(summary.conditions[0]) if summary.conditions else "",
            "expected_diagnosis_status": "Supported" if summary.conditions else "Missing Evidence",
            "expected_procedure_status": "Supported" if summary.procedures else "Missing Evidence",
            "expected_code_status": "Supported" if summary.conditions else "Missing Evidence",
        }
    )

    fieldnames = [
        "document_file",
        "claimed_diagnosis",
        "claimed_procedure",
        "claimed_code",
        "expected_diagnosis_status",
        "expected_procedure_status",
        "expected_code_status",
    ]
    with SAMPLE_CLAIMS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def generate_notes(patients: dict[str, PatientSummary], min_notes: int) -> int:
    generated_files: list[str] = []
    patient_items = sorted(patients.values(), key=lambda item: item.patient_id)
    reviewer_notes = {
        "discharge_summary": "Confirm the discharge diagnosis and supporting findings.",
        "progress_note": "Check whether the assessment is supported within the daily note context.",
        "coding_validation_note": "Validate whether diagnosis and procedure coding appear justified by the source note.",
        "claim_review_note": "Review the chart for evidence supporting the claimed service.",
    }

    for patient_index, summary in enumerate(patient_items, start=1):
        for note_type in NOTE_TYPES:
            filename = f"synthea_patient_{patient_index:03d}_{note_type}.txt"
            content = build_note(summary, note_type, reviewer_notes[note_type])
            write_note_file(filename, content)
            update_expected_labels(filename, summary)
            update_sample_claims(filename, summary)
            generated_files.append(filename)

    supplemental_index = 1
    while generated_files and len(generated_files) < min_notes:
        summary = patient_items[(supplemental_index - 1) % len(patient_items)]
        filename = f"synthea_patient_extra_{supplemental_index:03d}_progress_note.txt"
        content = build_note(
            summary,
            "progress_note",
            "Supplemental synthetic progress note generated to ensure adequate demo volume.",
        )
        write_note_file(filename, content)
        update_expected_labels(filename, summary)
        update_sample_claims(filename, summary)
        generated_files.append(filename)
        supplemental_index += 1

    return len(generated_files)


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FHIR_DIR.exists() or not any(FHIR_DIR.glob("*.json")):
        print(f"No FHIR JSON files found in {FHIR_DIR}")
        print("Add Synthea output first or use scripts/create_fallback_synthetic_notes.py.")
        return 0

    patients = collect_patient_summaries()
    if not patients:
        print("FHIR files were found, but no Patient resources could be parsed.")
        return 0

    generated_count = generate_notes(patients, min_notes=args.min_notes)
    print(f"Generated {generated_count} Synthea-derived clinical note files in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
