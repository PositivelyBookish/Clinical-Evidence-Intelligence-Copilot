"""Generate bundled fallback synthetic notes and evaluation labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = DATA_DIR / "synthetic_documents"
EXPECTED_LABELS_PATH = DATA_DIR / "expected_labels.json"
SAMPLE_CLAIMS_PATH = DATA_DIR / "sample_claims.csv"


EDGE_CASE_NOTES = {
    "clean_supported_claim.txt": """SYNTHETIC CLINICAL DOCUMENT

Document Type: Claim Review Note
Patient Name: Jordan Carter
Date of Service: 2026-02-14
Provider Name: Dr. Elena Morris, MD

Chief Complaint:
Routine follow-up for blood pressure management and intermittent palpitations.

History:
The patient has known hypertension and was seen in clinic for follow-up. An ECG was performed today because of intermittent palpitations. Blood pressure remained elevated during the visit.

Assessment:
Hypertension, stable but still requiring medication management.
ICD-10: I10

Procedures:
ECG completed in office.
CPT: 93000

Medications:
Aspirin 81 mg daily
Lisinopril 10 mg daily

Provider Signature:
Dr. Elena Morris, MD

Reviewer Note:
Diagnosis and procedure are both directly documented and clearly supported.
""",
    "missing_signature.txt": """SYNTHETIC CLINICAL DOCUMENT

Document Type: Coding Validation Note
Patient Name: Taylor Brooks
Date of Service: 2026-03-01
Provider Name: Dr. Maya Patel, MD

Chief Complaint:
Follow-up for hypertension and dizziness.

History:
The patient returned for follow-up of hypertension. ECG obtained in clinic during this visit. Symptoms improved by the end of the appointment.

Assessment:
Hypertension
ICD-10: I10

Procedures:
ECG completed and reviewed.
CPT: 93000

Medications:
Hydrochlorothiazide 12.5 mg daily

Provider Signature:

Reviewer Note:
Clinical support is present, but the signature field is missing.
""",
    "unsupported_procedure.txt": """SYNTHETIC CLINICAL DOCUMENT

Document Type: Claim Review Note
Patient Name: Morgan Ellis
Date of Service: 2026-03-10
Provider Name: Dr. Priya Shah, MD

Chief Complaint:
Blood pressure follow-up.

History:
The patient was evaluated for hypertension management. The note discusses lifestyle modification and medication adherence. No cardiac procedure is documented during this visit.

Assessment:
Hypertension
ICD-10: I10

Procedures:
No procedure performed today.

Medications:
Losartan 50 mg daily

Provider Signature:
Dr. Priya Shah, MD

Reviewer Note:
Hypertension is supported, but the claimed ECG is not mentioned anywhere in the note.
""",
    "negated_diagnosis.txt": """SYNTHETIC CLINICAL DOCUMENT

Document Type: Progress Note
Patient Name: Cameron Lee
Date of Service: 2026-03-15
Provider Name: Dr. Samuel Greene, MD

Chief Complaint:
Cough and fever evaluation.

History:
The patient presented with respiratory symptoms. Chest imaging showed no focal infiltrate. There is no evidence of pneumonia. Pneumonia was ruled out after examination and review of imaging.

Assessment:
Upper respiratory symptoms without confirmed bacterial infection.

Procedures:
Chest X-ray performed.

Medications:
Acetaminophen as needed

Provider Signature:
Dr. Samuel Greene, MD

Reviewer Note:
Pneumonia should not be treated as supported because the diagnosis is explicitly negated.
""",
    "unclear_support.txt": """SYNTHETIC CLINICAL DOCUMENT

Document Type: Progress Note
Patient Name: Avery Bennett
Date of Service: 2026-03-20
Provider Name: Dr. Nora Kim, MD

Chief Complaint:
Intermittent chest pressure.

History:
The patient reports intermittent chest pressure with exertion. Possible angina is being considered. There is concern for angina pending further workup and additional testing.

Assessment:
Possible angina, not yet confirmed.

Procedures:
Further cardiac workup planned.

Medications:
Aspirin 81 mg daily

Provider Signature:
Dr. Nora Kim, MD

Reviewer Note:
Angina is uncertain and should be marked unclear rather than fully supported.
""",
    "incomplete_documentation.txt": """SYNTHETIC CLINICAL DOCUMENT

Document Type: Claim Review Note
Patient Name: Riley Stone

Chief Complaint:
Follow-up.

History:
Brief note only.

Assessment:
Hypertension

Procedures:

Medications:

Provider Signature:

Reviewer Note:
Very sparse note with missing date of service, missing provider, and missing signature.
""",
    "noisy_ocr_style_note.txt": """SYNTHETIC CLINICAL DOCUMENT

Document Type: OCR Imported Note
Patlent Name: Casey Morgan
Date  of  Service: 2026-04-04
Provlder: Dr. Lila Gomez, MD

Chief   Complaint:
Follow-up  for  blood  pressure  and  chest  discomfort

Hlstory:
The patlent was seen for hypertenslon follow-up. An EKG was performed in cllnic.
Medicatlon adherence was reviewed.

Dlagnosis:
Hypertenslon
ICD-10: I10

Procedures:
EKG completed.
CPT: 93000

Medicatlon:
Aspirin 81 mg dally

Provider Slgnature:
Dr. Lila Gomez, MD

Reviewer Note:
This note intentionally includes OCR-like misspellings, extra spaces, and broken formatting.
""",
    "synthetic_discharge_note_001.txt": """SYNTHETIC CLINICAL DOCUMENT

Document ID: SYN-DS-001
Document Type: Discharge Summary

Chief Complaint:
Shortness of breath, fever, and productive cough.

History of Present Illness:
The patient presented with three days of worsening cough and fever. Chest imaging was concerning for right lower lobe pneumonia. The patient was admitted for IV antibiotics and supportive care.

Assessment:
Community-acquired pneumonia with hypoxia, improving on treatment.
ICD-10: J18.9

Plan:
Complete antibiotic course and follow up with primary care.

Provider Signature:
Dr. Synthetic Discharge, MD

Disclaimer:
This is a synthetic note created for demonstration only.
""",
    "synthetic_ed_note_002.txt": """SYNTHETIC CLINICAL DOCUMENT

Document ID: SYN-ED-002
Document Type: Emergency Department Note

Chief Complaint:
Chest pain.

History of Present Illness:
The patient reports intermittent chest discomfort after exertion. Denies shortness of breath. Denies fever. No clear documentation of acute myocardial infarction appears in this note.

Assessment:
Chest pain under evaluation. Acute myocardial infarction not confirmed in this record.

Plan:
Observe, trend labs, and obtain cardiology follow-up as needed.

Provider Signature:
Dr. Synthetic ED, MD

Disclaimer:
This is a synthetic note created for demonstration only.
""",
}


EXPECTED_LABELS = {
    "clean_supported_claim.txt": {
        "diagnoses": ["hypertension"],
        "procedures": ["ECG"],
        "medications": ["aspirin", "lisinopril"],
        "codes": ["I10", "93000"],
        "missing_fields": [],
        "negated_entities": [],
        "uncertain_entities": [],
        "completeness": "High",
    },
    "missing_signature.txt": {
        "diagnoses": ["hypertension"],
        "procedures": ["ECG"],
        "medications": ["hydrochlorothiazide"],
        "codes": ["I10", "93000"],
        "missing_fields": ["provider_signature"],
        "negated_entities": [],
        "uncertain_entities": [],
        "completeness": "Medium",
    },
    "unsupported_procedure.txt": {
        "diagnoses": ["hypertension"],
        "procedures": [],
        "medications": ["losartan"],
        "codes": ["I10"],
        "missing_fields": [],
        "negated_entities": [],
        "uncertain_entities": [],
        "completeness": "Medium",
    },
    "negated_diagnosis.txt": {
        "diagnoses": [],
        "procedures": ["Chest X-ray"],
        "medications": ["acetaminophen"],
        "codes": ["71046"],
        "missing_fields": [],
        "negated_entities": ["pneumonia"],
        "uncertain_entities": [],
        "completeness": "High",
    },
    "unclear_support.txt": {
        "diagnoses": [],
        "procedures": [],
        "medications": ["aspirin"],
        "codes": [],
        "missing_fields": [],
        "negated_entities": [],
        "uncertain_entities": ["angina"],
        "completeness": "Medium",
    },
    "incomplete_documentation.txt": {
        "diagnoses": ["hypertension"],
        "procedures": [],
        "medications": [],
        "codes": [],
        "missing_fields": ["date_of_service", "provider_name", "provider_signature"],
        "negated_entities": [],
        "uncertain_entities": [],
        "completeness": "Low",
    },
    "noisy_ocr_style_note.txt": {
        "diagnoses": ["hypertension"],
        "procedures": ["EKG"],
        "medications": ["aspirin"],
        "codes": ["I10", "93000"],
        "missing_fields": [],
        "negated_entities": [],
        "uncertain_entities": [],
        "completeness": "Medium",
    },
    "synthetic_discharge_note_001.txt": {
        "diagnoses": ["pneumonia"],
        "procedures": [],
        "medications": [],
        "codes": ["J18.9"],
        "missing_fields": [],
        "negated_entities": [],
        "uncertain_entities": [],
        "completeness": "High",
    },
    "synthetic_ed_note_002.txt": {
        "diagnoses": ["chest pain"],
        "procedures": [],
        "medications": [],
        "codes": ["R07.9"],
        "missing_fields": [],
        "negated_entities": ["acute myocardial infarction"],
        "uncertain_entities": [],
        "completeness": "Medium",
    },
}


SAMPLE_CLAIMS = [
    {
        "document_file": "clean_supported_claim.txt",
        "claimed_diagnosis": "Hypertension",
        "claimed_procedure": "ECG",
        "claimed_code": "I10",
        "expected_diagnosis_status": "Supported",
        "expected_procedure_status": "Supported",
        "expected_code_status": "Supported",
    },
    {
        "document_file": "missing_signature.txt",
        "claimed_diagnosis": "Hypertension",
        "claimed_procedure": "ECG",
        "claimed_code": "I10",
        "expected_diagnosis_status": "Supported",
        "expected_procedure_status": "Supported",
        "expected_code_status": "Supported",
    },
    {
        "document_file": "unsupported_procedure.txt",
        "claimed_diagnosis": "Hypertension",
        "claimed_procedure": "ECG",
        "claimed_code": "93000",
        "expected_diagnosis_status": "Supported",
        "expected_procedure_status": "Not Supported",
        "expected_code_status": "Not Supported",
    },
    {
        "document_file": "negated_diagnosis.txt",
        "claimed_diagnosis": "Pneumonia",
        "claimed_procedure": "",
        "claimed_code": "J18.9",
        "expected_diagnosis_status": "Negated",
        "expected_procedure_status": "Missing Evidence",
        "expected_code_status": "Not Supported",
    },
    {
        "document_file": "unclear_support.txt",
        "claimed_diagnosis": "Angina",
        "claimed_procedure": "",
        "claimed_code": "",
        "expected_diagnosis_status": "Unclear",
        "expected_procedure_status": "Missing Evidence",
        "expected_code_status": "Missing Evidence",
    },
    {
        "document_file": "incomplete_documentation.txt",
        "claimed_diagnosis": "Hypertension",
        "claimed_procedure": "",
        "claimed_code": "I10",
        "expected_diagnosis_status": "Supported",
        "expected_procedure_status": "Missing Evidence",
        "expected_code_status": "Unclear",
    },
    {
        "document_file": "noisy_ocr_style_note.txt",
        "claimed_diagnosis": "Hypertension",
        "claimed_procedure": "EKG",
        "claimed_code": "93000",
        "expected_diagnosis_status": "Supported",
        "expected_procedure_status": "Supported",
        "expected_code_status": "Supported",
    },
    {
        "document_file": "synthetic_discharge_note_001.txt",
        "claimed_diagnosis": "Pneumonia",
        "claimed_procedure": "",
        "claimed_code": "J18.9",
        "expected_diagnosis_status": "Supported",
        "expected_procedure_status": "Missing Evidence",
        "expected_code_status": "Supported",
    },
    {
        "document_file": "synthetic_ed_note_002.txt",
        "claimed_diagnosis": "Acute myocardial infarction",
        "claimed_procedure": "",
        "claimed_code": "",
        "expected_diagnosis_status": "Negated",
        "expected_procedure_status": "Missing Evidence",
        "expected_code_status": "Missing Evidence",
    },
]


def write_notes() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in EDGE_CASE_NOTES.items():
        (DOCS_DIR / filename).write_text(content, encoding="utf-8")


def write_expected_labels() -> None:
    EXPECTED_LABELS_PATH.write_text(json.dumps(EXPECTED_LABELS, indent=2), encoding="utf-8")


def write_sample_claims() -> None:
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
        writer.writerows(SAMPLE_CLAIMS)


def main() -> int:
    write_notes()
    write_expected_labels()
    write_sample_claims()
    print(f"Wrote {len(EDGE_CASE_NOTES)} synthetic note files to {DOCS_DIR}")
    print(f"Wrote expected labels to {EXPECTED_LABELS_PATH}")
    print(f"Wrote sample claims to {SAMPLE_CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
