"""Streamlit app shell for the Clinical Evidence Intelligence Copilot demo."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.claim_support_checker import check_claim_support
from src.documentation_checker import check_documentation_completeness
from src.document_loader import load_document
from src.entity_extractor import extract_entities
from src.evidence_retriever import (
    answer_question,
    build_retrieval_index,
    chunk_document,
    retrieve_evidence,
)
from src.evaluator import run_evaluation
from src.export_report import (
    build_documentation_csv,
    build_entities_csv,
    build_review_report,
    build_summary_text,
)
from src.section_detector import detect_sections
from src.summarizer import generate_reviewer_summary
from src.text_cleaner import clean_text


APP_TITLE = "Clinical Evidence Intelligence Copilot"
APP_SUBTITLE = (
    "NLP + GenAI Proof of Concept for Healthcare Payment Accuracy and Clinical "
    "Evidence Review"
)
BASE_DIR = Path(__file__).resolve().parent
SYNTHETIC_DOCS_DIR = BASE_DIR / "data" / "synthetic_documents"
SAMPLE_CLAIMS_PATH = BASE_DIR / "data" / "sample_claims.csv"
EXPECTED_LABELS_PATH = BASE_DIR / "data" / "expected_labels.json"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_app_styles() -> None:
    """Apply lightweight product-style visual polish for the demo."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .app-hero {
            background: linear-gradient(135deg, #f4fbfb 0%, #eef6ff 100%);
            border: 1px solid #d6e7ec;
            border-radius: 16px;
            padding: 0.38rem 1rem;
            margin-bottom: 0.45rem;
        }
        .app-hero h1 {
            font-size: 1.3rem;
            margin: 0;
            color: #163544;
            line-height: 1.05;
        }
        .summary-card {
            background: #fbfdfd;
            border: 1px solid #dbe7eb;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
        }
        .summary-card h3 {
            margin: 0 0 0.55rem 0;
            color: #173a48;
            font-size: 1rem;
        }
        .summary-card p {
            margin: 0.1rem 0;
            color: #355765;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Populate default document state."""
    defaults = {
        "current_document_text": "",
        "current_document_name": "No document loaded",
        "processing_status": "Awaiting document selection",
        "document_source": "None",
        "last_selected_sample": None,
        "current_document_result": {
            "success": False,
            "file_name": "No document loaded",
            "file_type": "",
            "extraction_method": "none",
            "raw_text": "",
            "warnings": [],
            "errors": [],
            "text_quality_score": 0.0,
        },
        "current_document_raw_text": "",
        "current_document_clean_text": "",
        "current_cleaning_result": {
            "clean_text": "",
            "normalization_notes": [],
        },
        "current_section_result": {
            "sections": {},
            "detected_section_names": [],
            "section_count": 0,
        },
        "current_entity_result": {
            "model_info": {
                "rule_based": True,
                "scispacy_used": False,
                "bc5cdr_used": False,
                "bioclinicalbert_used": False,
                "scispacy_available": False,
                "bc5cdr_available": False,
                "bioclinicalbert_available": False,
                "warnings": [],
            },
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
        },
        "claim_diagnosis_input": "",
        "claim_procedure_input": "",
        "claim_code_input": "",
        "current_claim_support_result": None,
        "current_chunks": [],
        "current_retrieval_index": {
            "method": "tfidf",
            "chunks": [],
            "model_name": "TfidfVectorizer",
            "index_object": None,
            "warnings": [],
        },
        "question_input": "",
        "current_qa_result": None,
        "qa_history": [],
        "current_documentation_result": {
            "completeness_score": 0.0,
            "score_label": "Low",
            "checks": [],
            "missing_fields": [],
            "reviewer_recommendations": [],
        },
        "current_summary_result": {
            "short_summary": "No document summary available yet.",
            "clinical_evidence_summary": "Insufficient evidence found.",
            "claim_support_summary": "Claim support has not been checked yet.",
            "documentation_summary": "Documentation completeness has not been evaluated yet.",
            "risk_flags": [],
            "human_review_note": "Human review required.",
        },
        "current_evaluation_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def get_sample_documents() -> list[str]:
    """Return available bundled synthetic documents."""
    if not SYNTHETIC_DOCS_DIR.exists():
        return []
    return sorted(path.name for path in SYNTHETIC_DOCS_DIR.glob("*.txt"))


@st.cache_data
def load_sample_claims() -> pd.DataFrame:
    """Load sample claims if available."""
    if not SAMPLE_CLAIMS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_CLAIMS_PATH)


def normalize_claim_value(value: object) -> str:
    """Convert CSV/session claim values into safe strings for text inputs."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


@st.cache_data
def load_expected_labels() -> dict:
    """Load expected evaluation labels if available."""
    if not EXPECTED_LABELS_PATH.exists():
        return {}
    return json.loads(EXPECTED_LABELS_PATH.read_text(encoding="utf-8"))


def set_document_state(result: dict, status: str, source: str) -> None:
    """Update the active document state."""
    raw_text = result.get("raw_text", "")
    cleaning_result = clean_text(raw_text)
    cleaned_text = cleaning_result.get("clean_text", "")
    section_result = detect_sections(cleaned_text)

    st.session_state["current_document_name"] = result.get("file_name", "Unknown document")
    st.session_state["current_document_text"] = cleaned_text
    st.session_state["processing_status"] = status
    st.session_state["document_source"] = source
    st.session_state["current_document_result"] = result
    st.session_state["current_document_raw_text"] = raw_text
    st.session_state["current_document_clean_text"] = cleaned_text
    st.session_state["current_cleaning_result"] = cleaning_result
    st.session_state["current_section_result"] = section_result
    st.session_state["current_claim_support_result"] = None
    st.session_state["current_qa_result"] = None
    st.session_state["qa_history"] = []
    st.session_state["current_documentation_result"] = {
        "completeness_score": 0.0,
        "score_label": "Low",
        "checks": [],
        "missing_fields": [],
        "reviewer_recommendations": [],
    }
    st.session_state["current_summary_result"] = {
        "short_summary": "No document summary available yet.",
        "clinical_evidence_summary": "Insufficient evidence found.",
        "claim_support_summary": "Claim support has not been checked yet.",
        "documentation_summary": "Documentation completeness has not been evaluated yet.",
        "risk_flags": [],
        "human_review_note": "Human review required.",
    }


def load_sample_document(sample_name: str, source_label: str = "Sample synthetic document") -> None:
    """Load a bundled synthetic text document."""
    sample_path = SYNTHETIC_DOCS_DIR / sample_name
    if not sample_path.exists():
        set_document_state(
            {
                "success": False,
                "file_name": sample_name,
                "file_type": Path(sample_name).suffix.lstrip("."),
                "extraction_method": "none",
                "raw_text": "",
                "warnings": [],
                "errors": ["Selected sample document could not be found."],
                "text_quality_score": 0.0,
            },
            "Selected sample document could not be found.",
            source_label,
        )
        return
    result = load_document(sample_path)
    status_prefix = "Loaded bundled synthetic document" if result["success"] else "Document load issue"
    set_document_state(result, f"{status_prefix}: {sample_name}", source_label)


def process_uploaded_file(uploaded_file) -> None:
    """Handle uploaded files through the shared document loader."""
    result = load_document(uploaded_file)
    status_prefix = "Loaded uploaded document" if result["success"] else "Document load issue"
    set_document_state(result, f"{status_prefix}: {uploaded_file.name}", "Uploaded document")


def load_demo_scenario(
    document_name: str,
    diagnosis: str = "",
    procedure: str = "",
    code: str = "",
) -> None:
    """Load a guided demo scenario with suggested claim fields."""
    load_sample_document(document_name, source_label="Demo scenario")
    st.session_state["claim_diagnosis_input"] = normalize_claim_value(diagnosis)
    st.session_state["claim_procedure_input"] = normalize_claim_value(procedure)
    st.session_state["claim_code_input"] = normalize_claim_value(code)
    st.session_state["question_input"] = ""
    st.session_state["current_claim_support_result"] = None
    st.session_state["current_qa_result"] = None
    st.session_state["qa_history"] = []


def maybe_auto_load_sample(selected_sample: str) -> None:
    """Auto-load sample text when the dropdown changes."""
    previous_sample = st.session_state.get("last_selected_sample")
    if selected_sample and selected_sample != previous_sample:
        load_sample_document(selected_sample)
    st.session_state["last_selected_sample"] = selected_sample


def preview_entities(text: str) -> pd.DataFrame:
    """Create a lightweight entity preview for the shell."""
    if not text.strip():
        return pd.DataFrame(columns=["Entity", "Type", "Assertion", "Evidence"])

    patterns = [
        ("Hypertension", "Diagnosis", r"\bhypertension\b"),
        ("Pneumonia", "Diagnosis", r"\bpneumonia\b"),
        ("Angina", "Diagnosis", r"\bangina\b"),
        ("Chest pain", "Symptom", r"\bchest pain\b"),
        ("ECG", "Procedure", r"\b(ecg|ekg)\b"),
        ("Chest X-ray", "Procedure", r"\bchest x-ray\b"),
        ("I10", "Code", r"\bI10\b"),
        ("J18.9", "Code", r"\bJ18\.9\b"),
        ("93000", "Code", r"\b93000\b"),
    ]
    entities: list[dict[str, str]] = []
    lowered_text = text.lower()
    for label, entity_type, pattern in patterns:
        match = re.search(pattern, lowered_text, flags=re.IGNORECASE)
        if not match:
            continue
        window_start = max(0, match.start() - 60)
        window_end = min(len(text), match.end() + 60)
        evidence = " ".join(text[window_start:window_end].split())
        assertion = "Present"
        if re.search(rf"(no evidence of|ruled out|denies).{{0,30}}{pattern}", lowered_text, re.IGNORECASE):
            assertion = "Negated"
        elif re.search(rf"(possible|concern for|pending).{{0,30}}{pattern}", lowered_text, re.IGNORECASE):
            assertion = "Uncertain"
        entities.append(
            {
                "Entity": label,
                "Type": entity_type,
                "Assertion": assertion,
                "Evidence": evidence,
            }
        )
    return pd.DataFrame(entities)


def build_model_messages(model_info: dict, use_advanced_models: bool) -> tuple[str | None, str | None]:
    """Create a compact model-status summary without repetitive warnings."""
    warning_text = None
    info_text = None

    if not model_info.get("scispacy_used"):
        warning_text = (
            "Advanced clinical NLP models are not installed locally. "
            "Using stable rule-based extraction for this demo."
        )

    if use_advanced_models and not model_info.get("bioclinicalbert_used"):
        info_text = (
            "Optional Bio_ClinicalBERT enhancement is unavailable locally. "
            "Continuing with the standard extraction pipeline."
        )

    return warning_text, info_text


def build_missing_documentation_table(document_name: str, expected_labels: dict) -> pd.DataFrame:
    """Return missing field expectations for the selected document."""
    details = expected_labels.get(document_name, {})
    missing_fields = details.get("missing_fields", [])
    if not missing_fields:
        return pd.DataFrame(
            [{"Field": "No expected missing fields", "Status": "No issue flagged in label set"}]
        )
    return pd.DataFrame(
        [{"Field": field, "Status": "Expected missing"} for field in missing_fields]
    )


def build_report_preview(document_name: str, status: str, text: str) -> str:
    """Return a reviewer-facing placeholder report preview."""
    preview = text[:600].strip() if text.strip() else "No document text is currently loaded."
    return (
        f"Reviewer Report Preview\n"
        f"Document: {document_name}\n"
        f"Status: {status}\n\n"
        f"Summary:\n"
        f"This initial app shell has loaded the document and prepared the workspace "
        f"for extraction, evidence review, and claim support analysis.\n\n"
        f"Preview Text:\n{preview}"
    )


def refresh_entity_state(use_advanced_models: bool) -> None:
    """Refresh clinical entity extraction from the cleaned document text."""
    clean_document_text = st.session_state.get("current_document_clean_text", "")
    section_result = st.session_state.get("current_section_result", {"sections": {}})
    if not clean_document_text.strip():
        st.session_state["current_entity_result"] = initialize_empty_entity_result()
        return
    st.session_state["current_entity_result"] = extract_entities(
        clean_text=clean_document_text,
        sections=section_result,
        use_advanced_models=use_advanced_models,
    )


def refresh_retrieval_state() -> None:
    """Rebuild retrieval chunks and index for the active document."""
    clean_document_text = st.session_state.get("current_document_clean_text", "")
    section_result = st.session_state.get("current_section_result", {"sections": {}})
    if not clean_document_text.strip():
        st.session_state["current_chunks"] = []
        st.session_state["current_retrieval_index"] = {
            "method": "tfidf",
            "chunks": [],
            "model_name": "TfidfVectorizer",
            "index_object": None,
            "warnings": ["No document text is currently available for retrieval."],
        }
        return

    chunks = chunk_document(clean_document_text, section_result)
    retrieval_index = build_retrieval_index(chunks, method="auto")
    st.session_state["current_chunks"] = chunks
    st.session_state["current_retrieval_index"] = retrieval_index


def refresh_documentation_state() -> None:
    """Rebuild documentation completeness checks for the active document."""
    clean_document_text = st.session_state.get("current_document_clean_text", "")
    section_result = st.session_state.get("current_section_result", {"sections": {}})
    entity_result = st.session_state.get("current_entity_result", {})
    if not clean_document_text.strip():
        st.session_state["current_documentation_result"] = {
            "completeness_score": 0.0,
            "score_label": "Low",
            "checks": [],
            "missing_fields": [],
            "reviewer_recommendations": [],
        }
        return
    st.session_state["current_documentation_result"] = check_documentation_completeness(
        entities=entity_result,
        sections=section_result,
        clean_text=clean_document_text,
    )


def refresh_summary_state() -> None:
    """Generate the grounded reviewer summary for the active document."""
    entities = st.session_state.get("current_entity_result", {})
    section_result = st.session_state.get("current_section_result", {"sections": {}})
    claim_support_result = st.session_state.get("current_claim_support_result")
    documentation_result = st.session_state.get("current_documentation_result")

    entities_with_metadata = dict(entities)
    entities_with_metadata["document_metadata"] = {
        "document_type": _extract_document_type(section_result),
    }
    st.session_state["current_summary_result"] = generate_reviewer_summary(
        entities=entities_with_metadata,
        claim_support_result=claim_support_result,
        documentation_result=documentation_result,
    )


def initialize_empty_entity_result() -> dict:
    """Return an empty entity extraction payload."""
    return {
        "model_info": {
            "rule_based": True,
            "scispacy_used": False,
            "bc5cdr_used": False,
            "bioclinicalbert_used": False,
            "scispacy_available": False,
            "bc5cdr_available": False,
            "bioclinicalbert_available": False,
            "warnings": [],
        },
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


def format_entity_table(entities: list[dict]) -> pd.DataFrame:
    """Convert entity dicts into a display-friendly dataframe."""
    if not entities:
        return pd.DataFrame(
            columns=[
                "Entity",
                "Type",
                "Section",
                "Evidence",
                "Confidence",
                "Source/model",
                "Status modifier",
                "Trigger",
                "Support Meaning",
            ]
        )
    return pd.DataFrame(
        [
            {
                "Entity": entity.get("normalized_text", entity.get("text", "")),
                "Type": entity.get("entity_type", ""),
                "Section": entity.get("section", "unknown"),
                "Evidence": entity.get("evidence", ""),
                "Confidence": entity.get("confidence", ""),
                "Source/model": ", ".join(entity.get("sources", [entity.get("source", "")])),
                "Status modifier": entity.get("status_modifier", "affirmed"),
                "Trigger": entity.get("trigger") or "",
                "Support Meaning": _support_meaning(entity),
            }
            for entity in entities
        ]
    )


def _support_meaning(entity: dict) -> str:
    """Translate entity status into a reviewer-facing support interpretation."""
    if entity.get("is_negated"):
        return "Not supporting evidence"
    if entity.get("is_uncertain"):
        return "Unclear evidence"
    return "Supporting evidence"


def _empty_claim_support_result() -> dict:
    """Return a default empty claim support payload."""
    return {
        "overall_status": "Unclear",
        "items": [],
        "reviewer_summary": "No claim has been checked yet.",
        "warnings": [],
    }


def _empty_qa_result() -> dict:
    """Return a default empty Q&A payload."""
    return {
        "question": "",
        "answer": "No question has been asked yet.",
        "answer_status": "Insufficient Evidence",
        "confidence": "Low",
        "retrieval_method": "unavailable",
        "evidence": [],
        "grounding_note": "Answer generated only from retrieved document evidence.",
        "human_review_required": True,
    }


def _status_message_renderer(status: str):
    """Return the Streamlit status function that best matches a result state."""
    if status == "Supported":
        return st.success
    if status == "Partially Supported":
        return st.warning
    if status == "Unclear":
        return st.info
    return st.error


def _extract_document_type(section_result: dict) -> str | None:
    """Pull document type from patient info or unknown section when available."""
    sections = section_result.get("sections", {}) if isinstance(section_result, dict) else {}
    patient_info = sections.get("patient_info", "")
    unknown = sections.get("unknown", "")
    for block in [patient_info, unknown]:
        for line in block.splitlines():
            if line.lower().startswith("document type:"):
                return line.split(":", 1)[1].strip()
    return None


def _status_label(is_ready: bool) -> str:
    """Return a short card-friendly status label."""
    return "Ready" if is_ready else "Pending"


def _workflow_mark(is_done: bool) -> str:
    """Render a markdown-friendly workflow checkbox."""
    return "x" if is_done else " "


def render_summary_card(title: str, body: str) -> None:
    """Render a compact product-style summary card."""
    st.markdown(
        f"""
        <div class="summary-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


apply_app_styles()
initialize_session_state()
sample_claims_df = load_sample_claims()
expected_labels = load_expected_labels()
sample_documents = get_sample_documents()

st.markdown(
    f"""
    <div class="app-hero">
        <h1>{APP_TITLE}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

sample_options = ["Select a sample synthetic document"]
sample_options.extend(sample_documents)

control_col1, control_col2 = st.columns([1.25, 1])
with control_col1:
    st.subheader("Load a Document")
    selected_sample = st.selectbox(
        "Sample synthetic document",
        options=sample_options,
        index=0,
        disabled=not sample_documents,
        help="Choose a bundled sample note, then click Process Document.",
    )
    uploaded_file = st.file_uploader(
        "Upload TXT, PDF, or image",
        type=["txt", "pdf", "png", "jpg", "jpeg"],
        help="TXT is the smoothest path for the demo. OCR-based image extraction degrades gracefully if local OCR tools are unavailable.",
    )
    use_advanced_models = st.checkbox(
        "Use optional advanced semantic enhancement",
        value=False,
        help=(
            "If local advanced model dependencies are available, the app can use them. "
            "The core workflow still works without them."
        ),
    )
    process_clicked = st.button("Process Document", type="primary")
    st.caption("This runs the full review pipeline on the currently selected or uploaded document.")

with control_col2:
    st.subheader("Quick Demo")
    demo_button_col1, demo_button_col2 = st.columns(2)
    if demo_button_col1.button("Best Demo Scenario", use_container_width=True):
        load_demo_scenario("clean_supported_claim.txt", "Hypertension", "ECG", "I10")
    if demo_button_col2.button("Unsupported Scenario", use_container_width=True):
        load_demo_scenario("unsupported_procedure.txt", "Hypertension", "ECG", "93000")
    if st.button("Negation Scenario", use_container_width=True):
        load_demo_scenario("negated_diagnosis.txt", "Pneumonia", "", "")
    st.caption(
        "Fastest walkthrough: load the best demo, review extracted entities, run claim check, then show evidence Q&A."
    )

if process_clicked:
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)
    elif sample_documents and selected_sample != sample_options[0]:
        load_sample_document(selected_sample)
    else:
        set_document_state(
            {
                "success": False,
                "file_name": "No document loaded",
                "file_type": "",
                "extraction_method": "none",
                "raw_text": "",
                "warnings": [],
                "errors": [],
                "text_quality_score": 0.0,
            },
            "Select a sample synthetic note or upload a TXT, PDF, or image file.",
            "None",
        )

refresh_entity_state(use_advanced_models=use_advanced_models)
refresh_retrieval_state()
refresh_documentation_state()
refresh_summary_state()

current_document_name = st.session_state["current_document_name"]
current_document_text = st.session_state["current_document_text"]
processing_status = st.session_state["processing_status"]
document_source = st.session_state["document_source"]
current_document_result = st.session_state["current_document_result"]
current_document_raw_text = st.session_state["current_document_raw_text"]
current_document_clean_text = st.session_state["current_document_clean_text"]
current_cleaning_result = st.session_state["current_cleaning_result"]
current_section_result = st.session_state["current_section_result"]
current_entity_result = st.session_state["current_entity_result"]
current_chunks = st.session_state["current_chunks"]
current_retrieval_index = st.session_state["current_retrieval_index"]
current_documentation_result = st.session_state["current_documentation_result"]
current_claim_support_result = st.session_state.get("current_claim_support_result") or _empty_claim_support_result()
current_qa_result = st.session_state.get("current_qa_result") or _empty_qa_result()
qa_history = st.session_state.get("qa_history", [])
current_summary_result = st.session_state["current_summary_result"]
current_evaluation_result = st.session_state.get("current_evaluation_result")
current_claim_rows = (
    sample_claims_df[sample_claims_df["document_file"] == current_document_name]
    if not sample_claims_df.empty and "document_file" in sample_claims_df
    else pd.DataFrame()
)

document_processed = bool(current_document_clean_text.strip())
entities_extracted = len(current_entity_result.get("all_entities", [])) > 0
claim_checked = bool(current_claim_support_result.get("items"))
report_ready = document_processed and entities_extracted
text_quality_display = f"{current_document_result.get('text_quality_score', 0.0):.1f}"
sections_detected_display = current_section_result.get("section_count", 0)
entities_found_display = len(current_entity_result.get("all_entities", []))
completeness_display = f"{current_documentation_result.get('completeness_score', 0.0):.2f}"

tabs = st.tabs(
    [
        "Overview",
        "Load Document",
        "Review Text",
        "Entities",
        "Claim Check",
        "Evidence Q&A",
        "Documentation",
        "More",
    ]
)

with tabs[0]:
    if current_document_text:
        st.success(
            f"Active document: {current_document_name} | Retrieval: {current_retrieval_index.get('model_name', 'Unavailable')}"
        )
    else:
        st.info("No document text is currently loaded. Start with a demo scenario or choose a sample note above.")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Document", "Loaded" if document_processed else "Not loaded")
    metric_col2.metric("Sections", sections_detected_display)
    metric_col3.metric("Entities", entities_found_display)
    metric_col4.metric("Completeness", completeness_display)

    summary_row_left, summary_row_right = st.columns([1.25, 1])
    with summary_row_left:
        if document_processed:
            render_summary_card(
                "What This App Is Doing",
                "It reads a synthetic clinical note, extracts evidence, checks whether a diagnosis or procedure is supported, flags missing documentation, and keeps the human reviewer in control.",
            )
        else:
            render_summary_card(
                "Start Here",
                "Load a demo scenario or choose a sample note above. Then move left to right through Review Text, Entities, Claim Check, and Evidence Q&A.",
            )
    with summary_row_right:
        if document_processed:
            render_summary_card(
                "Recommended Next Step",
                "Open 'Entities' if you want to inspect extracted evidence first, or go to 'Claim Check' if you want the quickest business-value demo.",
            )
        else:
            render_summary_card(
                "Best Demo Flow",
                "Use 'Best Demo Scenario' for a clean supported example. Then use the unsupported or negation scenarios to show safety and reviewer control.",
            )

    st.subheader("Why This Exists")
    st.write(
        "Healthcare payment and coding reviewers spend significant time checking whether a chart really supports a claimed diagnosis, procedure, or code. This prototype turns that manual search problem into a guided evidence-review workflow."
    )

    overview_col1, overview_col2 = st.columns([1.05, 1])
    with overview_col1:
        st.subheader("Who It Helps")
        st.markdown(
            "- Clinical reviewers\n"
            "- Coding validation analysts\n"
            "- Payment integrity analysts\n"
            "- Healthcare AI research and product teams"
        )
        st.subheader("Why This Matters for Cotiviti-Style Workflows")
        st.markdown(
            "- Supports payment accuracy review by surfacing clinical evidence.\n"
            "- Helps coding and clinical reviewers quickly find documentation support.\n"
            "- Flags missing or insufficient documentation.\n"
            "- Uses evidence-grounded responses instead of unsupported AI summaries.\n"
            "- Keeps final decisions with human experts."
        )
        st.subheader("What This POC Demonstrates Technically")
        st.markdown(
            "- NLP entity extraction\n"
            "- Negation and uncertainty handling\n"
            "- OCR-aware ingestion\n"
            "- Evidence retrieval\n"
            "- RAG-style Q&A\n"
            "- Claim support classification\n"
            "- Documentation completeness scoring\n"
            "- Responsible AI guardrails\n"
            "- Synthetic evaluation"
        )
        if current_document_clean_text:
            st.subheader("Reviewer Summary")
            st.info(current_summary_result["short_summary"])
            st.write(current_summary_result["clinical_evidence_summary"])
            st.write(current_summary_result["claim_support_summary"])
            st.write(current_summary_result["documentation_summary"])
            if current_summary_result.get("risk_flags"):
                st.write("Risk flags:")
                for flag in current_summary_result["risk_flags"]:
                    st.warning(flag)
            st.caption(current_summary_result["human_review_note"])
    with overview_col2:
        st.subheader("Simple Workflow")
        st.info(
            "Load a note -> Review text -> Review entities -> Check claim support -> Ask grounded question -> Check documentation -> Export reviewer report"
        )
        st.subheader("Best Way to Present It")
        st.markdown(
            "1. Load Best Demo Scenario  \n"
            "2. Show extracted text and sections  \n"
            "3. Show entities  \n"
            "4. Run claim support  \n"
            "5. Ask a grounded question  \n"
            "6. Show missing documentation, evaluation, governance, and export"
        )
        metrics_left, metrics_mid, metrics_right = st.columns(3)
        metrics_left.metric("Synthetic Notes", len(sample_documents))
        metrics_mid.metric("Text Quality Score", f"{current_document_result.get('text_quality_score', 0.0):.1f}")
        metrics_right.metric("Section Count", current_section_result.get("section_count", 0))

    with st.expander("How to Read the App", expanded=not document_processed):
        st.write(
            "If you are seeing this for the first time, the easiest path is left to right through the tabs. Start with the document, confirm what text was read, inspect extracted entities, then use the business-facing claim support and documentation views."
        )

with tabs[1]:
    st.subheader("Load a Document")
    st.write(
        "Use the controls above to select a bundled synthetic note or upload a TXT, PDF, or image file. If you want the quickest demo, use one of the quick demo buttons."
    )
    upload_summary = pd.DataFrame(
        [
            {"Field": "Current document", "Value": current_document_name},
            {"Field": "File type", "Value": current_document_result.get("file_type", "") or "Unknown"},
            {"Field": "Source", "Value": document_source},
            {"Field": "Processing status", "Value": processing_status},
        ]
    )
    st.dataframe(upload_summary, use_container_width=True, hide_index=True)

    if sample_documents:
        st.subheader("Available Sample Documents")
        st.dataframe(
            pd.DataFrame({"Sample synthetic documents": sample_documents}),
            use_container_width=True,
            hide_index=True,
        )

with tabs[2]:
    st.subheader("Review Extracted Text")
    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
    metadata_col1.metric(
        "Extraction Method",
        current_document_result.get("extraction_method", "none") or "none",
    )
    metadata_col2.metric(
        "Text Quality Score",
        f"{current_document_result.get('text_quality_score', 0.0):.1f}",
    )
    metadata_col3.metric(
        "Warnings / Errors",
        f"{len(current_document_result.get('warnings', []))} / {len(current_document_result.get('errors', []))}",
    )
    metrics_row_left, metrics_row_mid, metrics_row_right = st.columns(3)
    metrics_row_left.metric(
        "Document Length",
        f"{len(current_document_clean_text):,}",
    )
    metrics_row_mid.metric(
        "Section Count",
        current_section_result.get("section_count", 0),
    )
    metrics_row_right.metric(
        "Normalization Notes",
        len(current_cleaning_result.get("normalization_notes", [])),
    )

    metadata_df = pd.DataFrame(
        [
            {"Field": "File name", "Value": current_document_result.get("file_name", current_document_name)},
            {"Field": "File type", "Value": current_document_result.get("file_type", "") or "Unknown"},
            {"Field": "Success", "Value": str(current_document_result.get("success", False))},
            {
                "Field": "Extraction method",
                "Value": current_document_result.get("extraction_method", "none") or "none",
            },
            {
                "Field": "Text quality score",
                "Value": f"{current_document_result.get('text_quality_score', 0.0):.1f}",
            },
        ]
    )
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)

    for normalization_note in current_cleaning_result.get("normalization_notes", []):
        st.info(normalization_note)
    for warning_message in current_document_result.get("warnings", []):
        st.warning(warning_message)
    for error_message in current_document_result.get("errors", []):
        st.error(error_message)

    if current_document_raw_text or current_document_clean_text:
        st.success("Document text loaded successfully.")
        with st.expander("Raw Extracted Text", expanded=False):
            st.text_area(
                "Raw extracted document text",
                value=current_document_raw_text,
                height=260,
                disabled=True,
                key="raw_text_display",
            )
        with st.expander("Cleaned Text", expanded=True):
            st.text_area(
                "Cleaned document text",
                value=current_document_clean_text,
                height=260,
                disabled=True,
                key="clean_text_display",
            )

        section_rows = [
            {
                "Section": section_name,
                "Preview": section_text[:180].replace("\n", " "),
            }
            for section_name, section_text in current_section_result.get("sections", {}).items()
        ]
        if section_rows:
            st.subheader("Section Detection")
            st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No structured section headings were detected in this document.")
    else:
        st.info(
            "Load a sample note or upload a TXT, PDF, or image file to view extracted text here."
        )

with tabs[3]:
    st.subheader("Review Extracted Entities")
    st.caption("This is the best place to verify what the app thinks is actually documented before you check support.")
    model_info = current_entity_result.get("model_info", {})
    status_rows = pd.DataFrame(
        [
            {"Model Layer": "Rule-based extraction", "Status": "Active"},
            {
                "Model Layer": "scispaCy",
                "Status": "Active" if model_info.get("scispacy_used") else "Unavailable",
            },
            {
                "Model Layer": "BC5CDR model",
                "Status": "Active" if model_info.get("bc5cdr_used") else "Unavailable",
            },
            {
                "Model Layer": "Bio_ClinicalBERT advanced mode",
                "Status": (
                    "Active"
                    if model_info.get("bioclinicalbert_used")
                    else "Unavailable" if use_advanced_models else "Disabled"
                ),
            },
        ]
    )
    st.dataframe(status_rows, use_container_width=True, hide_index=True)

    model_warning_text, model_info_text = build_model_messages(model_info, use_advanced_models)
    if model_warning_text:
        st.warning(model_warning_text)
    if model_info_text:
        st.info(model_info_text)

    entity_metric_left, entity_metric_mid, entity_metric_right = st.columns(3)
    entity_metric_left.metric("Total Entities", len(current_entity_result.get("all_entities", [])))
    entity_metric_mid.metric("Diagnoses", len(current_entity_result.get("diagnoses", [])))
    entity_metric_right.metric("Procedures", len(current_entity_result.get("procedures", [])))

    admin_entities = (
        current_entity_result.get("patient", [])
        + current_entity_result.get("provider", [])
        + current_entity_result.get("dates", [])
    )
    entity_groups = [
        ("Admin fields", admin_entities),
        ("Diagnoses", current_entity_result.get("diagnoses", [])),
        ("Symptoms", current_entity_result.get("symptoms", [])),
        ("Procedures", current_entity_result.get("procedures", [])),
        ("Medications", current_entity_result.get("medications", [])),
        ("Codes", current_entity_result.get("codes", [])),
        ("Signatures", current_entity_result.get("signatures", [])),
    ]

    if not current_entity_result.get("all_entities"):
        st.info("Load a document to run layered clinical entity extraction.")
    else:
        for title, entities in entity_groups:
            st.subheader(title)
            entity_df = format_entity_table(entities)
            if entity_df.empty:
                st.info(f"No {title.lower()} detected in the current document.")
            else:
                st.dataframe(entity_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Check Claim Support")
    st.write(
        "Enter a claimed diagnosis, procedure, or code to see whether the current document "
        "appears to support it. This is a reviewer aid, not a final coding or payment decision."
    )
    st.caption("Enter the diagnosis, procedure, or code you want to validate against the current document.")

    if current_claim_rows.empty:
        st.info("No sample claim row is mapped to the current document. Manual entry is still available.")
    else:
        st.success("Matched sample claim configuration found for the current document.")
        sample_claim_options = [
            (
                f"Diagnosis: {normalize_claim_value(row['claimed_diagnosis']) or '-'} | "
                f"Procedure: {normalize_claim_value(row['claimed_procedure']) or '-'} | "
                f"Code: {normalize_claim_value(row['claimed_code']) or '-'}"
            )
            for _, row in current_claim_rows.iterrows()
        ]
        selected_claim_option = st.selectbox(
            "Load a demo claim from sample_claims.csv",
            options=sample_claim_options,
            key="sample_claim_selector",
        )
        if st.button("Load Sample Claim", key="load_sample_claim_button"):
            selected_row = current_claim_rows.iloc[sample_claim_options.index(selected_claim_option)]
            st.session_state["claim_diagnosis_input"] = normalize_claim_value(
                selected_row.get("claimed_diagnosis", "")
            )
            st.session_state["claim_procedure_input"] = normalize_claim_value(
                selected_row.get("claimed_procedure", "")
            )
            st.session_state["claim_code_input"] = normalize_claim_value(
                selected_row.get("claimed_code", "")
            )
        st.dataframe(current_claim_rows, use_container_width=True, hide_index=True)

    st.session_state["claim_diagnosis_input"] = normalize_claim_value(
        st.session_state.get("claim_diagnosis_input", "")
    )
    st.session_state["claim_procedure_input"] = normalize_claim_value(
        st.session_state.get("claim_procedure_input", "")
    )
    st.session_state["claim_code_input"] = normalize_claim_value(
        st.session_state.get("claim_code_input", "")
    )

    claim_col1, claim_col2, claim_col3 = st.columns(3)
    claim_col1.text_input("Claimed diagnosis", key="claim_diagnosis_input")
    claim_col2.text_input("Claimed procedure", key="claim_procedure_input")
    claim_col3.text_input("Claimed code", key="claim_code_input")

    if st.button("Check Support", key="check_support_button"):
        claim_payload = {
            "claimed_diagnosis": st.session_state.get("claim_diagnosis_input", ""),
            "claimed_procedure": st.session_state.get("claim_procedure_input", ""),
            "claimed_code": st.session_state.get("claim_code_input", ""),
        }
        st.session_state["current_claim_support_result"] = check_claim_support(
            claim=claim_payload,
            entities=current_entity_result,
            clean_text=current_document_clean_text,
            sections=current_section_result,
        )
        current_claim_support_result = st.session_state["current_claim_support_result"]
        refresh_summary_state()
        current_summary_result = st.session_state["current_summary_result"]

    overall_renderer = _status_message_renderer(current_claim_support_result["overall_status"])
    overall_renderer(f"Overall status: {current_claim_support_result['overall_status']}")
    summary_col1, summary_col2 = st.columns(2)
    summary_col1.metric("Support Engine", "Rule + Clinical Entity Review")
    summary_col2.metric("Reviewer Control", "Enabled")
    st.caption(current_claim_support_result["reviewer_summary"])

    for warning_message in current_claim_support_result.get("warnings", []):
        st.warning(warning_message)

    if not current_claim_support_result.get("items"):
        st.info("Run a claim support check to see item-level evidence and reasoning.")
    else:
        for item in current_claim_support_result["items"]:
            item_renderer = _status_message_renderer(item["status"])
            item_renderer(f"{item['claim_type'].title()}: {item['claim_value']} -> {item['status']}")
            item_df = pd.DataFrame(
                [
                    {"Field": "Confidence", "Value": item["confidence"]},
                    {"Field": "Evidence", "Value": item["evidence"] or "No direct evidence found."},
                    {"Field": "Reason", "Value": item["reason"]},
                    {
                        "Field": "Human review required",
                        "Value": "Yes" if item["human_review_required"] else "No",
                    },
                ]
            )
            st.dataframe(item_df, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Evidence Q&A")
    st.write(
        "Ask grounded questions about the current document. Answers are generated only from "
        "retrieved document chunks and extracted evidence."
    )
    st.caption("Evidence snippets and similarity scores are shown to keep answers inspectable and easy to present.")
    suggested_questions = [
        "Does this document support hypertension?",
        "Does this document support ECG?",
        "What procedures are documented?",
        "Is pneumonia ruled out?",
        "Is provider signature present?",
        "What documentation is missing?",
    ]
    button_columns = st.columns(3)
    selected_question = None
    for index, suggested_question in enumerate(suggested_questions):
        if button_columns[index % 3].button(
            suggested_question,
            key=f"suggested_question_{index}",
        ):
            selected_question = suggested_question
            st.session_state["question_input"] = suggested_question

    st.text_input("Ask a grounded question", key="question_input")
    run_question = st.button("Retrieve Evidence and Answer", key="answer_question_button")

    if selected_question or run_question:
        question_to_run = selected_question or st.session_state.get("question_input", "")
        retrieved_chunks = retrieve_evidence(question_to_run, current_retrieval_index, top_k=3)
        st.session_state["current_qa_result"] = answer_question(
            question=question_to_run,
            retrieved_chunks=retrieved_chunks,
            entities=current_entity_result,
            documentation_result=current_documentation_result,
        )
        current_qa_result = st.session_state["current_qa_result"]
        qa_history = st.session_state.get("qa_history", [])
        if (
            current_qa_result.get("question")
            and (
                not qa_history
                or qa_history[-1].get("question") != current_qa_result.get("question")
                or qa_history[-1].get("answer") != current_qa_result.get("answer")
            )
        ):
            qa_history.append(current_qa_result.copy())
            st.session_state["qa_history"] = qa_history

    answer_renderer = _status_message_renderer(
        "Supported"
        if current_qa_result["answer_status"] == "Answered"
        else "Unclear"
    )
    answer_renderer(current_qa_result["answer"])
    qa_metric_col1, qa_metric_col2 = st.columns(2)
    qa_metric_col1.metric("Confidence", current_qa_result["confidence"])
    qa_metric_col2.metric("Retrieval Method", current_qa_result["retrieval_method"])
    st.caption(current_qa_result["grounding_note"])
    if current_qa_result["human_review_required"]:
        st.warning("Human review required before any clinical, coding, or payment decision.")

    evidence_rows = [
        {
            "Rank": chunk["rank"],
            "Section": chunk["section"],
            "Score": f"{chunk['score']:.3f}",
            "Weak Retrieval": "Yes" if chunk.get("weak_retrieval") else "No",
            "Snippet": chunk["text"][:260].replace("\n", " "),
        }
        for chunk in current_qa_result.get("evidence", [])
    ]
    if evidence_rows:
        st.subheader("Evidence Snippets")
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Evidence snippets will appear here after you ask a grounded question.")

with tabs[6]:
    st.subheader("Check Documentation Completeness")
    st.caption("Use this checklist to quickly spot missing elements that could block coding or payment review.")
    if not current_document_name or current_document_name == "No document loaded":
        st.info("Load a document to review documentation completeness.")
    else:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(
            "Completeness Score",
            f"{current_documentation_result.get('completeness_score', 0.0):.2f}",
        )
        metric_col2.metric(
            "Score Label",
            current_documentation_result.get("score_label", "Low"),
        )

        checks_df = pd.DataFrame(current_documentation_result.get("checks", []))
        if not checks_df.empty:
            st.subheader("Checklist")
            st.dataframe(checks_df, use_container_width=True, hide_index=True)

        missing_fields = current_documentation_result.get("missing_fields", [])
        if missing_fields:
            st.subheader("Missing Fields")
            st.dataframe(
                pd.DataFrame({"Missing field": missing_fields}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No obvious missing core documentation fields were detected.")

        recommendations = current_documentation_result.get("reviewer_recommendations", [])
        if recommendations:
            st.subheader("Reviewer Recommendations")
            for recommendation in recommendations:
                st.info(recommendation)

with tabs[7]:
    st.subheader("More")
    st.write("Advanced validation, governance, and export tools are grouped here to keep the main review flow simpler.")

    with st.expander("Evaluation", expanded=False):
        st.write(
            "This evaluation is designed to demonstrate research discipline and model-aware "
            "validation, not production accuracy."
        )
        if st.button("Run Evaluation on Synthetic Test Set", key="run_evaluation_button"):
            st.session_state["current_evaluation_result"] = run_evaluation(
                data_dir=str(BASE_DIR / "data"),
                use_scispacy=True,
                use_sentence_transformers=True,
            )
            current_evaluation_result = st.session_state["current_evaluation_result"]

        if not current_evaluation_result:
            st.info("Run the synthetic test set to compare extraction, claim support behavior, and retrieval quality.")
        else:
            model_rows = pd.DataFrame(
                [
                    {"Model": model_name, "Availability": availability}
                    for model_name, availability in current_evaluation_result.get("models", {}).items()
                ]
            )
            st.dataframe(model_rows, use_container_width=True, hide_index=True)

            dataset_metrics = current_evaluation_result.get("dataset_metrics", {})
            claim_metrics = current_evaluation_result.get("claim_support_metrics", {})
            retrieval_metrics = current_evaluation_result.get("retrieval_metrics", {})
            extraction_metrics = current_evaluation_result.get("entity_extraction_metrics", {})

            active_extraction_metrics = extraction_metrics.get("scispacy_enhanced")
            if (
                not isinstance(active_extraction_metrics, dict)
                or active_extraction_metrics.get("diagnosis_exact_match_rate") == "Not available"
            ):
                active_extraction_metrics = extraction_metrics.get("rule_based", {})

            eval_metric_1, eval_metric_2, eval_metric_3, eval_metric_4 = st.columns(4)
            eval_metric_1.metric("Documents Tested", dataset_metrics.get("documents_tested", 0))
            eval_metric_2.metric(
                "Diagnosis Match Rate",
                str(active_extraction_metrics.get("diagnosis_exact_match_rate", "N/A")),
            )
            eval_metric_3.metric(
                "Overall Claim Accuracy",
                str(claim_metrics.get("overall_claim_status_accuracy", "N/A")),
            )
            eval_metric_4.metric(
                "Retrieval Hit Rate",
                str(retrieval_metrics.get("evidence_retrieval_hit_rate", "N/A")),
            )

            st.dataframe(
                pd.DataFrame(current_evaluation_result.get("per_document_results", [])),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Governance & Limitations", expanded=False):
        st.warning("AI-assisted evidence review only. Human reviewer decision required.")
        st.write(
            "This prototype is intended to assist human reviewers by extracting evidence from "
            "synthetic clinical documents. It is not intended to diagnose patients, recommend "
            "treatment, or make final payment decisions."
        )
        st.markdown(
            "- Synthetic data only\n"
            "- Human validation required\n"
            "- Grounded evidence display\n"
            "- Final medical or payment authority stays with people\n"
            "- Rule-based coverage and OCR quality remain practical limitations"
        )

    with st.expander("Export Reviewer Report", expanded=False):
        report_payload = build_review_report(dict(st.session_state))
        summary_text = build_summary_text(report_payload)
        entities_csv = build_entities_csv(current_entity_result)
        documentation_csv = build_documentation_csv(current_documentation_result)
        report_json = json.dumps(report_payload, indent=2)

        st.text_area(
            "Reviewer summary preview",
            value=summary_text,
            height=220,
            disabled=True,
        )
        download_col1, download_col2 = st.columns(2)
        download_col1.download_button(
            "Download Full JSON Report",
            data=report_json,
            file_name=f"{Path(current_document_name).stem or 'review'}_review_report.json",
            mime="application/json",
            use_container_width=True,
        )
        download_col2.download_button(
            "Download Reviewer Summary TXT",
            data=summary_text,
            file_name=f"{Path(current_document_name).stem or 'review'}_review_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

        download_col3, download_col4 = st.columns(2)
        download_col3.download_button(
            "Download Entities CSV",
            data=entities_csv,
            file_name=f"{Path(current_document_name).stem or 'review'}_entities.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_col4.download_button(
            "Download Documentation Checklist CSV",
            data=documentation_csv,
            file_name=f"{Path(current_document_name).stem or 'review'}_documentation_checklist.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.caption("Prototype only. Synthetic data only. Human review required.")
