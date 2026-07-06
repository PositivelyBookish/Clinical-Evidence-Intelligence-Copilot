# Clinical Evidence Intelligence Copilot

Clinical Evidence Intelligence Copilot is a healthcare NLP + GenAI proof of concept for evidence-grounded clinical document review in payment accuracy workflows.

## Why This Project Is a Strong Fit for the Role

This repository is intentionally designed to communicate the qualities a hiring evaluator would care about for a healthcare NLP, GenAI, and research-oriented internship:

- I understood the role and framed the project around a realistic healthcare analytics workflow.
- I researched Cotiviti-style business pain points such as payment accuracy, coding validation, chart review, and insufficient documentation.
- I chose a practical healthcare NLP and GenAI use case instead of a generic chatbot concept.
- I built a working local prototype rather than only describing an idea.
- I included responsible AI framing, human review boundaries, and governance thinking.
- I used synthetic data safely and explicitly avoided real PHI.
- I evaluated the prototype on synthetic labeled examples instead of making unsupported performance claims.
- I documented limitations and a future roadmap to show product and research judgment.

## Elevator Pitch

This project demonstrates how a reviewer-facing copilot can help healthcare payment and clinical review teams inspect synthetic medical documentation, extract grounded evidence, identify missing documentation, and check whether a diagnosis, procedure, or code appears supported. The system is intentionally human-in-the-loop, synthetic-data-only, and designed as a safe local demo rather than an autonomous decision-maker.

## Problem Statement

Healthcare payment and clinical review teams must determine whether claims, diagnoses, procedures, and codes are supported by medical documentation. Manual review is time-consuming, and missing or insufficient documentation can lead to improper payments.

Clinical Evidence Intelligence Copilot is built around that workflow reality. Instead of offering open-ended medical generation, it focuses on evidence visibility, documentation sufficiency, and grounded reviewer assistance.

## Cotiviti Relevance

This prototype is designed around the kinds of workflows that matter in payment accuracy and healthcare analytics:

- payment accuracy review
- clinical chart validation
- coding validation
- medical record review
- documentation sufficiency assessment
- responsible AI with human expert review

The app helps a reviewer find relevant evidence inside a synthetic clinical document faster, but it does not make final clinical, coding, utilization, or payment decisions. That boundary is deliberate and important.

## Features

- Synthetic document upload
- OCR-aware extraction
- Text cleaning
- Section detection
- Clinical entity extraction
- Negation and uncertainty detection
- Evidence snippets
- Claim support classification
- RAG-style evidence Q&A
- Missing documentation checker
- Evaluation dashboard
- Governance tab
- Exportable reviewer report

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- regex and rule-based heuristics
- PyMuPDF / pdfplumber optional
- pytesseract optional
- JSON / CSV export

Optional local enhancement paths include scispaCy, sentence-transformers, and Bio_ClinicalBERT-style experimentation when available.

## Architecture Diagram in Text

```text
Document Upload
→ Text Extraction
→ Cleaning
→ Section Detection
→ NLP Entity Extraction
→ Negation Detection
→ Evidence Retrieval
→ Claim Support Check
→ Documentation Completeness
→ Reviewer Summary
→ Export Report
```

## How to Run

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

The default requirements file installs the stable baseline demo. Optional advanced biomedical NLP models such as scispaCy, sentence-transformers, and Bio_ClinicalBERT-style dependencies can be added later if you want richer local experimentation.

4. Start the app:

```bash
streamlit run app.py
```

## How to Run Tests

```bash
pytest tests/
```

Notes:

- The test suite runs locally against bundled synthetic notes.
- Core tests do not depend on optional biomedical models.
- If optional packages are unavailable, the application falls back gracefully and the tests continue to validate the baseline workflow.

## How to Demo

1. Open the app.
2. Click `Load Best Demo Scenario`.
3. Process the document if needed.
4. Show the extracted text.
5. Show the NLP entities.
6. Run the claim support check.
7. Ask an evidence Q&A question.
8. Show missing documentation.
9. Run evaluation.
10. Export the report.
11. Show the governance page.

Recommended live-demo storyline:

- Start with the best supported case to show clean evidence extraction.
- Move to the unsupported or negation scenario to show safety and reviewer control.
- End on governance, evaluation, and export to show research discipline and product thinking.

## Assessment Context

This repository is structured as an internship assessment submission for a Generative AI / NLP / Research Engineer-style healthcare analytics role. The selected topic is:

**Clinical Natural Language Technology for Health Care: NLP, OCR, Computer Vision, LLM, and LMM**

The project is intended to support four related deliverables:

- a written report
- a hackathon-style proof of concept
- a PowerPoint presentation
- a recorded demo video

Supporting source files for those deliverables are included in:

- [report/final_report_draft.md](report/final_report_draft.md)
- [slides/final_presentation_content.md](slides/final_presentation_content.md)
- [video/final_demo_runbook.md](video/final_demo_runbook.md)
- [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)

## Synthetic Data Notice

This project uses synthetic documents only.

- No real PHI is included.
- No real patient data should be uploaded.
- Bundled synthetic notes allow the demo to run even without external data generation tools.
- Synthea-derived and controlled edge-case notes are used to simulate realistic review scenarios safely.

## Human Review Notice

This is not a final decision tool.

- It is not a medical diagnosis system.
- It does not recommend treatment.
- It does not make final payment decisions.
- It does not guarantee coding correctness.
- All outputs require human validation.

## Governance and Responsible AI Position

This proof of concept is intentionally conservative in how it uses AI:

- evidence snippets are surfaced for reviewer inspection
- grounded Q&A answers only from retrieved document text
- template-based summaries reduce unsupported generation
- missing evidence is explicitly reported rather than filled in
- governance and limitations are visible in the app, not hidden

The goal is to reduce reviewer search burden while preserving expert oversight.

## Evaluation Approach

The application is evaluated on synthetic labeled examples, including both Synthea-derived notes and controlled edge-case notes. The evaluation harness compares expected versus predicted behavior for:

- entity extraction
- negation and uncertainty detection
- claim support classification
- documentation completeness checks
- retrieval quality

These metrics are for proof-of-concept validation only and are not production accuracy claims.

## Repository Structure

```text
clinical-evidence-intelligence-copilot/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── synthea_raw/
│   ├── synthetic_documents/
│   ├── sample_claims.csv
│   └── expected_labels.json
├── scripts/
├── src/
│   ├── document_loader.py
│   ├── text_cleaner.py
│   ├── section_detector.py
│   ├── entity_extractor.py
│   ├── negation_detector.py
│   ├── evidence_retriever.py
│   ├── claim_support_checker.py
│   ├── documentation_checker.py
│   ├── summarizer.py
│   ├── evaluator.py
│   └── export_report.py
├── report/
├── slides/
├── video/
├── outputs/
├── tests/
└── assets/
```

## Module Overview

- `document_loader.py`: loads TXT, PDF, and image inputs with graceful fallbacks
- `text_cleaner.py`: normalizes whitespace and OCR-style issues
- `section_detector.py`: identifies clinical note sections
- `entity_extractor.py`: extracts diagnoses, symptoms, procedures, medications, codes, and administrative entities
- `negation_detector.py`: flags negated and uncertain evidence
- `evidence_retriever.py`: chunks documents and retrieves relevant evidence for grounded Q&A
- `claim_support_checker.py`: classifies whether a claim item appears supported
- `documentation_checker.py`: scores missing or incomplete documentation
- `summarizer.py`: builds a cautious reviewer summary
- `evaluator.py`: runs synthetic evaluation
- `export_report.py`: builds JSON, CSV, and text reviewer exports

## Known Limitations

- Rule-based extraction has limited coverage.
- Negation detection is intentionally simple and may miss complex phrasing.
- OCR support is optional and depends on local environment setup.
- Evaluation is synthetic-only and does not represent production performance.
- The system does not provide official medical coding validation.

## Future Roadmap

- Clinical language models
- RAG with vector database
- Layout-aware document AI
- LMM support
- Cloud deployment
- Reviewer feedback loop
- Audit logging
- HIPAA-compliant architecture
- Enterprise integration

## Assessment Deliverables

This repository is organized so the final assessment package can be assembled cleanly:

- `report/` for the written report
- `slides/` for the presentation deck
- `video/` for the demo script or recording notes
- `outputs/` for exported reviewer reports and demo artifacts

## Why This Project Works as a Portfolio Piece

This project is designed to show more than just code. It demonstrates:

- understanding of a real healthcare analytics pain point
- practical NLP and GenAI product framing
- safe use of synthetic data
- explicit responsible AI boundaries
- local prototyping discipline
- evaluation-minded engineering
- clear communication for a business and research audience
