# Final Presentation Content

This file is designed to be copied into PowerPoint. Each slide includes a recommended title, on-slide bullets, and short speaker notes.

## Slide 1: Title

### On-slide content

- **Clinical Evidence Intelligence Copilot**
- Clinical Natural Language Technology for Payment Accuracy
- NLP, OCR, and RAG-Based Evidence Review in Healthcare
- Vanaja Agarwal
- Cotiviti Generative AI / NLP Research Internship Assessment

### Speaker note

This project explores how clinical natural language technology can support payment accuracy and clinical documentation review through a practical, human-in-the-loop prototype.

## Slide 2: Problem and Cotiviti Relevance

### On-slide content

- Review teams must determine whether documentation supports:
  - diagnoses
  - procedures
  - codes
- Manual chart review is slow and inconsistent
- Missing documentation creates rework and payment risk
- Relevant workflow areas:
  - payment accuracy
  - coding validation
  - chart review
  - documentation sufficiency

### Speaker note

The core pain point is not a lack of records. It is the time and effort required to find grounded supporting evidence inside messy records.

## Slide 3: Technology Background

### On-slide content

- **Past**
  - manual review
  - keyword search
  - rule-based NLP
  - basic OCR
- **Present**
  - clinical NLP
  - biomedical NER
  - transformer models
  - grounded retrieval
- **Future**
  - multimodal LMMs
  - layout-aware chart intelligence
  - agentic reviewer workflows

### Speaker note

Different technologies solve different parts of the problem. OCR gets text into the system, NLP structures it, and retrieval-grounded AI helps answer reviewer questions with evidence.

## Slide 4: Why This Use Case Matters

### On-slide content

- Diagnoses may be:
  - affirmed
  - negated
  - uncertain
- Procedures may be claimed but not documented
- Codes may appear without enough textual support
- Records may be scanned, incomplete, or noisy
- Reviewer trust requires evidence traceability

### Speaker note

This is a strong use case for healthcare AI because it requires not only extraction, but contextual interpretation and cautious workflow design.

## Slide 5: Proposed Solution

### On-slide content

- **Clinical Evidence Intelligence Copilot**
- Reviewer-facing local prototype
- Uses synthetic clinical notes only
- Main functions:
  - extract evidence
  - detect negation and uncertainty
  - check claim support
  - flag missing documentation
  - answer grounded questions
- Human reviewer remains in control

### Speaker note

I intentionally avoided building a generic chatbot. The product idea is a targeted evidence-review copilot for healthcare payment accuracy workflows.

## Slide 6: POC Architecture

### On-slide content

```text
Document Upload
→ Text Extraction
→ Cleaning
→ Section Detection
→ Clinical Entity Extraction
→ Negation / Uncertainty Detection
→ Evidence Retrieval
→ Claim Support Check
→ Documentation Completeness
→ Reviewer Summary
→ Export Report
```

### Speaker note

The architecture is designed to be modular, local, and demo-friendly. It works without paid APIs and falls back gracefully when optional models are missing.

## Slide 7: Demo Results and Validation

### On-slide content

- Supported scenario:
  - Hypertension
  - ECG
  - I10
- Safety scenarios:
  - negated diagnosis
  - unsupported procedure
  - missing signature
- Evaluation areas:
  - entity extraction
  - negation handling
  - claim support classification
  - documentation completeness
  - retrieval quality

### Speaker note

The strongest part of the demo is showing both success and safe failure. The app should support a claim when evidence is present, and avoid overclaiming when evidence is negated, missing, or unclear.

## Slide 8: Risks, Governance, and Recommendation

### On-slide content

- Risks:
  - hallucination
  - OCR errors
  - privacy and PHI concerns
  - over-automation
  - coding correctness risk
- Governance:
  - synthetic data only
  - evidence-grounded outputs
  - human validation required
- Recommendation:
  - invest in human-in-the-loop evidence intelligence
  - do not automate final clinical or payment decisions

### Speaker note

My recommendation is a reviewer augmentation strategy. The most useful and responsible near-term healthcare AI products are the ones that improve evidence visibility and reviewer productivity while preserving traceability and expert oversight.
