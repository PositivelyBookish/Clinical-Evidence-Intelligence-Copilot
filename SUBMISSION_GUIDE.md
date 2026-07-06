# Submission Guide

This document translates the assessment requirements into a practical final submission package for **Clinical Evidence Intelligence Copilot**.

## Submission Objective

The goal of this project is to demonstrate four things clearly:

1. You understood the business problem.
2. You selected a realistic healthcare NLP and GenAI use case.
3. You built a working proof of concept rather than only proposing an idea.
4. You understand responsible AI boundaries in a healthcare analytics context.

## Recommended Final Submission Package

Submit the following items together:

1. **Written report**
   - Final file format: Microsoft Word or PDF
   - Target length: 2 pages of report content plus 1 bibliography page

2. **Hackathon proof of concept**
   - Final artifact: runnable Streamlit app
   - Main entry point: [app.py](app.py)
   - Supporting explanation: [README.md](README.md)

3. **Slide presentation**
   - Final file format: PowerPoint
   - Recommended length: 8 slides

4. **Recorded demo video**
   - Final artifact: screen recording with slides and app walkthrough
   - Recommended length: 5 to 7 minutes

## What the Evaluator Should Understand Quickly

Your submission should make these points obvious within the first minute:

- The problem is not generic healthcare AI.
- The problem is **clinical evidence review for payment accuracy and documentation sufficiency**.
- The app is designed for **human reviewers**, not autonomous adjudication.
- The prototype uses **synthetic data only**.
- The system is **evidence-grounded**, not a free-form chatbot.
- The project includes **evaluation, governance, and limitations**, not just a UI demo.

## Deliverable Mapping

### Written Report

The written report should communicate:

- what clinical natural language technology means in this context
- how the field evolved from manual review and keyword search to modern NLP, OCR, transformer, and retrieval methods
- why the use case matters for payment accuracy and chart review
- what opportunities and threats exist
- what Cotiviti-style strategic direction makes sense

The strongest framing is:

**Near-term value comes from human-in-the-loop evidence organization, not autonomous medical or payment decision-making.**

### Hackathon Proof of Concept

The proof of concept should demonstrate:

- synthetic note ingestion
- text extraction and cleaning
- clinical section detection
- entity extraction
- negation and uncertainty handling
- claim support classification
- evidence-grounded Q&A
- missing documentation scoring
- exportable reviewer output
- governance and evaluation

The strongest framing is:

**This prototype reduces reviewer search burden by surfacing evidence and documentation gaps faster.**

### Slide Presentation

The presentation should not repeat the entire report word for word. It should:

- define the problem
- show why the chosen technology stack fits it
- summarize the proof of concept architecture
- show one strong supported example and one safe failure-mode example
- end on governance and recommendation

### Recorded Demo

The video should show:

1. problem and motivation
2. solution concept
3. app walkthrough
4. supported scenario
5. negated or unsupported scenario
6. missing documentation
7. evaluation
8. governance
9. closing recommendation

## Best Storyline for the Submission

Use this exact narrative arc across the report, slides, and video:

1. **Healthcare reviewers spend time searching through documentation.**
2. **Not all evidence is equally useful because diagnoses may be negated, uncertain, or weakly documented.**
3. **Modern clinical NLP can help organize evidence, but healthcare workflows require traceability and human control.**
4. **Clinical Evidence Intelligence Copilot demonstrates a practical reviewer-facing workflow using synthetic data.**
5. **The prototype is intentionally grounded, cautious, and governance-aware.**
6. **The strategic recommendation is augmentation, not automation.**

## What to Highlight in the App Demo

### Strongest Demo Sequence

1. Load `clean_supported_claim.txt`
2. Show extracted sections and entities
3. Show claim support for:
   - Diagnosis: Hypertension
   - Procedure: ECG
   - Code: I10
4. Ask a grounded question such as:
   - `Does this document support hypertension?`
5. Show missing documentation score
6. Export the reviewer report

### Safety Sequence

Then switch to one of these:

- `negated_diagnosis.txt`
- `unsupported_procedure.txt`
- `missing_signature.txt`

Use this to show that the app:

- does not blindly match keywords
- does not treat negated evidence as supportive
- does not ignore documentation gaps
- keeps the human reviewer in control

## Recommended Talking Points by Deliverable

### For the Report

- Clinical NLP is useful in healthcare when the goal is evidence extraction from unstructured text.
- OCR matters because many operational medical records are scanned or messy.
- LLM and RAG methods are most valuable when grounded in source evidence.
- Payment review is a strong use case because it depends on documentation support, traceability, and reviewer judgment.

### For the Slides

- Keep text concise.
- Use one architecture diagram slide.
- Use one slide that compares supported versus negated/unsupported outcomes.
- Use one slide for risks and governance.

### For the Video

- Do not over-explain every tab.
- Move quickly through the interface.
- Spend more time on the business value and the safety framing.
- Use the app to prove the concept, not to show every implementation detail.

## Recommended Deliverable Mapping

- [README.md](README.md): project overview for evaluator or reviewer
- written report: final Word or PDF document
- slide deck: final PowerPoint
- recorded demo: final video file
- [outputs/](outputs/): exported reviewer reports for screenshots or appendix material

## Suggested Final File Names

Use clean, professional names when you export the final submission files:

- `Vanaja_Agarwal_Cotiviti_Report.docx`
- `Vanaja_Agarwal_Cotiviti_Presentation.pptx`
- `Vanaja_Agarwal_Cotiviti_Demo.mp4`
- `clinical-evidence-intelligence-copilot.zip`

## Final QA Checklist Before Submission

### Content QA

- The report clearly defines the topic and ties it to healthcare payment review.
- The report includes opportunities, threats, and a recommendation.
- The slides tell the same story as the report but in shorter form.
- The demo script matches the actual app behavior.

### Product QA

- `streamlit run app.py` works locally.
- Demo scenarios load successfully.
- Claim support results are understandable.
- Governance messaging is visible and credible.
- Export report works.

### Safety QA

- The submission says synthetic data only.
- The submission says human review required.
- The submission avoids claiming production accuracy.
- The submission avoids claiming final coding or payment correctness.

### Presentation QA

- The first 30 seconds explain the problem well.
- The app UI looks organized and not cluttered.
- The video includes both a success case and a failure/safety case.
- The conclusion makes a strategic recommendation rather than only describing features.

## Final Recommendation

For the final hand-in, present this project as:

**a practical, evidence-grounded healthcare NLP and GenAI prototype for payment-accuracy-style review workflows that balances technical ambition with responsible AI discipline.**
