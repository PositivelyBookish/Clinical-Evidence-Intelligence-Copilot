# Final Demo Runbook

This runbook is optimized for a 5 to 7 minute final submission video.

## Goal

Show that the project is:

- relevant to Cotiviti-style healthcare analytics workflows
- technically practical
- evidence-grounded
- safe and human-in-the-loop

## Recommended Video Structure

### 0:00 to 0:30 - Introduce the assignment and topic

Say:

This project was built for a Generative AI, NLP, and healthcare analytics internship assessment. I chose the topic of clinical natural language technology in healthcare, focusing on how NLP, OCR, and retrieval-grounded AI can support payment accuracy and clinical documentation review.

### 0:30 to 1:15 - Explain the problem

Say:

In healthcare payment and chart review workflows, teams often need to determine whether a diagnosis, procedure, or code is actually supported by documentation. That can be difficult because records may be long, scanned, messy, incomplete, or ambiguous. Reviewers spend time searching for evidence, checking whether findings are affirmed or negated, and identifying documentation gaps.

### 1:15 to 1:45 - Introduce the product concept

Say:

I built Clinical Evidence Intelligence Copilot, a local proof of concept that helps reviewers inspect synthetic clinical documents, extract evidence, check support for a claim, flag missing documentation, and ask grounded questions over the document while keeping the human reviewer in control.

### 1:45 to 3:00 - Show the strongest supported scenario

Actions:

1. Open the app.
2. Load `clean_supported_claim.txt`.
3. Show the extracted text and detected sections.
4. Open the Entities tab and show:
   - Hypertension
   - ECG
   - Aspirin
   - I10
   - 93000
5. Open Claim Check and load the sample claim.
6. Show that the diagnosis, procedure, and code are treated as supported with evidence.

Say:

This is the clean supported scenario. The system extracts the main entities, links them to evidence, and shows a reviewer-facing support interpretation instead of just generating a generic summary.

### 3:00 to 4:00 - Show a safety scenario

Recommended choice:

- `negated_diagnosis.txt`

Actions:

1. Load the negation scenario.
2. Show the entity table.
3. Point out that pneumonia is present as a term but marked negated.
4. Run claim support for pneumonia.
5. Show that the result is not supported or unclear rather than falsely supported.

Say:

This demonstrates why contextual handling matters. The system should not treat keyword presence alone as valid support when the diagnosis is negated or ruled out.

### 4:00 to 4:40 - Show missing documentation

Recommended choice:

- `missing_signature.txt`

Actions:

1. Load the missing signature scenario.
2. Open the Documentation tab.
3. Show the completeness score and checklist.
4. Point out the missing provider signature.

Say:

This matters because a chart can contain clinical content but still be operationally incomplete for review.

### 4:40 to 5:20 - Show Evidence Q&A

Actions:

1. Ask:
   - `Does this document support hypertension?`
   - or `Is pneumonia ruled out?`
2. Show the answer together with evidence snippets and retrieval method.

Say:

The important design choice here is that answers are grounded in retrieved document chunks rather than generated from outside knowledge.

### 5:20 to 6:00 - Show evaluation

Actions:

1. Open the Evaluation section.
2. Run the synthetic evaluation set.
3. Show:
   - entity extraction metrics
   - claim support metrics
   - negation metrics
   - documentation metrics
   - retrieval metrics
4. Point out that unavailable optional models are reported honestly.

Say:

The goal of the evaluation is not to claim production accuracy. It is to demonstrate model-aware validation and research discipline.

### 6:00 to 6:40 - Show governance

Actions:

1. Open Governance & Limitations.
2. Highlight:
   - synthetic data only
   - not a diagnosis tool
   - not a final payment decision tool
   - human validation required

Say:

This is a healthcare AI prototype, so governance is part of the product, not an afterthought.

### 6:40 to 7:00 - Close with recommendation

Say:

My recommendation is not to automate final payment or clinical decisions. Instead, I would invest in a human-in-the-loop evidence intelligence platform that improves reviewer speed, evidence traceability, and documentation visibility while preserving expert oversight.

## Best Demo Documents by Purpose

- `clean_supported_claim.txt`: best end-to-end positive example
- `unsupported_procedure.txt`: best unsupported-claim example
- `negated_diagnosis.txt`: best safety and assertion-status example
- `missing_signature.txt`: best documentation completeness example

## Final Delivery Tips

- Record at 125 percent browser zoom if text feels small.
- Keep the app on the most polished tabs.
- Avoid long pauses while typing.
- Rehearse the claim values before recording.
- Do one clean take rather than a very long improvisational recording.
