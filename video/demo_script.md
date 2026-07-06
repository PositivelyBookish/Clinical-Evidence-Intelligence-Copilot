# Demo Script: Clinical Evidence Intelligence Copilot

## Target Length

Approximately 5 to 7 minutes.

## Opening

Hello, my name is [Your Name], and this project was built for a Generative AI, NLP, and healthcare analytics internship assessment.  
My topic is clinical natural language technology for healthcare, specifically how NLP, OCR, and retrieval-based AI can support evidence-grounded review workflows in payment accuracy and clinical documentation analysis.

## Introduce the Healthcare Payment Review Pain Point

In healthcare payment and coding workflows, teams often need to determine whether a diagnosis, procedure, or code is actually supported by documentation inside a medical record.  
That sounds straightforward, but in practice it can be difficult because records may be long, inconsistent, scanned, partially missing, or ambiguous.

A reviewer may need to answer questions like:

- Is this diagnosis clearly documented?
- Was this procedure actually performed?
- Is the documentation sufficient for review?
- Is the evidence affirmed, negated, or uncertain?

That manual review process is time-consuming, and if documentation is weak or insufficient, it can contribute to improper payments, rework, and audit burden.

## Explain the Research Motivation

That is what motivated this proof of concept.  
Rather than building a general chatbot, I wanted to build a practical reviewer copilot that helps organize evidence inside a synthetic clinical document while keeping the human reviewer in control.

The app combines several ideas:

- OCR-aware document ingestion
- clinical text cleaning
- section detection
- entity extraction
- negation and uncertainty detection
- grounded evidence retrieval
- claim support checking
- documentation completeness review

The key design principle is that the app should help a human reviewer inspect evidence faster, but it should not make final medical, coding, or payment decisions.

## Walk Through the App

I’ll start at the top of the app, which is called Clinical Evidence Intelligence Copilot.  
The interface is designed to feel like a product prototype for healthcare analytics, with safety messaging, workflow status, and demo shortcuts in the sidebar.

On the left side, there are guided demo scenarios so I can quickly load examples for presentation.  
There is also a workflow checklist showing the end-to-end path from document loading to export.

## Show Supported Claim Scenario

First, I’ll click **Load Best Demo Scenario**.  
This loads a synthetic note where hypertension, an ECG, and the ICD code I10 are clearly documented.

When the note is loaded, I can show the extracted text and the cleaned text.  
The app also detects sections such as the assessment, procedures, medications, and provider signature.

Next, in the NLP Entities tab, the app extracts structured evidence including:

- the diagnosis hypertension
- the procedure ECG
- medications such as aspirin
- codes such as I10 and 93000
- administrative elements like patient name, provider, and date of service

Then, in the Claim Support Check tab, I can run the prefilled claim:

- Diagnosis: Hypertension
- Procedure: ECG
- Code: I10

The output shows this scenario as supported, along with evidence and reasoning tied to the document.

## Show Unsupported or Negated Claim Scenario

Next, I’ll show an edge case.  
I can either load the unsupported claim scenario or the negation scenario.

If I load the unsupported claim scenario, the note supports hypertension but does not support the ECG claim.  
This demonstrates that the app is not simply matching keywords loosely. It checks whether the procedure is actually documented in a supportive context.

If I load the negation scenario, the note includes pneumonia language, but it says things like “no evidence of pneumonia” and “pneumonia was ruled out.”  
That is important because healthcare review systems need to distinguish affirmed evidence from negated evidence.

When I run the claim support check for pneumonia, the app marks it as not supported because the evidence is negated.

## Show Missing Documentation

Next, in the Missing Documentation tab, the app checks whether the note includes basic documentation elements such as:

- patient name
- date of service
- provider name
- provider signature
- diagnosis support
- procedural support
- code references

This is useful because insufficient documentation is a very practical review issue.  
For example, a note may support a diagnosis clinically but still be missing a provider signature or date of service, which matters in real workflows.

The app summarizes that in a checklist and gives a completeness score.

## Show Evidence Q&A

In the Evidence Q&A tab, I can ask grounded questions such as:

- Does this document support hypertension?
- What procedures are documented?
- Is pneumonia ruled out?
- What documentation is missing?

The important part is that the answer is shown together with retrieved evidence snippets from the note.  
That grounding behavior helps reduce unsupported answers and makes the output easier for a reviewer to trust and validate.

## Show Evaluation

Next, I’ll show the Evaluation tab.  
This runs the proof of concept on a small synthetic test set with both Synthea-derived style notes and controlled edge-case notes.

The evaluation compares:

- entity extraction behavior
- claim support classification
- negation and uncertainty handling
- documentation completeness checks
- retrieval quality

I also explicitly report which optional models are actually available locally.  
If a model is not installed, the app shows that honestly rather than inventing performance.

## Show Governance

Finally, I’ll show the Governance and Limitations tab.  
This is an important part of the project because healthcare AI needs responsible framing.

The app clearly states that:

- it is intended only to assist human reviewers
- it uses synthetic data only
- it should not be used with real PHI in its current form
- it is not a diagnosis system
- it is not a final payment decision tool

It also lists important limitations such as OCR quality sensitivity, rule-based coverage limits, simple negation handling, and the fact that synthetic evaluation does not equal production performance.

## Conclude with Strategic Recommendation

To conclude, my recommendation is not to use AI as an autonomous adjudication tool.  
Instead, I would recommend a human-in-the-loop clinical evidence intelligence platform that improves reviewer efficiency, surfaces evidence clearly, flags missing documentation, and maintains traceability and governance.

This proof of concept shows how NLP, OCR, retrieval, and cautious summarization can be combined into a practical healthcare analytics workflow that is useful, explainable, and safer to evaluate.

Thank you for your time.
