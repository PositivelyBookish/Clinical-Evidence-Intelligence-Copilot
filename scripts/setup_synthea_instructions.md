# Synthea Setup Instructions

## Overview

Synthea is **optional but recommended** for this project. It is the preferred synthetic patient generator because it produces realistic, non-real patient records that can be converted into clinical-note-style documents for the Clinical Evidence Intelligence Copilot demo.

The application is designed to continue working even if Synthea is not installed locally. When Synthea output is unavailable, the repo falls back to bundled synthetic sample notes and controlled edge-case notes for testing.

## Safety Requirements

- Use synthetic data only.
- Do not upload or process real patient data.
- Do not use protected health information (PHI).
- Store local Synthea export files under `data/synthea_raw/`.

## Recommended Local Workflow

If Synthea is already installed or you want to install it locally, you can use commands like the following. These are example commands only and may not work exactly the same in every environment.

```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea
./gradlew build check test
./run_synthea -p 20
```

## Where to Put Synthea Output

After generating synthetic records, copy the exported files into this repository:

- Copy generated FHIR JSON files into `data/synthea_raw/fhir/`
- Copy generated CSV files into `data/synthea_raw/csv/`

The conversion script primarily looks for FHIR JSON files under:

```text
data/synthea_raw/fhir/
```

## Repo Scripts

- `scripts/generate_synthea_data.py`
  Attempts to locate a local Synthea installation, optionally run generation, and copy outputs into this repo.
- `scripts/convert_synthea_to_notes.py`
  Converts Synthea FHIR JSON data into readable clinical note text files for NLP testing.
- `scripts/create_fallback_synthetic_notes.py`
  Generates bundled fallback notes and edge-case review documents even when Synthea is unavailable.

## Fallback Behavior

If Synthea is not installed:

- The project still runs locally.
- Bundled fallback synthetic notes remain available in `data/synthetic_documents/`.
- Evaluation labels remain available in `data/expected_labels.json`.

This design keeps the proof of concept portable for internship-demo conditions without requiring external services or real patient data.
