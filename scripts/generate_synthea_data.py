"""Optional helper for generating or importing Synthea synthetic data."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "synthea_raw"
FHIR_DIR = RAW_DIR / "fhir"
CSV_DIR = RAW_DIR / "csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally run a local Synthea installation and copy generated output "
            "into this repository."
        )
    )
    parser.add_argument(
        "--synthea-path",
        type=Path,
        default=None,
        help="Path to a local Synthea checkout.",
    )
    parser.add_argument(
        "--patient-count",
        type=int,
        default=20,
        help="Number of synthetic patients to generate if Synthea is run.",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Skip generation and only copy existing export files into this repo.",
    )
    return parser.parse_args()


def find_synthea_path(explicit_path: Path | None) -> Path | None:
    candidates = [
        explicit_path,
        Path.cwd() / "synthea",
        REPO_ROOT / "synthea",
        REPO_ROOT.parent / "synthea",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return None


def copy_tree_contents(source_dir: Path, target_dir: Path, pattern: str) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_path in source_dir.glob(pattern):
        if source_path.is_file():
            shutil.copy2(source_path, target_dir / source_path.name)
            count += 1
    return count


def copy_exports(synthea_dir: Path) -> tuple[int, int]:
    output_root = synthea_dir / "output"
    fhir_source_candidates = [
        output_root / "fhir",
        output_root / "fhir_stu3",
        output_root / "fhir_r4",
    ]
    csv_source = output_root / "csv"

    fhir_count = 0
    for candidate in fhir_source_candidates:
        if candidate.exists():
            fhir_count += copy_tree_contents(candidate, FHIR_DIR, "*.json")

    csv_count = 0
    if csv_source.exists():
        csv_count += copy_tree_contents(csv_source, CSV_DIR, "*.csv")

    return fhir_count, csv_count


def maybe_run_synthea(synthea_dir: Path, patient_count: int) -> None:
    run_script = synthea_dir / "run_synthea"
    gradlew = synthea_dir / "gradlew"
    if not run_script.exists():
        raise FileNotFoundError(
            f"Could not find run_synthea in {synthea_dir}. "
            "Use --copy-only if you only want to import existing data."
        )

    if gradlew.exists():
        print("Synthea build files found. You may need to run './gradlew build check test' manually first.")

    print(f"Running Synthea from {synthea_dir} for {patient_count} synthetic patients...")
    subprocess.run(
        [str(run_script), "-p", str(patient_count)],
        cwd=synthea_dir,
        check=True,
    )


def main() -> int:
    args = parse_args()
    FHIR_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    synthea_dir = find_synthea_path(args.synthea_path)
    if not synthea_dir:
        print("No local Synthea installation found.")
        print("You can still use bundled fallback notes via scripts/create_fallback_synthetic_notes.py.")
        return 0

    print(f"Using Synthea directory: {synthea_dir}")
    if not args.copy_only:
        maybe_run_synthea(synthea_dir, args.patient_count)

    fhir_count, csv_count = copy_exports(synthea_dir)
    print(f"Copied {fhir_count} FHIR JSON files into {FHIR_DIR}")
    print(f"Copied {csv_count} CSV files into {CSV_DIR}")

    if fhir_count == 0 and csv_count == 0:
        print(
            "No export files were found under the expected Synthea output folders. "
            "Check the Synthea version and export configuration."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
