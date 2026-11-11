"""Check the published seven-operator results from saved CPU inputs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "cpu"))

from measure_labelled_zigzags import analyze
from tools.audit_labelled_sampling_bounds import verify


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    results = ROOT / "results"
    status = read_json(ROOT / "STATUS.json")
    barcodes = read_json(ROOT / "cpu" / "MEASURED_ZIGZAG_BARCODES.json")
    expected_diagrams = {row["case"]: row for row in barcodes["cases"]}
    expected_bounds = read_json(results / "INDEPENDENT_SAMPLING_BOUNDS_AUDIT.json")

    checked_bounds = []
    checked_diagrams = set()
    for row in status["registered_cases"]:
        case = row["case"]
        directory = results / row["directory"]
        word = row["assessment"] == "word_family_evaluation_5500"
        raw_audit = read_json(directory / "CASE_RAW_AUDIT.json")
        if raw_audit["case"] != case or raw_audit["scientific_case_pass"] is not True:
            raise ValueError(f"Missing successful raw audit for {case}")

        if word:
            report = read_json(directory / "RESULT.json")
        else:
            final = read_json(directory / "FINAL_RESULT.json")
            report = read_json(directory / final["report"])
            diagram = {**analyze(report), "assessment": row["assessment"]}
            if diagram != expected_diagrams[case]:
                raise ValueError(f"Measured diagram differs for {case}")
            checked_diagrams.add(case)
        if report["case"] != case:
            raise ValueError(f"Report case ID differs for {case}")
        checked_bounds.append(verify(report, word))

    score_bounds = sum(row["score_bounds_verified"] for row in checked_bounds)
    contrasts = sum(row["paired_contrasts_verified"] for row in checked_bounds)
    if (len(checked_bounds) != status["registered_total"]
            or len({row["case"] for row in checked_bounds}) != len(checked_bounds)
            or checked_diagrams != expected_diagrams.keys()
            or score_bounds != expected_bounds["individual_score_bounds"]
            or contrasts != expected_bounds["paired_contrasts"]):
        raise ValueError("The verified reports do not cover the archived results")

    print(f"Verified {len(checked_bounds)} cases, {score_bounds} score bounds, "
          f"{contrasts} paired contrasts, and {len(checked_diagrams)} measured diagrams.")


if __name__ == "__main__":
    main()
