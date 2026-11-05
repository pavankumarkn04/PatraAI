"""Run the local analysis model against the included, synthetic benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import load_project_env  # noqa: E402

load_project_env()

from document_processing import analyze_document_all_pages  # noqa: E402
from evaluation.evaluate import evaluate  # noqa: E402


def make_cases() -> list[dict[str, object]]:
    """Build 30 transparent, reproducible document/question pairs."""
    types = [
        ("scholarship notice", "District Scholarship Office"),
        ("admission notice", "State Admissions Cell"),
        ("utility notice", "Regional Electricity Board"),
        ("tax notice", "Revenue Assessment Office"),
        ("permit notice", "Transport Licensing Authority"),
        ("payment notice", "Public Works Department"),
    ]
    cases: list[dict[str, object]] = []
    for index in range(30):
        kind, office = types[index % len(types)]
        number = index + 1
        date = f"{(index % 27) + 1:02d}/{(index % 12) + 1:02d}/2026"
        reference = f"PAI-2026-{number:04d}"
        amount = f"Rs {1250 + index * 375:,}"
        days = (index % 25) + 5
        text = (
            f"{office}\n{kind.title()}\nDate: {date}\nReference Number: {reference}\n"
            f"Amount Due: {amount}. The applicant must reply within {days} days of this notice.\n"
            "Please quote the reference number when contacting the office."
        )
        cases.append({
            "id": f"synthetic-{number:03d}",
            "document_text": text,
            "question": "What is the amount due?",
            "expected_fields": {"date": date, "reference_number": reference, "amount": amount},
            "expected_answer": amount,
            "failure_category": "",
            "source_kind": "synthetic, generated from the visible template in run_cases.py",
        })
    return cases


def main() -> int:
    cases = make_cases()
    results_path = ROOT / "evaluation" / "results-live.jsonl"
    print(f"Running {len(cases)} synthetic cases against the configured analysis backend.", flush=True)
    results = []
    for index, case in enumerate(cases, start=1):
        try:
            prediction = analyze_document_all_pages(
                [str(case["document_text"])], str(case["question"])
            )
            predicted_fields = prediction.get("fields") or {}
            # Normalize common spellings while leaving unrecognized model keys intact.
            if isinstance(predicted_fields, dict):
                aliases = {"reference_no": "reference_number", "ref_no": "reference_number", "amount_due": "amount"}
                predicted_fields = {aliases.get(key, key): value for key, value in predicted_fields.items()}
            else:
                predicted_fields = {}
            result = {
                **case,
                "predicted_fields": predicted_fields,
                "predicted_answer": prediction.get("answer", ""),
                "predicted_summary": prediction.get("summary", ""),
                "run_error": "",
            }
        except Exception as exc:  # Keep failed model requests visible in the result file.
            result = {
                **case, "predicted_fields": {}, "predicted_answer": "",
                "predicted_summary": "", "run_error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(f"[{index}/{len(cases)}] {case['id']}: " + ("error: " + result["run_error"] if result["run_error"] else "done"), flush=True)
        results_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")

    # Include model errors in the explicit failure list and score observed outputs only.
    report = evaluate(results_path)
    report["source_kind"] = "30 synthetic cases; not a representative real-document benchmark"
    report["run_errors"] = [
        {"id": row["id"], "error": row["run_error"]} for row in results if row["run_error"]
    ]
    report_path = ROOT / "evaluation" / "report-live.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, ensure_ascii=False, indent=2), flush=True)
    print(f"Detailed report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
