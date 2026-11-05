"""Exercise real local model inference on bundled long and scanned PDFs."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import load_project_env  # noqa: E402

load_project_env()

from document_processing import analyze_document_all_pages, extract_pdf_pages  # noqa: E402
from evaluation.evaluate import normalize  # noqa: E402


def run_sample(path: Path, question: str, expected_marker: str) -> dict[str, object]:
    payload = path.read_bytes()
    extraction = extract_pdf_pages(payload)
    page_texts, page_count = extraction["page_texts"], extraction["page_count"]
    if len(page_texts) != page_count:
        raise RuntimeError(f"Page extraction mismatch: PDF reports {page_count} pages but extraction returned {len(page_texts)}.")
    prediction = analyze_document_all_pages(
        page_texts, question, pdf_bytes=payload, page_kinds=extraction["page_kinds"],
        stage_callback=lambda stage, done, total: print(f"  {stage}: {done}/{total}", flush=True),
    )
    answer = str(prediction.get("answer", ""))
    extracted_text = str(prediction.get("extracted_text", ""))
    result = {
        "sample": path.name,
        "page_count": page_count,
        "text_pages": len(extraction["text_pages"]),
        "scanned_pages": extraction["scanned_pages"],
        "blank_pages": extraction["blank_pages"],
        "ocr_pages": prediction.get("ocr_pages", []),
        "expected_marker": expected_marker,
        "marker_found_in_actual_output": normalize(expected_marker) in normalize(answer),
        "marker_found_in_extracted_text": normalize(expected_marker) in normalize(extracted_text),
        "fields": prediction.get("fields", {}),
        "field_sources": prediction.get("field_sources", {}),
        "analysis_backends": prediction.get("analysis_backends", []),
        "analysis_models": prediction.get("analysis_models", []),
        "summary": prediction.get("summary", ""),
        "answer": answer,
        "extracted_text": extracted_text,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="*", help="Optional exact sample filenames to run")
    args = parser.parse_args()
    samples = ROOT / "evaluation" / "samples"
    report = ROOT / "evaluation" / "pdf-samples-live.json"
    jobs = [
        (ROOT / "sample-check.pdf", "What is the amount due?", "Rs 25,000"),
        (samples / "long-searchable-11-pages.pdf", "What is the review code on the final page?", "END-PAGE-7319"),
        (samples / "long-searchable-101-pages.pdf", "What is the review code on the final page?", "FINAL-101-PAGE-8472"),
        (samples / "scanned-3-pages.pdf", "What is the verification code on the final scanned page?", "VISION-9182"),
        (samples / "kannada-scanned-2-pages.pdf", "What is the last date to submit the application?", "30-11-2026"),
        (samples / "hindi-scanned-2-pages.pdf", "What is the last date to submit the application?", "30-11-2026"),
    ]
    if args.samples:
        requested = set(args.samples)
        jobs = [job for job in jobs if job[0].name in requested]
        missing = requested - {job[0].name for job in jobs}
        if missing:
            parser.error("Unknown sample name(s): " + ", ".join(sorted(missing)))
    if not jobs:
        parser.error("No PDF samples selected.")
    results = []
    # Start fresh for this run so a failed run cannot be mistaken for the
    # output from an earlier version of the application.
    report.write_text("[]", encoding="utf-8")
    for index, (path, question, marker) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Analyzing {path.name}…", flush=True)
        try:
            results.append(run_sample(path, question, marker))
        except Exception as exc:
            results.append({
                "sample": path.name,
                "expected_marker": marker,
                "marker_found_in_actual_output": False,
                "run_error": f"{type(exc).__name__}: {exc}",
            })
        report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        result = results[-1]
        status = result.get("run_error") or str(result["marker_found_in_actual_output"])
        print(f"[{index}/{len(jobs)}] {path.name}: {status}", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"Detailed report: {report}", flush=True)
    return 0 if all(row["marker_found_in_actual_output"] and row.get("marker_found_in_extracted_text", False) for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
