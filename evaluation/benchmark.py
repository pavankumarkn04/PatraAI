"""Developer evaluation: run the curated dataset through the real PatraAI pipeline.

    py -3.11 evaluation/benchmark.py            # run pending cases, then write the report
    py -3.11 evaluation/benchmark.py --rerun    # discard previous results and run everything
    py -3.11 evaluation/benchmark.py --report   # rebuild the report from saved results

Each document is analyzed once (text extraction, OCR of scanned pages, field
extraction, summary) and cached in evaluation/results/cache, so its questions do
not repeat that work. Results are saved after every case: an interrupted run
continues where it stopped. A result is reused only while the case and the
model configuration are unchanged.

Outputs in evaluation/results/: CS3235-evaluation-report.md (academic report),
evaluation-results.csv (every check) and evaluation-results.json (everything).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import load_project_env  # noqa: E402

load_project_env()

import document_processing as dp  # noqa: E402
from evaluation import scoring  # noqa: E402

DATASET = ROOT / "evaluation" / "dataset"
RESULTS = ROOT / "evaluation" / "results"
RESULTS_JSON, RESULTS_CSV, REPORT_FILE = "evaluation-results.json", "evaluation-results.csv", "CS3235-evaluation-report.md"
Progress = Callable[[str], None]


IMPORTED_CASES = "cases_imported.jsonl"
REQUIRED_KEYS = ("id", "document", "document_type", "language", "format", "source", "source_reference", "question",
                 "expected_answer", "answer_pages", "expected_fields", "field_pages", "verification")


def load_cases() -> list[dict[str, Any]]:
    """The curated cases plus any verified cases imported from JSONL."""
    cases = []
    for name in ("cases.jsonl", IMPORTED_CASES):
        path = DATASET / name
        if path.is_file():
            cases += [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return cases


def import_cases(payload: bytes) -> tuple[int, list[tuple[int, str]]]:
    """Validate JSONL test cases and append the valid ones to cases_imported.jsonl."""
    existing = {case["id"] for case in load_cases()}
    added, problems = [], []
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append((number, f"invalid JSON ({exc.msg})"))
            continue
        missing = [key for key in REQUIRED_KEYS if not case.get(key)]
        if missing:
            problems.append((number, "missing " + ", ".join(missing)))
        elif case["id"] in existing:
            problems.append((number, f"duplicate id {case['id']}"))
        elif not (DATASET / "documents" / case["document"]).is_file():
            problems.append((number, f"document {case['document']} not found in evaluation/dataset/documents"))
        elif set(case["field_pages"]) != set(case["expected_fields"]):
            problems.append((number, "field_pages must list a page for every expected field"))
        else:
            added.append(case)
            existing.add(case["id"])
    if added:
        with (DATASET / IMPORTED_CASES).open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(case, ensure_ascii=False) + "\n" for case in added)
    return len(added), problems


def pipeline_signature() -> dict[str, Any]:
    """Configuration that changes predictions; results are reused only while it matches."""
    service = dp.analysis_service_description()
    return {
        "analysis_service": service["kind"], "model": service["model"], "vision_model": service.get("vision_model"),
        "single_call_tokens": dp.SINGLE_CALL_TOKENS, "field_evidence_chars": dp.FIELD_EVIDENCE_CHARS,
        "summary_sections": dp.SUMMARY_MAX_SECTIONS, "summary_section_chars": dp.SUMMARY_SECTION_CHARS,
        "qa_top_k": dp.QA_RETRIEVAL_TOP_K, "qa_chunk_chars": dp.QA_CHUNK_CHARS,
        "indic_chars_per_token": dp.INDIC_CHARS_PER_TOKEN, "prompt_version": dp.ANALYSIS_VERSION,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def extract_document(name: str, payload: bytes) -> tuple[dict[str, Any], Any]:
    """Read a PDF, Word or image document the same way the website does."""
    suffix = Path(name).suffix.casefold()
    if suffix == ".pdf":
        return dp.extract_pdf_pages(payload), None
    if suffix == ".docx":
        return dp.extract_docx(payload), None
    from PIL import Image

    image = Image.open(io.BytesIO(payload)).convert("RGB")
    text = dp.local_ocr([image])
    scanned = [] if text.strip() else [1]
    return {"page_count": 1, "page_texts": [text], "page_kinds": ["text" if text.strip() else "scanned"],
            "scanned_pages": scanned, "garbled_pages": [], "blank_pages": [], "unit": "page"}, image


def analyze_document(name: str, progress: Progress) -> dict[str, Any]:
    """Run stages 1-3 of the pipeline for one dataset document, with a disk cache."""
    payload = (DATASET / "documents" / name).read_bytes()
    cache_path = RESULTS / "cache" / f"{hashlib.sha256(payload).hexdigest()[:16]}-{_digest(pipeline_signature())}.json"
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    if cached.get("analysis"):
        return cached
    started = time.perf_counter()
    extraction, image = extract_document(name, payload)
    ocr = {int(page): value for page, value in (cached.get("ocr") or {}).items()}
    scanned = extraction["scanned_pages"]
    if scanned and image is not None:
        if 1 not in ocr:
            progress("    OCR 0/1 image")
            ocr[1] = dp.ocr_page_image(image, 1)
    elif scanned:
        def save_ocr(stage: str, done: int, total: int) -> None:
            progress(f"    OCR {done}/{total} scanned page(s)")
            _write_json(cache_path, {"ocr": ocr})  # resume point
        dp.ocr_scanned_pages(payload, scanned, cache=ocr, stage_callback=save_ocr)
    pages = dp.merge_ocr_text(extraction["page_texts"], {page: ocr[page] for page in scanned})
    progress("    field extraction and summary")
    analysis = dp.analyze_document_text(pages)
    result = {
        "document": name, "pages": pages, "page_count": extraction["page_count"], "unit": extraction.get("unit", "page"),
        "scanned_pages": scanned, "garbled_pages": extraction.get("garbled_pages", []), "blank_pages": extraction.get("blank_pages", []),
        "ocr": ocr, "analysis": analysis, "seconds": round(time.perf_counter() - started, 1),
    }
    _write_json(cache_path, result)
    return result


def score_case(case: dict[str, Any], document: dict[str, Any] | None, answer: dict[str, Any] | None, error: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    fields = ((document or {}).get("analysis") or {}).get("fields") or {}
    for field, expected in case["expected_fields"].items():
        predicted = fields.get(field)
        correct = not error and scoring.field_matches(expected, predicted)
        category = "" if correct else ("Processing error" if error else scoring.field_error_category(expected, predicted))
        if not correct and not error and category == "Missing extracted field":
            pages = (document or {}).get("pages") or []
            scanned = set((document or {}).get("scanned_pages") or [])
            if any(page in scanned and len((pages[page - 1] if 0 < page <= len(pages) else "").strip()) < 80 for page in case["field_pages"].get(field, [])):
                category = "OCR or document-reading failure"
        checks.append({"kind": "field", "check": field, "expected": expected, "predicted": predicted, "correct": correct,
                       "error_category": category, "pages": case["field_pages"].get(field, [])})
    predicted_answer = str((answer or {}).get("answer") or "")
    correct = not error and scoring.answer_matches(case["expected_answer"], predicted_answer)
    category = "" if correct else ("Processing error" if error else scoring.answer_error_category(case["expected_answer"], predicted_answer))
    if not correct and not error:
        pages = (document or {}).get("pages") or []
        scanned = set((document or {}).get("scanned_pages") or [])
        unreadable = [page for page in case["answer_pages"] if page in scanned and len((pages[page - 1] if 0 < page <= len(pages) else "").strip()) < 80]
        if unreadable:
            category = "OCR or document-reading failure"
        elif (answer or {}).get("grounding_warning"):
            category = "Unsupported or hallucinated information"
    checks.append({"kind": "qa", "check": "answer", "expected": case["expected_answer"], "predicted": predicted_answer,
                   "correct": correct, "error_category": category, "pages": case["answer_pages"]})
    predicted_pages = (answer or {}).get("answer_pages") or []
    page_ok = not error and bool(set(predicted_pages) & set(case["answer_pages"]))
    checks.append({"kind": "page", "check": "answer page", "expected": case["answer_pages"], "predicted": predicted_pages,
                   "correct": page_ok, "error_category": "" if page_ok else ("Processing error" if error else "Wrong or missing page reference"),
                   "pages": case["answer_pages"]})
    return {
        "checks": checks,
        "fields_correct": all(check["correct"] for check in checks if check["kind"] == "field"),
        "qa_correct": next(check["correct"] for check in checks if check["kind"] == "qa"),
    }


def load_results() -> dict[str, Any]:
    # latest.json is the file name used by earlier versions of this script;
    # whichever results file was written most recently is current.
    paths = [path for path in (RESULTS / RESULTS_JSON, RESULTS / "latest.json") if path.is_file()]
    if not paths:
        return {}
    return json.loads(max(paths, key=lambda path: path.stat().st_mtime).read_text(encoding="utf-8"))


def failed_case_ids() -> list[str]:
    return [row["case"]["id"] for row in load_results().get("cases", []) if row["error"] or not (row["fields_correct"] and row["qa_correct"])]


def run(rerun: bool = False, progress: Progress = print, only: list[str] | None = None) -> dict[str, Any]:
    """Evaluate pending cases (or all with rerun=True, or exactly `only`)."""
    cases = load_cases()
    signature = pipeline_signature()
    saved = {} if rerun else {row["case"]["id"]: row for row in load_results().get("cases", [])}
    for case_id in only or []:
        saved.pop(case_id, None)
    dp.check_model_service()
    rows: dict[str, Any] = {}
    documents: dict[str, Any] = {}
    for index, case in enumerate(cases, start=1):
        fingerprint = _digest({"case": case, "pipeline": signature})
        if case["id"] in saved and saved[case["id"]].get("fingerprint") == fingerprint:
            rows[case["id"]] = saved[case["id"]]
            continue
        progress(f"[{index}/{len(cases)}] {case['id']} ({case['document']})")
        started = time.perf_counter()
        document = answer = None
        error = ""
        try:
            if case["document"] not in documents:
                documents[case["document"]] = analyze_document(case["document"], progress)
            document = documents[case["document"]]
            answer = dp.answer_document_question(document["pages"], case["question"], summary=dp.summary_text(document["analysis"]))
        except Exception as exc:  # Keep failures visible as scored processing errors.
            error = f"{type(exc).__name__}: {exc}"
        scored = score_case(case, document, answer, error)
        rows[case["id"]] = {
            "case": case, "fingerprint": fingerprint, "error": error,
            "predicted_answer": (answer or {}).get("answer", ""), "predicted_pages": (answer or {}).get("answer_pages", []),
            "answer_method": (answer or {}).get("answer_method", ""),
            "predicted_fields": ((document or {}).get("analysis") or {}).get("fields", {}),
            "summary": ((document or {}).get("analysis") or {}).get("summary", ""),
            "models": ((document or {}).get("analysis") or {}).get("analysis_models", []),
            "ocr_pages": (document or {}).get("scanned_pages", []),
            "seconds": round(time.perf_counter() - started, 1),
            **scored,
        }
        status = "error" if error else f"fields {'ok' if scored['fields_correct'] else 'WRONG'}, answer {'ok' if scored['qa_correct'] else 'WRONG'}"
        progress(f"    {status}")
        write_outputs(cases, rows, signature)
    return write_outputs(cases, rows, signature)


def _rate(checks: list[dict[str, Any]]) -> float | None:
    return round(sum(check["correct"] for check in checks) / len(checks), 4) if checks else None


def summarize(cases: list[dict[str, Any]], rows: dict[str, Any]) -> dict[str, Any]:
    evaluated = [rows[case["id"]] for case in cases if case["id"] in rows]
    checks = [check for row in evaluated for check in row["checks"]]

    def group(key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for value in sorted({row["case"][key] for row in evaluated}):
            subset = [check for row in evaluated if row["case"][key] == value for check in row["checks"]]
            out[value] = {
                "cases": sum(1 for row in evaluated if row["case"][key] == value),
                "field_accuracy": _rate([check for check in subset if check["kind"] == "field"]),
                "qa_accuracy": _rate([check for check in subset if check["kind"] == "qa"]),
            }
        return out

    categories: dict[str, int] = {}
    for check in checks:
        if not check["correct"]:
            categories[check["error_category"]] = categories.get(check["error_category"], 0) + 1
    field_checks = [check for check in checks if check["kind"] == "field"]
    qa_checks = [check for check in checks if check["kind"] == "qa"]
    return {
        "total_cases": len(cases), "evaluated_cases": len(evaluated),
        "field_checks": len(field_checks), "field_correct": sum(check["correct"] for check in field_checks),
        "field_accuracy": _rate(field_checks),
        "qa_checks": len(qa_checks), "qa_correct": sum(check["correct"] for check in qa_checks),
        "qa_accuracy": _rate(qa_checks),
        "page_reference_accuracy": _rate([check for check in checks if check["kind"] == "page"]),
        "cases_fully_correct": sum(1 for row in evaluated if all(check["correct"] for check in row["checks"] if check["kind"] != "page")),
        "processing_errors": sum(1 for row in evaluated if row["error"]),
        "by_language": group("language"), "by_document_type": group("document_type"),
        "by_format": group("format"), "by_source": group("source"),
        "by_model": {", ".join(sorted({model for row in evaluated for model in row.get("models", [])})) or "not reported": {
            "cases": len(evaluated), "field_accuracy": _rate(field_checks), "qa_accuracy": _rate(qa_checks)}},
        "failure_categories": dict(sorted(categories.items(), key=lambda item: -item[1])),
    }


def check_rows(rows: dict[str, Any]) -> list[dict[str, Any]]:
    table = []
    for row in rows.values():
        case = row["case"]
        for check in row["checks"]:
            table.append({
                "case_id": case["id"], "document": case["document"], "document_type": case["document_type"],
                "language": case["language"], "format": case["format"], "check_type": check["kind"], "check": check["check"],
                "question": case["question"] if check["kind"] != "field" else "",
                "expected": scoring.display(check["expected"]), "predicted": scoring.display(check["predicted"]),
                "result": "correct" if check["correct"] else "incorrect", "error_category": check["error_category"],
                "source_pages": ", ".join(map(str, check["pages"])),
            })
    return table


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _table_cell(value: Any, limit: int = 160) -> str:
    text = scoring.display(value) if not isinstance(value, str) else value
    return " ".join(text.replace("|", "/").split())[:limit]


def dataset_overview(cases: list[dict[str, Any]]) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    for case in cases:
        entry = documents.setdefault(case["document"], {key: case[key] for key in ("document_type", "language", "format", "source")} | {"cases": 0})
        entry["cases"] += 1
    count = lambda key: dict(sorted(Counter(case[key] for case in cases).items(), key=lambda item: -item[1]))  # noqa: E731
    return {
        "test_cases": len(cases), "documents": documents,
        "field_checks": sum(len(case["expected_fields"]) for case in cases),
        "languages": count("language"), "document_types": count("document_type"),
        "formats": count("format"), "sources": count("source"),
    }


def observations(report: dict[str, Any]) -> list[str]:
    """Statements derived only from the measured results."""
    m, notes = report["metrics"], []
    if m["evaluated_cases"] < m["total_cases"]:
        notes.append(f"This is a partial run: {m['evaluated_cases']} of {m['total_cases']} test cases have results. All figures cover only the evaluated cases.")
    for key, label in (("by_format", "document format"), ("by_language", "language"), ("by_document_type", "document type")):
        groups = [(name, values) for name, values in m[key].items() if values["cases"] >= 2 and values["qa_accuracy"] is not None]
        if len(groups) >= 2:
            best = max(groups, key=lambda item: item[1]["qa_accuracy"])
            worst = min(groups, key=lambda item: item[1]["qa_accuracy"])
            if best[1]["qa_accuracy"] != worst[1]["qa_accuracy"]:
                notes.append(f"By {label}, QA accuracy was highest for {best[0]} ({_percent(best[1]['qa_accuracy'])}, {best[1]['cases']} cases) "
                             f"and lowest for {worst[0]} ({_percent(worst[1]['qa_accuracy'])}, {worst[1]['cases']} cases).")
    if m["failure_categories"]:
        name, count = next(iter(m["failure_categories"].items()))
        total = sum(m["failure_categories"].values())
        notes.append(f"The most common failure category was \"{name}\": {count} of {total} incorrect checks.")
    methods = Counter(row.get("answer_method") or "none" for row in report["cases"] if not row["error"])
    if methods:
        notes.append("Answers came from: " + ", ".join(f"{name} ({count})" for name, count in methods.most_common()) + ".")
    if m["processing_errors"]:
        failed_docs = sorted({row["case"]["document"] for row in report["cases"] if row["error"]})
        notes.append(f"{m['processing_errors']} case(s) failed with a processing error, on: {', '.join(failed_docs)}.")
    return notes


LIMITATIONS = [
    "Patram was not evaluated: it needs a GPU inference endpoint that was not available. These results measure the local pipeline with the model named above.",
    "Telugu document understanding is not measured: no public Unicode Telugu government document was found, so the dataset has no Telugu document. Tamil (3 cases) and Malayalam (2 cases) use real documents, but their per-language figures rest on very few cases.",
    "Scoring is automatic and lexical: a correct answer phrased in a way the verified alternatives do not cover is counted as incorrect. Dates, amounts and numbers are compared exactly by design.",
    "Summaries and translations are not scored. The project metrics are field-extraction accuracy and QA accuracy.",
    "Each case was run once on a CPU-only laptop with temperature 0.1, so results can vary slightly between runs.",
]


def markdown_report(report: dict[str, Any]) -> str:
    m, d = report["metrics"], report["dataset"]
    lines = [
        "# PatraAI — CS3235 Project 3 evaluation report", "",
        f"*Indian Document Understanding Tool. Generated {report['generated_at'].replace('T', ' ').replace('+00:00', '')} UTC from an actual evaluation run "
        f"with the {report['pipeline']['model']} model ({report['pipeline']['analysis_service']}).*", "",
        "## 1. Results", "",
        "| Metric | Result |", "| --- | --- |",
        f"| Test cases evaluated | {m['evaluated_cases']} of {m['total_cases']} |",
        f"| **Field-extraction accuracy** | **{_percent(m['field_accuracy'])}** ({m['field_correct']} of {m['field_checks']} fields) |",
        f"| **QA accuracy** | **{_percent(m['qa_accuracy'])}** ({m['qa_correct']} of {m['qa_checks']} answers) |",
        f"| Answer page-reference accuracy | {_percent(m['page_reference_accuracy'])} |",
        f"| Test cases with every field and the answer correct | {m['cases_fully_correct']} |",
        f"| Processing errors | {m['processing_errors']} |", "",
        "## 2. Dataset", "",
        f"{d['test_cases']} test cases over {len(d['documents'])} documents, with {d['field_checks']} field checks and one question per case. "
        "Expected values were verified against the documents and were not produced by the model under evaluation. "
        "Sources and verification are described in `evaluation/dataset/README.md`.", "",
        "| Coverage | Test cases |", "| --- | --- |",
        "| Languages | " + ", ".join(f"{name} {count}" for name, count in d["languages"].items()) + " |",
        "| Document types | " + ", ".join(f"{name} {count}" for name, count in d["document_types"].items()) + " |",
        "| Formats | " + ", ".join(f"{name} {count}" for name, count in d["formats"].items()) + " |",
        "| Sources | " + ", ".join(f"{name} {count}" for name, count in d["sources"].items()) + " |", "",
        "| Document | Language | Type | Format | Source | Cases |", "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {name} | {info['language']} | {info['document_type']} | {info['format']} | {info['source']} | {info['cases']} |" for name, info in d["documents"].items()]
    lines.append("")
    for number, (title, key) in enumerate((("Results by document language", "by_language"), ("Results by document type", "by_document_type"),
                                           ("Results by document format", "by_format"), ("Results by document source", "by_source")), start=3):
        lines += [f"## {number}. {title}", "", "| Group | Cases | Field-extraction accuracy | QA accuracy |", "| --- | --- | --- | --- |"]
        lines += [f"| {name} | {values['cases']} | {_percent(values['field_accuracy'])} | {_percent(values['qa_accuracy'])} |" for name, values in m[key].items()]
        lines.append("")
    lines += ["## 7. Correct and incorrect predictions", "",
              "| Case | Language | Type | Format | Fields correct | Answer | Answer page |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["cases"]:
        case, checks = row["case"], row["checks"]
        fields = [check for check in checks if check["kind"] == "field"]
        answer = next(check for check in checks if check["kind"] == "qa")
        page = next(check for check in checks if check["kind"] == "page")
        lines.append(f"| {case['id']} | {case['language']} | {case['document_type']} | {case['format']} | "
                     f"{sum(check['correct'] for check in fields)}/{len(fields)} | {'correct' if answer['correct'] else 'incorrect'} | "
                     f"{'correct' if page['correct'] else 'incorrect'} |")
    lines += ["", "## 8. Failure categories", "", "| Category | Incorrect checks |", "| --- | --- |"]
    lines += [f"| {name} | {count} |" for name, count in m["failure_categories"].items()] or ["| none | 0 |"]
    lines += ["", "## 9. Examples: expected versus predicted", "", "| Case | Language | Check | Expected (verified) | Predicted | Category |", "| --- | --- | --- | --- | --- | --- |"]
    failures = [row for row in report["checks"] if row["result"] == "incorrect" and row["check_type"] != "page"]
    for row in failures[:30]:
        lines.append(f"| {row['case_id']} | {row['language']} | {row['check']} | {_table_cell(row['expected'])} | {_table_cell(row['predicted'])} | {row['error_category']} |")
    if not failures:
        lines.append("| — | — | — | — | — | no incorrect predictions |")
    lines += ["", "## 10. Observations", ""] + [f"- {note}" for note in observations(report)]
    lines += ["", "## 11. Limitations", ""] + [f"- {item}" for item in LIMITATIONS]
    lines += ["", "## 12. Method", "",
              "Each test case's document runs through the same pipeline as the PatraAI website: text extraction, OCR of scanned or "
              "unreadable pages, field extraction, summary and retrieval-based question answering. Predicted fields and answers are compared "
              "with the verified values using the rules in `evaluation/scoring.py`: differences in case, spacing, punctuation, and in how "
              "a date or rupee amount is written are accepted, while a different number, date, amount or reference is not. An abstention "
              "counts as incorrect. Answer page references are scored separately.", "",
              "Reproduce with `py -3.11 evaluation/run_evaluation.py`. Every check is listed in `evaluation-results.csv`, and "
              "`evaluation-results.json` contains all predictions.", ""]
    return "\n".join(lines)


def write_outputs(cases: list[dict[str, Any]], rows: dict[str, Any], signature: dict[str, Any]) -> dict[str, Any]:
    ordered = {case["id"]: rows[case["id"]] for case in cases if case["id"] in rows}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pipeline": signature,
        "dataset": dataset_overview(cases),
        "metrics": summarize(cases, ordered), "checks": check_rows(ordered), "cases": list(ordered.values()),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS / RESULTS_JSON, report)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(report["checks"][0]) if report["checks"] else ["case_id"])
    writer.writeheader()
    writer.writerows(report["checks"])
    (RESULTS / RESULTS_CSV).write_text(buffer.getvalue(), encoding="utf-8-sig")
    (RESULTS / REPORT_FILE).write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", action="store_true", help="ignore saved results and evaluate every case again")
    parser.add_argument("--report", action="store_true", help="only rebuild the report from saved results")
    parser.add_argument("--failed", action="store_true", help="re-run only the cases that failed in the saved results")
    args = parser.parse_args()
    if args.report:
        report = load_results()
        if not report:
            raise SystemExit("No saved results yet. Run without --report first.")
        rows = {row["case"]["id"]: row for row in report["cases"]}
        report = write_outputs(load_cases(), rows, report["pipeline"])
    else:
        report = run(rerun=args.rerun, progress=lambda message: print(message, flush=True), only=failed_case_ids() if args.failed else None)
    m = report["metrics"]
    print(f"\nEvaluated {m['evaluated_cases']}/{m['total_cases']} cases | field-extraction accuracy {_percent(m['field_accuracy'])} | QA accuracy {_percent(m['qa_accuracy'])}")
    print(f"Report: {RESULTS / REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
