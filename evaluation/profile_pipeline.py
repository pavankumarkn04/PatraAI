"""Developer profiling: time every stage of the PatraAI pipeline on real documents.

    py -3.11 evaluation/profile_pipeline.py --label before cbse-three-language-press-release-2026.pdf ...

For each document this runs the same stages as the website, in the same
order, and records the measured time of each: text extraction, OCR of scanned
pages, language detection, every model call of the analysis (with Ollama's own
prompt and output token counts), document categorisation, retrieval,
translation into Kannada (cold, then again from the cache) and question
answering for the document's verified benchmark questions plus one Kannada
question. Extracted fields and answers are scored against the verified cases
with evaluation/scoring.py.

Results are written to evaluation/results/profile-<label>.json. Nothing here
is used by the website.
"""

from __future__ import annotations

import argparse
import contextvars
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import load_project_env  # noqa: E402

load_project_env()

import requests  # noqa: E402

import document_processing as dp  # noqa: E402
from document_profiles import document_category  # noqa: E402
from evaluation import benchmark, scoring  # noqa: E402
from language_detection import detect_language  # noqa: E402
from localization import AUTOMATIC, localize  # noqa: E402
from question_answering import answer_question  # noqa: E402
import translation  # noqa: E402

KANNADA_QUESTION = "ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?"  # "Who issued this document?"
_MODE: contextvars.ContextVar[str] = contextvars.ContextVar("mode", default="")
CALLS: list[dict[str, Any]] = []


def _instrument() -> None:
    """Record every model call: its mode, wall time and Ollama token counts."""
    original_once = dp._request_analysis_once
    original_post = requests.Session.post

    def timed_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        mode = "reduce" if kwargs.get("reduce") else kwargs.get("mode", "document")
        token = _MODE.set(mode)
        started = time.perf_counter()
        try:
            return original_once(*args, **kwargs)
        finally:
            _MODE.reset(token)
            CALLS.append({"mode": mode, "seconds": round(time.perf_counter() - started, 2), "images": len(args[0]) if args else 0,
                          **getattr(_LAST, "usage", {})})
            _LAST.usage = {}

    def recording_post(self: requests.Session, url: str, *args: Any, **kwargs: Any) -> requests.Response:
        response = original_post(self, url, *args, **kwargs)
        if url.endswith("/api/chat") and response.ok:
            try:
                payload = response.json()
                _LAST.usage = {key: payload.get(key) for key in ("prompt_eval_count", "eval_count")}
                for key in ("prompt_eval_duration", "eval_duration", "load_duration"):
                    if payload.get(key) is not None:
                        _LAST.usage[key.replace("duration", "seconds")] = round(payload[key] / 1e9, 2)
            except ValueError:
                pass
        return response

    dp._request_analysis_once = timed_once
    requests.Session.post = recording_post


class _Last:
    usage: dict[str, Any] = {}


_LAST = _Last()


def _calls_since(start: int) -> list[dict[str, Any]]:
    return CALLS[start:]


def profile_document(name: str, cases: list[dict[str, Any]], languages: list[str]) -> dict[str, Any]:
    payload = (benchmark.DATASET / "documents" / name).read_bytes()
    stages: dict[str, Any] = {}
    started = time.perf_counter()
    extraction, image = benchmark.extract_document(name, payload)
    stages["text extraction"] = round(time.perf_counter() - started, 3)

    scanned = extraction["scanned_pages"]
    ocr: dict[int, dict[str, Any]] = {}
    mark = len(CALLS)
    started = time.perf_counter()
    if scanned and image is not None:
        ocr[1] = dp.ocr_page_image(image, 1)
    elif scanned:
        dp.ocr_scanned_pages(payload, scanned, cache=ocr)
    stages["ocr"] = round(time.perf_counter() - started, 2)
    ocr_calls = _calls_since(mark)
    pages = dp.merge_ocr_text(extraction["page_texts"], {page: ocr[page] for page in scanned if page in ocr})

    started = time.perf_counter()
    detected = detect_language(pages)
    stages["language detection"] = round(time.perf_counter() - started, 4)

    mark = len(CALLS)
    started = time.perf_counter()
    analysis = dp.analyze_document_text(pages)
    stages["analysis (fields + summary)"] = round(time.perf_counter() - started, 2)
    analysis_calls = _calls_since(mark)

    started = time.perf_counter()
    analysis["category"] = document_category(analysis, pages)
    stages["categorisation"] = round(time.perf_counter() - started, 4)

    started = time.perf_counter()
    dp.qa_context(pages, "What are the important dates and deadlines?")
    stages["retrieval (one question)"] = round(time.perf_counter() - started, 4)

    translations = {}
    for language in languages:
        mark = len(CALLS)
        started = time.perf_counter()
        try:
            localized = localize(analysis, language)
            error = ""
        except Exception as exc:  # noqa: BLE001 - recorded in the report
            localized, error = {}, f"{type(exc).__name__}: {exc}"
        cold = round(time.perf_counter() - started, 2)
        started = time.perf_counter()
        if not error:
            localize(analysis, language)
        warm = round(time.perf_counter() - started, 3)
        translations[language] = {"cold_seconds": cold, "cached_seconds": warm, "error": error,
                                  "service": localized.get("service"), "model_calls": len(_calls_since(mark)),
                                  "output": {key: value for key, value in localized.items() if key != "labels"}}

    document_language = detected["language"]
    questions = [(case["id"], case["question"], case) for case in cases] + [("kannada-who-issued", KANNADA_QUESTION, None)]
    answers = []
    for case_id, question, case in questions:
        mark = len(CALLS)
        started = time.perf_counter()
        try:
            record = answer_question(pages, question, response_choice=AUTOMATIC, automatic=AUTOMATIC,
                                     document_language=document_language, summary=dp.summary_text(analysis) if hasattr(dp, "summary_text") else str(analysis.get("summary") or ""))
            error = ""
        except Exception as exc:  # noqa: BLE001
            record, error = {}, f"{type(exc).__name__}: {exc}"
        row = {"id": case_id, "question": question, "seconds": round(time.perf_counter() - started, 2), "error": error,
               "answer": record.get("answer", ""), "answer_language": record.get("answer_language"),
               "answer_pages": record.get("answer_pages", []), "found": record.get("found"),
               "answer_method": record.get("answer_method"), "model_calls": _calls_since(mark)}
        if case:
            row["answer_correct"] = not error and scoring.answer_matches(case["expected_answer"], row["answer"])
            row["expected"] = case["expected_answer"]
        answers.append(row)

    field_checks = []
    for case in cases:
        for field, expected in case["expected_fields"].items():
            predicted = (analysis.get("fields") or {}).get(field)
            field_checks.append({"case": case["id"], "field": field, "expected": expected, "predicted": predicted,
                                 "correct": scoring.field_matches(expected, predicted)})
    return {
        "document": name, "pages": extraction["page_count"], "scanned_pages": scanned,
        "characters": sum(len(text) for text in pages), "detected_language": document_language,
        "stages": stages, "ocr_calls": ocr_calls, "analysis_calls": analysis_calls,
        "translations": translations, "answers": answers, "field_checks": field_checks,
        "analysis": {key: value for key, value in analysis.items() if key != "retrieval_pages"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="+", help="Dataset document file names")
    parser.add_argument("--label", required=True, help="Name of this run, e.g. before or after")
    parser.add_argument("--languages", default="Kannada", help="Comma-separated translation targets")
    args = parser.parse_args()
    _instrument()
    dp.check_model_service()
    cases = benchmark.load_cases()
    out_path = benchmark.RESULTS / f"profile-{args.label}.json"
    report: dict[str, Any] = {"label": args.label, "signature": benchmark.pipeline_signature(), "documents": []}
    available, reason = translation.local_indictrans2_status()
    started = time.perf_counter()
    if available and not translation.os.getenv("INDIC_TRANS2_URL", "").strip():
        translation.load_indictrans2()
    report["indictrans2_load_seconds"] = round(time.perf_counter() - started, 2)
    report["indictrans2_status"] = reason
    for index, name in enumerate(args.documents, start=1):
        print(f"[{index}/{len(args.documents)}] {name}", flush=True)
        started = time.perf_counter()
        try:
            result = profile_document(name, [case for case in cases if case["document"] == name], args.languages.split(","))
        except Exception as exc:  # noqa: BLE001 - keep profiling the other documents
            result = {"document": name, "error": f"{type(exc).__name__}: {exc}"}
        result["total_seconds"] = round(time.perf_counter() - started, 1)
        report["documents"].append(result)
        benchmark._write_json(out_path, report)
        stages = result.get("stages", {})
        print(f"    {result.get('error') or ''} total {result['total_seconds']}s · " + " · ".join(f"{key} {value}s" for key, value in stages.items()), flush=True)
        for call in result.get("analysis_calls", []):
            print(f"      {call['mode']:10s} {call['seconds']:7.1f}s  prompt {call.get('prompt_eval_count')} tok  output {call.get('eval_count')} tok", flush=True)
    print(f"Report: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
