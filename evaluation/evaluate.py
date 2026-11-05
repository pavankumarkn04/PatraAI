"""Score field extraction and answer exact match from a UTF-8 JSONL results file."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def evaluate(path: Path) -> dict[str, Any]:
    # PowerShell's default Set-Content encoding can include a UTF-8 BOM.
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not cases:
        raise ValueError("The results file has no cases.")

    fields_total = fields_correct = qa_total = qa_correct = qa_hits = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id", "(no id)"))
        expected_fields = case.get("expected_fields") or {}
        predicted_fields = case.get("predicted_fields") or {}
        wrong_fields = []
        for key, expected in expected_fields.items():
            fields_total += 1
            actual = predicted_fields.get(key)
            if normalize(expected) == normalize(actual):
                fields_correct += 1
            else:
                wrong_fields.append({"field": key, "expected": expected, "predicted": actual})

        expected_answer = case.get("expected_answer")
        predicted_answer = case.get("predicted_answer")
        qa_wrong = False
        if expected_answer is not None:
            qa_total += 1
            expected_normalized = normalize(expected_answer)
            predicted_normalized = normalize(predicted_answer)
            exact = expected_normalized == predicted_normalized
            hit = bool(expected_normalized and expected_normalized in predicted_normalized)
            qa_wrong = not exact
            if not qa_wrong:
                qa_correct += 1
            if hit:
                qa_hits += 1
        if wrong_fields or qa_wrong:
            failures.append({
                "id": case_id,
                "failure_category": case.get("failure_category", ""),
                "field_errors": wrong_fields,
                "qa_error": ({"expected": expected_answer, "predicted": predicted_answer, "exact_match": exact, "answer_hit": hit} if qa_wrong else None),
            })

    return {
        "case_count": len(cases),
        "field_pairs": fields_total,
        "field_extraction_accuracy": fields_correct / fields_total if fields_total else None,
        "qa_pairs": qa_total,
        "qa_exact_match_accuracy": qa_correct / qa_total if qa_total else None,
        "qa_answer_hit_accuracy": qa_hits / qa_total if qa_total else None,
        "failures": failures,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python evaluation/evaluate.py results.jsonl")
    try:
        print(json.dumps(evaluate(Path(sys.argv[1])), ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc
