"""Document language detection and response-language output.

The analysis is produced once, in English, from the uploaded document. Other
response languages are produced by translating that structured result piece by
piece with IndicTrans2 (see translation.py), so changing the language never
re-reads or re-analyzes the document. Values that must stay exact (reference
numbers, dates, amounts) are not translated.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from language_detection import detect_language  # noqa: F401 - re-exported for callers
from translation import translate_segments

# Automatic: results follow the document's language and each answer follows
# the language of its own question.
AUTOMATIC = "Automatic"
RESPONSE_LANGUAGES = [AUTOMATIC, "English", "Kannada", "Hindi", "Tamil", "Telugu", "Malayalam"]

FIELD_LABELS = {
    "document_title": "Document title", "document_type": "Document type", "issuing_department": "Issuing authority",
    "date": "Date of issue", "reference_number": "Reference number", "amount": "Fees / amount", "deadline": "Deadline",
    "important_dates": "Important dates", "eligibility_criteria": "Eligibility", "required_documents": "Required documents",
    "application_procedure": "Application procedure", "important_instructions": "Important instructions",
    "exceptions_or_special_conditions": "Exceptions or special conditions",
}
UI_LABELS = {
    "overview": "Overview", "main_points": "Key points", "important_dates": "Important dates",
    "applies_to": "Who does this apply to?", "actions": "Important actions",
    "source_language": "Source language", "pages": "Number of pages",
    "not_found": "Not found in document",
}
# Values of these fields are kept exactly as extracted (codes, dates, money).
EXACT_FIELDS = {"reference_number", "date", "amount"}


def effective_language(choice: str, detected: dict[str, Any]) -> tuple[str, str]:
    """The response language to use, and a note when Automatic cannot follow the document."""
    if choice != AUTOMATIC:
        return choice, ""
    if detected.get("supported"):
        return detected["language"], ""
    return "English", f"The document's language ({detected.get('language', 'unknown')}) is not supported for output yet, so results are in English."


def _is_exact(value: str) -> bool:
    """Codes, dates and amounts: few letters relative to digits and symbols."""
    letters = sum(char.isalpha() for char in value)
    return bool(re.search(r"\d", value)) and letters <= max(6, len(value) // 3)


def localize(analysis: dict[str, Any], language: str, progress: Callable[[int, int], None] | None = None,
             extra_segments: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Translate the structured analysis into `language` (one batched call).

    Only the text a reader sees is sent: the summary parts, field values that
    are words (not codes, dates or amounts), labels and headings. Field keys,
    page references, dates of the summary and empty values are never sent.
    ``extra_segments`` (such as the suggested questions) are translated in
    the same batch and left in the translation cache for their caller.
    Returns {"summary", "main_points", "summary_dates", "applies_to",
    "actions", "fields", "additional_fields", "labels", "service",
    "is_indictrans2", "fallback_reason", "segments", "segments_cached"}.
    """
    fields = analysis.get("fields") or {}
    segments: list[str] = []
    slots: list[tuple] = []

    def add(slot: tuple, text: str) -> None:
        text = " ".join(str(text).split())
        if text:
            slots.append(slot)
            segments.append(text)

    add(("summary",), analysis.get("summary") or "")
    for index, point in enumerate(analysis.get("main_points") or []):
        add(("point", index), point)
    for index, item in enumerate(analysis.get("summary_dates") or []):
        add(("date_event", index), item.get("event") or "")
    add(("applies_to",), analysis.get("applies_to") or "")
    for index, action in enumerate(analysis.get("actions") or []):
        add(("action", index), action)
    for key, value in fields.items():
        if key in EXACT_FIELDS:
            continue
        for index, item in enumerate(value if isinstance(value, list) else [value]):
            if not _is_exact(str(item)):
                add(("field", key, index), item)
    for index, item in enumerate(analysis.get("additional_fields") or []):
        add(("extra_label", index), item["label"])
        if not _is_exact(item["value"]):
            add(("extra_value", index), item["value"])
    for key, label in {**FIELD_LABELS, **{f"ui:{key}": value for key, value in UI_LABELS.items()}}.items():
        add(("label", key), label)
    for index, text in enumerate(extra_segments):
        add(("extra", index), text)

    result = translate_segments(segments, language, progress_callback=progress)
    translated = dict(zip(slots, result["texts"]))
    out_fields: dict[str, Any] = {}
    for key, value in fields.items():
        items = value if isinstance(value, list) else [value]
        localized = [translated.get(("field", key, index), item) for index, item in enumerate(items)]
        out_fields[key] = localized if isinstance(value, list) else localized[0]
    labels = {key: translated.get(("label", key), label) for key, label in FIELD_LABELS.items()}
    labels.update({f"ui:{key}": translated.get(("label", f"ui:{key}"), label) for key, label in UI_LABELS.items()})
    return {
        "summary": translated.get(("summary",), analysis.get("summary") or ""),
        "main_points": [translated.get(("point", index), point) for index, point in enumerate(analysis.get("main_points") or [])],
        "summary_dates": [{**item, "event": translated.get(("date_event", index), item.get("event") or "")}
                          for index, item in enumerate(analysis.get("summary_dates") or [])],
        "applies_to": translated.get(("applies_to",), analysis.get("applies_to") or ""),
        "actions": [translated.get(("action", index), action) for index, action in enumerate(analysis.get("actions") or [])],
        "segments": len(segments), "segments_cached": result.get("cached", 0),
        "fields": out_fields,
        "additional_fields": [
            {**item, "label": translated.get(("extra_label", index), item["label"]), "value": translated.get(("extra_value", index), item["value"])}
            for index, item in enumerate(analysis.get("additional_fields") or [])
        ],
        "labels": labels, "service": result["service"], "is_indictrans2": result["is_indictrans2"],
        "fallback_reason": result.get("fallback_reason", ""),
    }


def english_labels() -> dict[str, str]:
    return {**FIELD_LABELS, **{f"ui:{key}": value for key, value in UI_LABELS.items()}}
