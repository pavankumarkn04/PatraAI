"""PatraAI — Indian document understanding in Streamlit.

Run with: Run Indian App.bat (or ``py -3.11 run_app.py``).

Model and service settings are read from a project-local .env file; see
.env.example and README.md. They are never shown in the interface.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
import hashlib
import html
import io
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    if exc.name == "streamlit":
        if __name__ == "__main__":
            raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).with_name("run_app.py"))]))
        raise SystemExit("Streamlit is missing. Run `Run Indian App.bat` from this folder to install packages and open the website.") from None
    raise
from PIL import Image


def load_local_env() -> None:
    """Load simple KEY=VALUE entries from a project-local .env without a dependency."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key.replace("_", "").isalnum():
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_local_env()

# Load project-local model configuration before importing helpers that read
# those values during module initialization.
from document_processing import (  # noqa: E402
    analysis_config_fingerprint,
    analysis_service_description,
    extract_docx,
    render_pdf_page,
    analyze_document_text,
    check_model_service,
    extract_pdf_pages,
    local_ocr,
    merge_ocr_text,
    ocr_page_image,
    ocr_scanned_pages,
    prefetch_analysis,
    prefetch_question_context,
    strip_page_citations,
    summary_text,
    tesseract_available,
)
from document_profiles import document_category, field_label, field_rows, suggested_questions  # noqa: E402
from language_detection import detect_question_language, is_in_language, resolve_answer_language  # noqa: E402
from localization import AUTOMATIC, RESPONSE_LANGUAGES, detect_language, effective_language, english_labels, localize  # noqa: E402
from question_answering import answer_in_language, answer_question  # noqa: E402
from site_sections import render_privacy, render_terms  # noqa: E402
from translation import TRANSLATION_LANGUAGES, load_indictrans2, local_indictrans2_status, translate_segments, translation_service_description  # noqa: E402
from ui_theme import StageBoard, apply_theme, masthead  # noqa: E402

logging.basicConfig(level=os.getenv("PATRAAI_LOG_LEVEL", "WARNING").upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).with_name("run_app.py"))]))


# How many documents keep their extracted text and results in server memory.
MAX_DOCUMENTS_IN_MEMORY = 8


@st.cache_data(show_spinner=False, max_entries=MAX_DOCUMENTS_IN_MEMORY)
def cached_extract_pdf(document_hash: str, _file_bytes: bytes) -> dict[str, Any]:
    started = time.perf_counter()
    return {**extract_pdf_pages(_file_bytes), "extraction_seconds": time.perf_counter() - started}


@st.cache_data(show_spinner=False, max_entries=MAX_DOCUMENTS_IN_MEMORY)
def cached_extract_docx(document_hash: str, _file_bytes: bytes) -> dict[str, Any]:
    started = time.perf_counter()
    return {**extract_docx(_file_bytes), "extraction_seconds": time.perf_counter() - started}


@st.cache_data(show_spinner=False, max_entries=24)
def cached_render_page(document_hash: str, _pdf_bytes: bytes, page_number: int) -> Image.Image:
    return render_pdf_page(_pdf_bytes, page_number, scale=1.3)


@st.cache_data(show_spinner=False, max_entries=MAX_DOCUMENTS_IN_MEMORY)
def cached_read_image(document_hash: str, _file_bytes: bytes) -> tuple[Image.Image, str]:
    image = Image.open(io.BytesIO(_file_bytes)).convert("RGB")
    return image, local_ocr([image])


@st.cache_resource(show_spinner=False)
def preload_translation_model() -> bool:
    """Load IndicTrans2 in the background once per server, so the first
    Translate click does not wait for the model to load."""
    available, _reason = local_indictrans2_status()
    if available and not os.getenv("INDIC_TRANS2_URL", "").strip():
        threading.Thread(target=_preload_indictrans2, daemon=True).start()
    return available


def _preload_indictrans2() -> None:
    try:
        load_indictrans2()
    except RuntimeError:
        pass  # The reason is reported when the user translates.


@st.cache_resource(show_spinner=False)
def _document_store() -> dict[str, Any]:
    return {"lock": threading.Lock(), "documents": OrderedDict()}


def document_entry(document_hash: str) -> dict[str, Any]:
    """Per-document stage results, shared across reruns and kept in memory only.

    Each stage (OCR per page, analysis, answers per question, translations)
    is stored separately so a new question never repeats earlier stages.
    Entries are keyed by the document's SHA-256 hash and the analysis
    configuration, so a changed model or prompt version starts afresh.
    ``partial`` holds the model replies of an analysis that has not finished
    (it failed, or a rerun interrupted it), so the next attempt repeats only
    the missing calls.
    """
    store = _document_store()
    key = f"{document_hash}:{analysis_config_fingerprint()}"
    with store["lock"]:
        documents: OrderedDict[str, dict[str, Any]] = store["documents"]
        if key not in documents:
            documents[key] = {"ocr": {}, "analysis": None, "timings": {}}
        # Everything shown for a document lives in its own entry, so results of
        # one document can never appear for another.
        for name, factory in (("answers", dict), ("qa_order", list), ("localized", dict), ("localize_errors", dict),
                              ("translation_requested", set), ("timings", dict), ("partial", dict)):
            documents[key].setdefault(name, factory())
        documents.move_to_end(key)
        while len(documents) > MAX_DOCUMENTS_IN_MEMORY:
            documents.popitem(last=False)
        return documents[key]


def prefetch_once(entry: dict[str, Any], name: str, work: Any, *args: Any) -> None:
    """Start a background prompt read (see document_processing._prefetch_prompt)
    once per document, while the user looks at the preview or the results."""
    if entry.get(f"prefetch:{name}"):
        return
    entry[f"prefetch:{name}"] = True
    threading.Thread(target=work, args=args, daemon=True).start()


def question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def service_label(local: bool) -> str:
    return "on this computer" if local else "external service configured for this installation"


CORE_FIELDS = ("document_title", "document_type", "issuing_department", "date", "reference_number")
OPTIONAL_FIELDS = ("amount", "deadline", "important_dates", "eligibility_criteria", "required_documents",
                   "application_procedure", "important_instructions", "exceptions_or_special_conditions")
TRANSLATED_NAME_FIELDS = {"document_title", "issuing_department"}
SECTIONS = (("sec-overview", "Document Overview"), ("sec-extracted", "Extracted Information"), ("sec-summary", "Document Summary"),
            ("sec-translation", "Translation"), ("sec-questions", "Ask Questions"), ("sec-details", "Processing Details"))
HIGHLIGHT = re.compile(r"(₹\s?[\d,]+(?:\.\d+)?(?:/-)?|(?:Rs\.?|INR)\s?[\d,]+(?:\.\d+)?(?:/-)?|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}(?:-\d{2})?\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b)")


def section_title(number: int, anchor: str, title: str) -> None:
    st.markdown(f'<div class="section-title" id="{anchor}"><span class="num">{number}</span>{html.escape(title)}</div>', unsafe_allow_html=True)


def unit_count(document: dict[str, Any]) -> str:
    """'1 page', '7 pages' or '9 sections' (Word files)."""
    count = len(document["page_kinds"])
    return f"{count} {'section' if document.get('unit') == 'section' else 'page'}{'' if count == 1 else 's'}"


def ref_label(document: dict[str, Any], pages: list[int]) -> str:
    """'p. 1, 2' for PDFs and images; 'Section 2 · Eligibility' for Word files."""
    if not pages:
        return ""
    if document.get("unit") == "section":
        titles = document.get("unit_titles") or []
        return "; ".join(f"Section {page}" + (f" · {titles[page - 1]}" if 0 < page <= len(titles) else "") for page in pages)
    return "p. " + ", ".join(map(str, pages))


def value_html(value: Any, not_found: str) -> str:
    if value is None or value == [] or str(value).strip() == "":
        return f'<span class="missing">{html.escape(not_found)}</span>'
    if isinstance(value, list):
        return "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in value) + "</ul>"
    return html.escape(str(value))


def answer_html(text: str) -> str:
    """Escape, keep **bold** and bullet lines, and highlight dates and amounts."""
    lines, out, in_list = text.strip().splitlines(), [], False
    for line in lines:
        stripped = line.strip()
        bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)", stripped)
        body = html.escape(bullet.group(1) if bullet else stripped)
        body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
        body = HIGHLIGHT.sub(r"<mark>\1</mark>", body)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{body}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if body:
                out.append(f"<p>{body}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def localized(entry: dict[str, Any], language: str, extra_segments: list[str] | None = None) -> dict[str, Any] | None:
    """Analysis in `language`, translated once per document and language.

    ``extra_segments`` (the suggested questions) join the same IndicTrans2
    batch, so they need no second translation pass."""
    if language == "English":
        return None
    if language not in entry["localized"]:
        progress = st.progress(0.0, text=f"Preparing results in {language} with IndicTrans2...")
        started = time.perf_counter()
        try:
            result = localize(entry["analysis"], language, extra_segments=extra_segments or (),
                              progress=lambda done, total: progress.progress(done / max(total, 1), text=f"Translating results into {language}: part {done} of {total}"))
            result["seconds"] = time.perf_counter() - started
            entry["localized"][language] = result
            entry["timings"][f"translation ({language})"] = result["seconds"]
            entry["localize_errors"].pop(language, None)
        except Exception as exc:
            entry["localize_errors"][language] = str(exc)
        progress.empty()
    if entry["localize_errors"].get(language):
        st.error(f"Could not produce the results in {language}: {entry['localize_errors'][language]}. Showing them in English.")
        return None
    return entry["localized"].get(language)


def labels_for(loc: dict[str, Any] | None) -> dict[str, str]:
    return loc["labels"] if loc else english_labels()


def category_of(entry: dict[str, Any], document: dict[str, Any]) -> str:
    """Document category, decided once per analysis."""
    analysis = entry["analysis"]
    if "category" not in analysis:
        analysis["category"] = document_category(analysis, entry.get("pages") or document["page_texts"])
    return analysis["category"]


def render_overview(entry: dict[str, Any], document: dict[str, Any], loc: dict[str, Any] | None) -> None:
    analysis, labels = entry["analysis"], labels_for(loc)
    fields = loc["fields"] if loc else analysis.get("fields") or {}
    category = category_of(entry, document)
    detected = entry["language"]
    language_text = detected["language"] + (f" (also {', '.join(detected['others'])})" if detected.get("multilingual") else "")
    kinds = document["page_kinds"]
    count_text = unit_count(document) + (f" ({kinds.count('scanned')} scanned)" if kinds.count("scanned") else "")
    # The same validated field values as Extracted Information (no second source).
    items = (
        (labels["document_type"], fields.get("document_type")),
        (field_label(category, "document_title", labels["document_title"], bool(loc)), fields.get("document_title")),
        (field_label(category, "issuing_department", labels["issuing_department"], bool(loc)), fields.get("issuing_department")),
        (labels["ui:source_language"], language_text),
        (labels["ui:pages"], count_text),
    )
    body = "".join(f'<div class="overview-item"><div class="k">{html.escape(label)}</div><div class="v">{value_html(value, labels["ui:not_found"])}</div></div>' for label, value in items)
    st.markdown(f'<div class="overview-grid">{body}</div>', unsafe_allow_html=True)


def render_extracted(entry: dict[str, Any], document: dict[str, Any], loc: dict[str, Any] | None) -> None:
    analysis, labels = entry["analysis"], labels_for(loc)
    original = analysis.get("fields") or {}
    fields = loc["fields"] if loc else original
    sources = analysis.get("field_sources") or {}
    category = category_of(entry, document)
    rows = []
    for key, _always in field_rows(category, original, CORE_FIELDS + OPTIONAL_FIELDS):
        value = fields.get(key)
        extra = ""
        if loc and key in TRANSLATED_NAME_FIELDS and original.get(key) and str(original[key]) != str(value):
            extra = f'<div class="original">{html.escape(str(original[key]))}</div>'
        source = ref_label(document, sources.get(key) or []) if key in original else ""
        if key in original and not source:
            source = "not located"
        rows.append(f'<tr><th>{html.escape(field_label(category, key, labels[key], bool(loc)))}</th><td>{value_html(value, labels["ui:not_found"])}{extra}</td>'
                    f'<td class="pages">{html.escape(source)}</td></tr>')
    for item in (loc or analysis).get("additional_fields") or []:
        rows.append(f'<tr><th>{html.escape(item["label"])}</th><td>{html.escape(item["value"])}</td><td class="pages">{html.escape(ref_label(document, item.get("pages") or []))}</td></tr>')
    head = '<tr class="head"><th>Field</th><td>Value</td><td class="pages">Source</td></tr>'
    st.markdown(f'<div class="info-table-wrap"><table class="info-table">{head}{"".join(rows)}</table></div>', unsafe_allow_html=True)
    if not original:
        st.warning("The model returned no fields for this document. Use **Retry analysis** at the end of the results to try again.")
    removed = analysis.get("ungrounded_values") or []
    if removed:
        st.caption(f"{len(removed)} value(s) the model returned were left out because their numbers do not appear in the document. See Processing details.")
    if analysis.get("unverified_dates"):
        st.caption("A date the model returned was left out because that date is not printed in the document. See Processing details.")


def summary_parts(entry: dict[str, Any], loc: dict[str, Any] | None) -> list[tuple[str, list[str]]]:
    """(heading, items) of the concise summary after its overview; parts the
    document does not support are left out. The summary has its own key
    points and dates, so it does not repeat the Extracted Information table."""
    data, labels = loc or entry["analysis"], labels_for(loc)
    dates = [" — ".join(part for part in (item.get("date"), item.get("event")) if part) for item in data.get("summary_dates") or []]
    return [(heading, values) for heading, values in (
        (labels["ui:main_points"], [str(point) for point in data.get("main_points") or [] if str(point).strip()]),
        (labels["ui:important_dates"], [date for date in dates if date.strip()]),
        (labels["ui:applies_to"], [str(data["applies_to"])] if str(data.get("applies_to") or "").strip() else []),
        (labels["ui:actions"], [str(action) for action in data.get("actions") or [] if str(action).strip()]),
    ) if values]


def summary_plain_text(entry: dict[str, Any], document_name: str, loc: dict[str, Any] | None) -> str:
    labels = labels_for(loc)
    overview = (loc or entry["analysis"]).get("summary") or ""
    parts = [f"PatraAI summary: {document_name}", "", labels["ui:overview"], overview]
    for heading, values in summary_parts(entry, loc):
        parts += ["", heading] + [f"- {value}" for value in values]
    return "\n".join(parts).strip() + "\n"


def summary_html(entry: dict[str, Any], loc: dict[str, Any] | None) -> str:
    labels = labels_for(loc)
    overview = str((loc or entry["analysis"]).get("summary") or "").strip()
    body = f'<h4>{html.escape(labels["ui:overview"])}</h4><p>{html.escape(overview)}</p>' if overview else ""
    for heading, values in summary_parts(entry, loc):
        body += f"<h4>{html.escape(heading)}</h4><ul>" + "".join(f"<li>{html.escape(value)}</li>" for value in values) + "</ul>"
    return body


def render_summary(entry: dict[str, Any], document: dict[str, Any], loc: dict[str, Any] | None) -> None:
    analysis = entry["analysis"]
    body = summary_html(entry, loc)
    if analysis.get("summary_status") == "failed" or not body:
        st.error("The summary could not be generated: the model returned an empty summary, also when asked again. "
                 "The extracted information above is still valid. Use **Retry analysis** at the end of the results to try again.")
        return
    st.markdown(f'<div class="summary-text">{body}</div>', unsafe_allow_html=True)
    notes = []
    coverage = analysis.get("summary_coverage") or {}
    if coverage.get("sections", 1) > 1:
        notes.append(f"Long document: summarized in {coverage['sections']} parts covering all {document['page_count']} {'sections' if document.get('unit') == 'section' else 'pages'}; ask a question to look up any specific detail.")
    if analysis.get("summary_unverified_numbers"):
        notes.append("Check these numbers in the summary against the document; they were not found in its text: " + ", ".join(analysis["summary_unverified_numbers"]) + ".")
    for note in notes:
        st.caption(note)


def render_translation(entry: dict[str, Any], response_language: str) -> None:
    languages = [language for language in TRANSLATION_LANGUAGES if language != "English"]
    route = translation_service_description()
    picker, button = st.columns([3, 1], vertical_alignment="bottom")
    language = picker.selectbox("Translate the summary and extracted information into", languages, key="translation-language",
                                index=languages.index(response_language) if response_language in languages else 0)
    if button.button("Translate", type="primary", width="stretch", key="translate"):
        entry["translation_requested"].add(language)
    st.caption(f"Model: AI4Bharat {route['name']}." if route["is_indictrans2"] else f"IndicTrans2 is not available ({route['fallback_reason']}); the {route['name']} translates instead.")
    if language not in entry["translation_requested"] and language not in entry["localized"]:
        st.caption(f"Press Translate to see the summary and extracted information in {language}.")
        return
    loc = localized(entry, language)
    if not loc:
        return
    st.markdown(f'<div class="translation-head">{html.escape(language)} · {html.escape(loc["service"])}</div>', unsafe_allow_html=True)
    if not loc.get("is_indictrans2"):
        st.warning(f"IndicTrans2 was not used: {loc.get('fallback_reason') or 'not available'}.")
    side_by_side = st.toggle("Show the original English next to the translation", value=False, key="translation-compare")

    def block(source: dict[str, Any] | None) -> str:
        labels = labels_for(source)
        data = source or entry["analysis"]
        fields = data["fields"] if source else entry["analysis"].get("fields") or {}
        out = summary_html(entry, source)
        rows = "".join(f'<tr><th>{html.escape(labels[key])}</th><td>{value_html(fields.get(key), labels["ui:not_found"])}</td></tr>'
                       for key in CORE_FIELDS + OPTIONAL_FIELDS if (entry["analysis"].get("fields") or {}).get(key))
        rows += "".join(f'<tr><th>{html.escape(item["label"])}</th><td>{html.escape(item["value"])}</td></tr>' for item in data.get("additional_fields") or [])
        return out + (f'<table class="info-table compact">{rows}</table>' if rows else "")

    if side_by_side:
        left, right = st.columns(2)
        left.markdown(f'<div class="translation-card"><div class="lang">English (original)</div>{block(None)}</div>', unsafe_allow_html=True)
        right.markdown(f'<div class="translation-card"><div class="lang">{html.escape(language)}</div>{block(loc)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="translation-card">{block(loc)}</div>', unsafe_allow_html=True)
    st.caption("Machine translation. Reference numbers, dates and amounts are kept as printed; check names and figures against the original document.")


def ask_question(entry: dict[str, Any], question: str, choice: str) -> None:
    """Answer from the stored pages; never re-reads or re-analyzes the document.

    Answers are cached per question and answer language, so asking the same
    question again (or switching back to a language) needs no model call.
    """
    detected = detect_question_language(question)
    target = resolve_answer_language(choice, AUTOMATIC, detected, entry.get("last_answer_language"))
    key = (question_key(question), target)
    entry["qa_error"] = ""
    if key not in entry["answers"]:
        try:
            with st.spinner(f"Finding the answer in the document ({target})..."):
                record = answer_question(
                    entry["pages"], question, response_choice=choice, automatic=AUTOMATIC,
                    previous_language=entry.get("last_answer_language"), document_language=entry["language"]["language"],
                    summary=summary_text(entry["analysis"]))
        except Exception as exc:
            entry["qa_error"] = f"Question answering failed: {exc}"
            return
        entry["answers"][key] = record
        entry["timings"][f"question {len(entry['answers'])} ({record['answer_language']})"] = record["timings"]["total"]
    else:
        entry["answers"][key]["reused"] = entry["answers"][key].get("reused", 0) + 1
    entry["last_answer_language"] = target
    entry["qa_order"] = [item for item in entry["qa_order"] if item != key] + [key]


def answer_meta(document: dict[str, Any], record: dict[str, Any], shown_language: str, method: str) -> str:
    """'Question: Kannada · Answer: Kannada · Source: p. 1 · 4.2 s'."""
    parts = [f"Question: {record.get('question_language', 'English')}", f"Answer: {shown_language}"]
    if not record.get("found", True):
        parts.append("Not found in the uploaded document")
    elif record.get("answer_method") == "document summary":
        parts.append("From the document summary")
    elif record.get("answer_pages"):
        parts.append("Source: " + ref_label(document, record["answer_pages"]))
    else:
        parts.append("No source page could be identified")
    if method and ("translated" in method or "rewritten" in method):
        parts.append(method)
    if record.get("timings", {}).get("total") is not None:
        parts.append(f"{record['timings']['total']:.1f} s")
    return " · ".join(parts)


def localized_suggestions(suggestions: list[str], language: str) -> list[str]:
    """Suggested questions in the results language, so a click asks (and is
    answered) in that language. Kept in English if translation fails."""
    if language == "English":
        return suggestions
    try:
        texts = translate_segments(suggestions, language)["texts"]
    except Exception:  # noqa: BLE001 - English suggestions still work
        return suggestions
    return [text if text and is_in_language(text, language) else original for text, original in zip(texts, suggestions)]


def render_questions(entry: dict[str, Any], document: dict[str, Any], choice: str, results_language: str) -> None:
    category = category_of(entry, document)
    suggestions = localized_suggestions(suggested_questions(category, entry["analysis"].get("fields") or {}), results_language)
    st.markdown('<div class="qa-hint">Ask in English, Kannada, Hindi, Tamil, Telugu or Malayalam (native script or romanised). '
                + ("Each answer is written in the language of its question." if choice == AUTOMATIC else f"Answers are written in {html.escape(choice)}.")
                + "</div>", unsafe_allow_html=True)
    columns = st.columns(3)
    for index, suggestion in enumerate(suggestions):
        if columns[index % 3].button(suggestion, key=f"suggest-{index}", width="stretch"):
            ask_question(entry, suggestion, choice)
    with st.form("ask-form", clear_on_submit=True, border=False):
        ask_column, button_column = st.columns([5, 1], vertical_alignment="bottom")
        question = ask_column.text_input("Your question (English or an Indian language)", placeholder="ಉದಾ: ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ? · Who issued this document?")
        if button_column.form_submit_button("Ask", type="primary", width="stretch") and question.strip():
            ask_question(entry, question, choice)
    if entry.get("qa_error"):
        st.error(entry["qa_error"])
    for key in reversed(entry["qa_order"]):
        record = entry["answers"][key]
        shown_language = choice if choice != AUTOMATIC else record["answer_language"]
        try:
            if shown_language == record["answer_language"]:
                shown = {"text": record["answer"], "method": record.get("language_method", "")}
            else:
                with st.spinner(f"Showing this answer in {shown_language}..."):
                    shown = answer_in_language(record, shown_language)
        except Exception as exc:
            # Never fall back silently to an answer in the wrong language.
            st.error(f"Could not show the answer to “{record['question']}” in {shown_language}: {exc}")
            continue
        text = strip_page_citations(shown["text"])
        not_found = not record.get("found", True)
        card, copy = st.columns([12, 1], vertical_alignment="top")
        card.markdown(
            f'<div class="answer-card{" not-found" if not_found else ""}"><div class="q"><span class="q-label">Q</span>{html.escape(record["question"])}</div>'
            f'<div class="a">{answer_html(text)}</div><div class="meta">{html.escape(answer_meta(document, record, shown_language, shown["method"]))}</div></div>',
            unsafe_allow_html=True)
        with copy.popover("", icon=":material/content_copy:", help="Copy answer"):
            st.code(text, language=None, wrap_lines=True)
        if record.get("grounding_warning"):
            card.warning(record["grounding_warning"])


def render_processing_details(entry: dict[str, Any], document: dict[str, Any], language: str) -> None:
    analysis = entry["analysis"]
    with st.expander("Show processing details", icon=":material/info:"):
        kinds = document["page_kinds"]
        if document.get("unit") == "section":
            st.write(f"Word document: {len(kinds)} sections read from its headings, paragraphs, lists and {document.get('tables', 0)} table(s). References point to sections.")
        else:
            st.write(f"Pages: {len(kinds)} read · {kinds.count('text')} with selectable text · {kinds.count('scanned')} scanned or unreadable (OCR) · {kinds.count('blank')} blank.")
        if entry["ocr"]:
            methods = sorted({str(item.get("method")) for item in entry["ocr"].values()})
            st.write(f"OCR: page(s) {', '.join(map(str, sorted(entry['ocr'])))} read by {', '.join(methods)}.")
        detected = entry["language"]
        st.write("Detected document language: " + ", ".join(f"{name} {share:.0%}" for name, share in list(detected["shares"].items())[:3])
                 + f". Results shown in {language}. Document category: {category_of(entry, document)}.")
        st.markdown(model_status_table(entry), unsafe_allow_html=True)
        if analysis.get("fields_retried"):
            st.write("The combined analysis returned no fields, so a second, dedicated field-extraction request was made.")
        if analysis.get("summary_retried"):
            st.write("The combined analysis returned no summary, so a second, dedicated summary request was made.")
        removed = analysis.get("ungrounded_values") or []
        if removed:
            st.write("Left out because their numbers do not appear in the document: " + "; ".join(f"{item['field']}: {item['value']}" for item in removed) + ".")
        if analysis.get("unverified_dates"):
            st.write("Left out because the date is not printed in the document: " + ", ".join(key.replace("_", " ") for key in analysis["unverified_dates"]) + ".")
        for key in entry["qa_order"]:
            record = entry["answers"][key]
            steps = " · ".join(f"{name} {seconds:.1f}s" for name, seconds in record.get("timings", {}).items() if name != "total")
            reused = f" · shown again {record['reused']}× from cache" if record.get("reused") else ""
            st.write(f"Q: “{record['question']}” — question {record.get('question_language')} ({record.get('question_script')}), answer {record['answer_language']}; "
                     f"{record.get('answer_method') or 'model'}; {record.get('language_method')}; {record.get('question_method') or 'no question translation needed'}; {steps}{reused}.")
        if analysis.get("summary_status") == "failed":
            st.write("Summary: failed. The model returned an empty summary, also when asked a second time; nothing is shown as a summary.")
        elif analysis.get("summary_status") == "retried" and not analysis.get("summary_retried"):
            st.write("Summary: the first merged summary was empty, so it was requested once more.")
        if analysis.get("resumed_from"):
            st.write("Resumed an unfinished analysis: replies already received (" + ", ".join(analysis["resumed_from"]) + ") were reused, not requested again.")
        calls = analysis.get("model_calls") or []
        if calls:
            st.markdown(model_calls_table(calls), unsafe_allow_html=True)
        timings = entry.get("timings") or {}
        extraction = document.get("extraction_seconds")
        lines = []
        if extraction is not None:
            lines.append(f"text extraction {extraction:.2f}s (no model; cached for later reruns)")
        for name, seconds in timings.items():
            if name.startswith("translation (") or name.startswith("question "):
                continue  # listed with their details below and above
            kind = "no model" if "no model" in name else "fresh model inference"
            lines.append(f"{name.replace(' (no model)', '')} {seconds:.1f}s ({kind})")
        if lines:
            st.write("Measured time: " + " · ".join(lines) + ".")
        for name, data in entry["localized"].items():
            if data.get("seconds") is not None:
                cached = data.get("segments_cached", 0)
                st.write(f"Translation into {name}: {data['seconds']:.1f}s, {data.get('service')}, {data.get('segments', 0)} text segments"
                         + (f" ({cached} reused from the translation cache)" if cached else "") + "; computed once, later views reuse it.")
        st.caption("Changing the response language or asking another question reuses the extracted text and analysis; only the new answer or translation is computed.")


CALL_LABELS = {"document": "Information extraction + summary (one call)", "fields": "Field extraction", "section": "Section summary",
               "reduce": "Combining section summaries", "summary": "Summary (second request)", "ocr": "OCR"}


def model_calls_table(calls: list[dict[str, Any]]) -> str:
    """Every model call of the analysis with its measured time and token counts."""
    rows = []
    for call in calls:
        tokens = " · ".join(f"{label} {call[key]:,} tokens" for key, label in (("prompt_eval_count", "read"), ("eval_count", "written")) if isinstance(call.get(key), int))
        seconds = f"{call['seconds']:.1f} s" if isinstance(call.get("seconds"), (int, float)) else "time not measured"
        rows.append(f"<tr><th>{html.escape(CALL_LABELS.get(call.get('mode'), str(call.get('mode'))))}</th>"
                    f"<td>{html.escape(seconds)}{(' · ' + html.escape(tokens)) if tokens else ''} · {html.escape(str(call.get('model') or ''))}</td></tr>")
    return f'<table class="info-table compact"><tr class="head"><th>Model call</th><td>Measured</td></tr>{"".join(rows)}</table>'


def model_status_table(entry: dict[str, Any]) -> str:
    """Which models are configured, which actually ran, and which fallbacks were used."""
    analysis = entry["analysis"]
    service = analysis_service_description()
    backends = set(analysis.get("analysis_backends") or [])
    for record in entry["answers"].values():
        if record.get("backend"):
            backends.add(record["backend"])
    models = ", ".join(analysis.get("analysis_models") or []) or "not reported"
    rows = [("Document analysis and Q&A", f"{models} — {', '.join(sorted(backends)) or 'not reported'} ({service_label(service['local'])})")]
    if service.get("kind") == "endpoint":
        patram = "Configured and active" if "BharatGen Patram" in backends else "Endpoint configured, but it did not report a Patram model, so it is not shown as Patram"
    else:
        patram = "Not configured: it needs a GPU inference endpoint (HF_CHAT_COMPLETIONS_URL). Analysis is done by the local model above, not by Patram."
    rows.append(("BharatGen Patram", patram))
    route = translation_service_description()
    used = [name for name, data in entry["localized"].items() if data.get("is_indictrans2")]
    used += [record["answer_language"] for record in entry["answers"].values() if "IndicTrans2" in str(record.get("language_method"))]
    if route["is_indictrans2"]:
        status = "loaded and active" if local_indictrans2_status()[1] == "loaded" or os.getenv("INDIC_TRANS2_URL", "").strip() else "available (loads on first use)"
        indictrans = f"{route['name']}: {status}" + (f"; used for {', '.join(sorted(set(used)))}" if used else "; not used yet in this document")
    else:
        indictrans = f"Not available ({route.get('fallback_reason')}); translations use {route['name']} instead"
    rows.append(("AI4Bharat IndicTrans2 (English → Indian languages)", indictrans))
    if service.get("kind") == "ollama":
        fallback = os.getenv("OLLAMA_FALLBACK_MODEL", "gemma4:e4b")
        rows.append(("Fallback model", f"{fallback}: " + ("used for at least one request (see above)" if "Local Ollama fallback" in backends else "configured, not used")))
    body = "".join(f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>" for name, value in rows)
    return f'<table class="info-table compact">{body}</table>'


PDF_CSS = (
    "* {font-family: nui, sans-serif;} body {font-size: 10.5px; color: #20251f;} "
    "h1 {font-size: 17px; color: #28694f; margin: 0 0 4px 0;} h2 {font-size: 13.5px; color: #28694f; margin: 14px 0 5px 0;} "
    "h3 {font-size: 15px; color: #28694f;} h4 {font-size: 11px; color: #1f4f3b; margin: 8px 0 3px 0;} "
    "p {margin: 0 0 4px 0;} ul {margin: 0 0 4px 0;} li {margin: 0 0 2px 0;} .muted {color: #626960; font-size: 9px;} "
    # MuPDF lays out HTML tables poorly (overlapping cells), so fields are blocks.
    ".k {font-weight: bold; color: #1f2a22; margin: 7px 0 1px 0;} .src {font-weight: normal; color: #3f6b55; font-size: 9px;} "
    ".missing {color: #5f655c; font-style: italic;} .q {font-weight: bold; margin: 8px 0 2px 0;}"
)


def _pdf_field(label: str, value_markup: str, source: str = "") -> str:
    source_markup = f" <span class='src'>· {html.escape(source)}</span>" if source else ""
    return f"<p class='k'>{html.escape(label)}{source_markup}</p><div>{value_markup}</div>"


def html_pdf(body: str) -> bytes:
    """An A4 PDF of `body` (escaped HTML), as many pages as it needs. Nirmala UI
    renders the Indian scripts when it is available."""
    import fitz

    folder = indic_font_folder()
    css = ("@font-face {font-family: nui; src: url(NirmalaUI.ttf);} " if folder else "") + PDF_CSS
    story = fitz.Story(html=f"<body>{body}</body>", user_css=css, archive=fitz.Archive(folder) if folder else None)
    buffer = io.BytesIO()
    writer = fitz.DocumentWriter(buffer)
    page_rect = fitz.paper_rect("a4")
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(page_rect + (45, 45, -45, -45))
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


def summary_pdf(text: str) -> bytes:
    """A PDF of the summary text (its first line is the title)."""
    lines = text.splitlines()
    return html_pdf("".join(f"<h3>{html.escape(line)}</h3>" if index == 0 else f"<p>{html.escape(line) or '&nbsp;'}</p>"
                            for index, line in enumerate(lines)))


def results_pdf(entry: dict[str, Any], document: dict[str, Any], file_name: str, loc: dict[str, Any] | None,
                shown_language: str, choice: str) -> bytes:
    """All results as one PDF, in the language shown on the page: overview,
    extracted information with sources, summary, translations, and the
    questions and answers with their sources."""
    analysis, labels = entry["analysis"], labels_for(loc)
    fields = loc["fields"] if loc else analysis.get("fields") or {}
    sources = analysis.get("field_sources") or {}
    category = category_of(entry, document)
    detected = entry["language"]
    esc = html.escape
    parts = [f"<h1>PatraAI results</h1><p class='muted'>{esc(file_name)} · detected document language: {esc(detected['language'])} · "
             f"results in {esc(shown_language)} · {time.strftime('%d %b %Y, %H:%M')}</p>"]
    overview_rows = [(labels["document_type"], fields.get("document_type")),
                     (field_label(category, "document_title", labels["document_title"], bool(loc)), fields.get("document_title")),
                     (field_label(category, "issuing_department", labels["issuing_department"], bool(loc)), fields.get("issuing_department")),
                     (labels["ui:source_language"], detected["language"]), (labels["ui:pages"], unit_count(document))]
    parts.append("<h2>Document Overview</h2>" + "".join(_pdf_field(label, value_html(value, labels["ui:not_found"])) for label, value in overview_rows))
    rows = [_pdf_field(field_label(category, key, labels[key], bool(loc)), value_html(fields.get(key), labels["ui:not_found"]),
                       ref_label(document, sources.get(key) or []))
            for key, _always in field_rows(category, analysis.get("fields") or {}, CORE_FIELDS + OPTIONAL_FIELDS)]
    rows += [_pdf_field(item["label"], esc(item["value"]), ref_label(document, item.get("pages") or []))
             for item in (loc or analysis).get("additional_fields") or []]
    parts.append("<h2>Extracted Information</h2>" + "".join(rows))
    summary = summary_html(entry, loc)
    parts.append("<h2>Document Summary</h2>" + (summary or f"<p>{esc(labels['ui:not_found'])}</p>"))
    for language, data in entry["localized"].items():
        if language != shown_language:
            parts.append(f"<h2>Translation: {esc(language)}</h2><p class='muted'>{esc(data.get('service') or '')}</p>" + summary_html(entry, data))
    if entry["qa_order"]:
        parts.append("<h2>Questions and Answers</h2>")
        for key in entry["qa_order"]:
            record = entry["answers"][key]
            target = choice if choice != AUTOMATIC else record["answer_language"]
            text = strip_page_citations((record.get("translations") or {}).get(target) or record["answer"])
            parts.append(f"<p class='q'>Q: {esc(record['question'])}</p>{answer_html(text)}"
                         f"<p class='muted'>{esc(answer_meta(document, record, target, ''))}</p>")
    parts.append("<p class='muted'>AI output may contain errors. Verify names, dates and amounts against the original document.</p>")
    return html_pdf("".join(parts))


@st.cache_resource(show_spinner=False)
def indic_font_folder() -> str:
    """Folder with Windows' Nirmala UI as a standalone TTF (Indian scripts), or ''."""
    try:
        import tempfile

        from fontTools.ttLib import TTCollection

        folder = tempfile.mkdtemp(prefix="patraai-font-")
        TTCollection("C:/Windows/Fonts/Nirmala.ttc").fonts[0].save(str(Path(folder) / "NirmalaUI.ttf"))
        return folder
    except Exception:
        return ""


def extracted_csv(entry: dict[str, Any], document: dict[str, Any], loc: dict[str, Any] | None) -> str:
    labels = labels_for(loc)
    analysis = entry["analysis"]
    fields = loc["fields"] if loc else analysis.get("fields") or {}
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Field", "Value", "Source"])
    for key in CORE_FIELDS + OPTIONAL_FIELDS:
        value = fields.get(key)
        if value or key in CORE_FIELDS:
            writer.writerow([labels[key], "; ".join(map(str, value)) if isinstance(value, list) else (value or labels["ui:not_found"]),
                             ref_label(document, (analysis.get("field_sources") or {}).get(key) or [])])
    for item in (loc or analysis).get("additional_fields") or []:
        writer.writerow([item["label"], item["value"], ref_label(document, item.get("pages") or [])])
    return "\ufeff" + buffer.getvalue()


def render_results(entry: dict[str, Any], document: dict[str, Any], file_name: str, pdf_bytes: bytes | None, image: Image.Image | None) -> None:
    analysis = entry["analysis"]
    if "language" not in entry:
        entry["language"] = detect_language(entry.get("pages") or document["page_texts"])
    nav = "".join(f'<a href="#{anchor}">{html.escape(title)}</a>' for anchor, title in SECTIONS)
    complete = analysis.get("summary_status") != "failed" and analysis.get("fields_status") != "failed"
    st.markdown(f'<div id="patra-results" class="results-head"><h2>Results</h2><span class="tag">{"ANALYSIS READY" if complete else "PARTIAL RESULTS"}</span></div>'
                f'<nav class="results-nav">{nav}</nav>', unsafe_allow_html=True)
    if st.session_state.pop("scroll_to_results", False):
        # Fixed script (no user or model content): bring the new results into view.
        st.iframe("<script>const el = window.parent.document.getElementById('patra-results'); if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }</script>", height=1)
    detected = entry["language"]
    language_column, note_column = st.columns([2, 3], vertical_alignment="bottom")
    choice = language_column.selectbox("Response language", RESPONSE_LANGUAGES, key="response-language",
                                       help="Overview, extracted information, summary and answers are shown in this language. "
                                            "Automatic: results follow the document's language, and each answer follows the language of its question.")
    language, note = effective_language(choice, detected)
    note_column.markdown(f'<div class="results-sub">Detected document language: <b>{html.escape(detected["language"])}</b>'
                         + (f" (also {html.escape(', '.join(detected['others']))})" if detected.get("multilingual") else "")
                         + f" · showing results in <b>{html.escape(language)}</b>.</div>", unsafe_allow_html=True)
    if note:
        st.caption(note)
    loc = localized(entry, language, suggested_questions(category_of(entry, document), analysis.get("fields") or {}))
    shown_language = language if loc or language == "English" else "English"

    section_title(1, "sec-overview", "Document Overview")
    render_overview(entry, document, loc)
    section_title(2, "sec-extracted", "Extracted Information")
    render_extracted(entry, document, loc)
    section_title(3, "sec-summary", "Document Summary")
    render_summary(entry, document, loc)
    summary_text = summary_plain_text(entry, file_name, loc)
    actions = st.columns([1, 1, 1, 1, 2])
    with actions[0].popover("Copy summary", icon=":material/content_copy:", width="stretch"):
        st.code(summary_text, language=None, wrap_lines=True)
    actions[1].download_button("Summary (TXT)", data=summary_text, file_name="patraai-summary.txt", mime="text/plain", on_click="ignore", width="stretch", icon=":material/description:")
    actions[2].download_button("Summary (PDF)", data=lambda text=summary_text: summary_pdf(text), file_name="patraai-summary.pdf", mime="application/pdf", on_click="ignore", width="stretch", icon=":material/picture_as_pdf:")
    actions[3].download_button("Fields (CSV)", data=extracted_csv(entry, document, loc), file_name="patraai-extracted-information.csv", mime="text/csv", on_click="ignore", width="stretch", icon=":material/table:")
    section_title(4, "sec-translation", "Translation")
    render_translation(entry, shown_language)
    section_title(5, "sec-questions", "Ask Questions")
    render_questions(entry, document, choice, shown_language)
    section_title(6, "sec-details", "Processing Details")
    render_processing_details(entry, document, shown_language)

    retry_column, download_column = st.columns(2)
    if retry_column.button("Retry analysis", width="stretch", key="retry-analysis", icon=":material/refresh:"):
        reset_document_results(entry)
        if run_analysis(entry, document, pdf_bytes, image):
            st.rerun()
    # Built only when clicked: the PDF of everything shown, in the shown language.
    download_column.download_button("Download all results (PDF)", data=lambda: results_pdf(entry, document, file_name, loc, shown_language, choice),
                                     file_name="patraai-results.pdf", mime="application/pdf", on_click="ignore", width="stretch",
                                     icon=":material/picture_as_pdf:")


def reset_document_results(entry: dict[str, Any]) -> None:
    """Forget the analysis and everything derived from it (the extracted text
    and OCR are kept: they do not depend on the model's analysis)."""
    entry.update(analysis=None, answers={}, qa_order=[], localized={}, localize_errors={}, partial={},
                 translation_requested=set(), qa_error="", analysis_error="", last_answer_language=None)


def run_analysis(entry: dict[str, Any], document: dict[str, Any], pdf_bytes: bytes | None, image: Image.Image | None) -> bool:
    """OCR scanned pages, then extract fields and summarize. Each stage is marked
    done only when it returned usable output; failures stay visible."""
    board = StageBoard(st.empty())
    early_fields = st.empty()
    current = {"stage": ""}
    entry["analysis_error"] = ""

    def show_fields(fields: dict[str, Any]) -> None:
        values = [(str(key).replace("_", " ").title(), "; ".join(map(str, value)) if isinstance(value, list) else str(value))
                  for key, value in (fields.get("fields") or {}).items()]
        if values:
            rows = "".join(f'<div class="field"><div class="field-label">{html.escape(label)}</div><div class="field-value">{html.escape(value)}</div></div>' for label, value in values)
            early_fields.markdown('<div class="panel-sub" style="margin:4px 0 0">Extracted so far (the summary is still being written):</div>' + rows, unsafe_allow_html=True)

    stage_started: dict[str, float] = {}
    stage_names = {"fields": "field extraction", "summary": "summary generation", "analysis": "information extraction + summary (one combined call)"}

    def report(stage: str, done: int, total: int) -> None:
        current["stage"] = stage
        board.update(stage, done, total)
        now = time.perf_counter()
        stage_started.setdefault(stage, now)
        if done >= total and stage in stage_names:
            entry["timings"][stage_names[stage]] = now - stage_started[stage]

    try:
        scanned = [number for number, kind in enumerate(document["page_kinds"], start=1) if kind == "scanned"]
        if [number for number in scanned if number not in entry["ocr"]]:
            if not tesseract_available():
                check_model_service()
            started = time.perf_counter()
            if pdf_bytes is not None:
                ocr_scanned_pages(pdf_bytes, scanned, cache=entry["ocr"], stage_callback=report)
            elif image is not None:
                report("ocr", 0, 1)
                entry["ocr"][1] = ocr_page_image(image, 1)
                report("ocr", 1, 1)
            entry["timings"]["OCR"] = time.perf_counter() - started
            empty_pages = [number for number in scanned if not str(entry["ocr"].get(number, {}).get("text") or "").strip()]
            if empty_pages:
                board.set("ocr", "failed" if len(empty_pages) == len(scanned) else "done",
                          f"no text read from page(s) {', '.join(map(str, empty_pages))}")
        entry["pages"] = merge_ocr_text(document["page_texts"], {number: entry["ocr"][number] for number in scanned if number in entry["ocr"]})
        started = time.perf_counter()
        entry["language"] = detect_language(entry["pages"])
        entry["timings"]["language detection (no model)"] = time.perf_counter() - started
        check_model_service()
        for name in stage_names.values():
            entry["timings"].pop(name, None)
        resumed = sorted(entry["partial"])
        started = time.perf_counter()
        # Replies that succeeded in an earlier, unfinished attempt are kept in
        # entry["partial"], so only the missing model calls run now.
        analysis = analyze_document_text(entry["pages"], stage_callback=report, on_fields=show_fields, partial=entry["partial"])
        early_fields.empty()
        entry["timings"]["analysis total"] = time.perf_counter() - started
        analysis["resumed_from"] = resumed
        analysis["category"] = document_category(analysis, entry["pages"])
    except Exception as exc:
        stage = current["stage"]
        if stage:
            board.fail(stage)
        entry["analysis_error"] = f"{StageBoard.LABELS.get(stage, 'Document analysis')} failed: {exc}"
        if entry["partial"]:
            entry["analysis_error"] += " The steps that already finished are kept; analyzing again repeats only the missing steps."
        st.error(entry["analysis_error"])
        return False
    entry["partial"] = {}
    field_count = len(analysis.get("fields") or {}) + len(analysis.get("additional_fields") or [])
    fields_text = f"{field_count} field(s) found" if field_count else "no fields returned"
    summary_ok = analysis.get("summary_status") != "failed"
    summary_note = {"retried": "summary written on a second request"}.get(analysis.get("summary_status"), "summary written")
    if analysis.get("analysis_path") == "combined":
        board.set("analysis", "done" if field_count and summary_ok else "failed",
                  f"{fields_text} · {summary_note if summary_ok else 'empty summary returned'}")
    else:
        board.set("fields", "done" if field_count else "failed", fields_text)
        board.set("summary", "done" if summary_ok else "failed", summary_note if summary_ok else "empty summary returned")
    entry["analysis"] = analysis
    st.session_state["scroll_to_results"] = True
    prefetch_once(entry, "questions", prefetch_question_context, entry["pages"])
    return True


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
IMAGE_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF")


def validate_upload(name: str, file_bytes: bytes) -> str:
    """'pdf', 'docx' or 'image' for a valid upload; ValueError with a clear reason otherwise."""
    if not file_bytes:
        raise ValueError("the file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("the file is larger than 15 MB")
    lower = name.lower()
    if lower.endswith(".pdf"):
        if b"%PDF" not in file_bytes[:1024]:
            raise ValueError("this is not a valid PDF file (it may be damaged or renamed)")
        return "pdf"
    if lower.endswith(".docx"):
        if not file_bytes.startswith(b"PK"):
            raise ValueError("this is not a valid Word .docx file (older .doc files are not supported; save it as .docx)")
        return "docx"
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        if not file_bytes.startswith(IMAGE_SIGNATURES):
            raise ValueError("this is not a valid PNG, JPG or WEBP image")
        return "image"
    raise ValueError("unsupported file type; upload a PDF, Word (.docx) or image file")


def read_upload(uploaded: Any) -> tuple[dict[str, Any], bytes | None, Image.Image | None, str]:
    file_bytes = uploaded.getvalue()
    kind = validate_upload(uploaded.name, file_bytes)
    document_hash = hashlib.sha256(file_bytes).hexdigest()
    if kind == "pdf":
        with st.spinner("Reading document..."):
            try:
                document = cached_extract_pdf(document_hash, file_bytes)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError("the PDF could not be opened; it may be damaged") from exc
        return document, file_bytes, document["preview"], document_hash
    if kind == "docx":
        with st.spinner("Reading Word document..."):
            try:
                document = cached_extract_docx(document_hash, file_bytes)
            except ValueError:
                raise
            except ImportError as exc:
                raise ValueError("Word support is not installed (python-docx); start PatraAI with Start PatraAI.bat to install it") from exc
            except Exception as exc:
                raise ValueError("the Word document could not be opened; it may be damaged") from exc
        return document, None, None, document_hash
    with st.spinner("Reading image..."):
        try:
            image, image_text = cached_read_image(document_hash, file_bytes)
        except Exception as exc:
            raise ValueError("the image could not be read; it may be damaged") from exc
    document = {"page_count": 1, "page_texts": [image_text], "page_kinds": ["text" if image_text.strip() else "scanned"], "unit": "page"}
    return document, None, image, document_hash


def home_page() -> None:
    """Upload and preview on the left, analysis on the right, results below."""
    left, right = st.columns([0.95, 1.05], gap="large")
    document: dict[str, Any] | None = None
    entry: dict[str, Any] | None = None
    pdf_bytes: bytes | None = None
    image: Image.Image | None = None
    file_name = ""
    with left:
        st.markdown('<div class="layout-label">01 | UPLOAD YOUR DOCUMENT</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="panel-title">Upload your document</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">PDF, Word (.docx) or image (JPG, PNG, WEBP), up to 15 MB.</div>', unsafe_allow_html=True)
            uploader_key = f"upload-{st.session_state.get('upload_generation', 0)}"
            uploaded = st.file_uploader("Choose a document", type=["pdf", "docx", "png", "jpg", "jpeg", "webp"], max_upload_size=15, key=uploader_key)
            if uploaded and st.button("Clear document", icon=":material/close:", key="clear-document"):
                st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1
                for key in ("response-language", "translation-language", "preview-page", "translation-compare"):
                    st.session_state.pop(key, None)
                st.rerun()
        if uploaded:
            file_name = uploaded.name
            try:
                document, pdf_bytes, image, document_hash = read_upload(uploaded)
                entry = document_entry(document_hash)
                if st.session_state.get("current_document") != document_hash:
                    # A different document: start its view from the top.
                    st.session_state["current_document"] = document_hash
                    for key in ("preview-page", "translation-compare"):
                        st.session_state.pop(key, None)
            except Exception as exc:
                st.error(f"Could not read this document: {exc}")
            if document is not None and entry is not None:
                if not entry["analysis"] and "scanned" not in document["page_kinds"]:
                    # The analysis prompt is known now (scanned pages would
                    # need OCR first), so the model reads it during the preview.
                    prefetch_once(entry, "analysis", prefetch_analysis, document["page_texts"])
                kinds = document["page_kinds"]
                total = len(kinds)
                if document.get("unit") == "section":
                    rows = [("Reading document", f"{total} sections · {document.get('tables', 0)} table(s)")]
                else:
                    rows = [("Reading document", f"{total}/{total} pages")]
                    if pdf_bytes is not None:
                        rows.append(("Text extraction", f"{kinds.count('text')}/{total} pages"))
                    scanned_count = kinds.count("scanned")
                    if scanned_count:
                        rows.append(("Scanned-page OCR", f"{len(entry['ocr'])}/{scanned_count} pages" + ("" if len(entry["ocr"]) >= scanned_count else " · runs on analysis")))
                    if kinds.count("blank"):
                        rows.append(("Blank pages", f"{kinds.count('blank')} · nothing to read"))
                items = "".join(f'<li class="done"><span class="label">{html.escape(label)}</span><span class="value">{html.escape(value)}</span></li>' for label, value in rows)
                st.markdown(f'<ul class="stages">{items}</ul>', unsafe_allow_html=True)
                if pdf_bytes is not None:
                    if total > 1:
                        page = st.number_input("Preview page", min_value=1, max_value=total, value=1, step=1, key="preview-page")
                    else:
                        page = 1
                    preview = image if page == 1 else cached_render_page(st.session_state["current_document"], pdf_bytes, int(page))
                    st.image(preview, width="stretch", caption=f"Document preview · page {page} of {total}")
                elif image is not None:
                    st.image(image, width="stretch", caption="Document preview")
                else:
                    outline = "".join(f"<li>{html.escape(title)}</li>" for title in document.get("unit_titles") or [])
                    st.markdown(f'<div class="summary-text"><h4>Document outline</h4><ol>{outline}</ol></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="paper-preview"><div class="empty-preview"><div class="empty-icon">PDF · DOCX · JPG</div>Your document preview will appear here<br><span style="font-size:11px">Upload a document to begin</span></div></div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="layout-label">02 | ANALYZE YOUR DOCUMENT</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="panel-title">Analyze your document</div>', unsafe_allow_html=True)
            if entry is not None and entry["analysis"]:
                st.markdown('<div class="panel-sub">Extract, summarise and understand your document with AI.</div>', unsafe_allow_html=True)
                problems = [part for part, key in (("the summary", "summary_status"), ("the extracted information", "fields_status")) if entry["analysis"].get(key) == "failed"]
                if problems:
                    st.warning(f"Analysis finished, but {' and '.join(problems)} could not be generated. See Results below, or use Retry analysis.")
                else:
                    st.success("Analysis complete — see Results below.")
            else:
                st.markdown('<div class="panel-sub">Extract, summarise and understand your document with AI.</div>', unsafe_allow_html=True)
                if st.button("Analyze document", type="primary", width="stretch", disabled=document is None):
                    # Rerun so this panel shows the finished state; the results
                    # are cached, so the rerun repeats no processing.
                    if run_analysis(entry, document, pdf_bytes, image):
                        st.rerun()
                elif entry is not None and entry.get("analysis_error"):
                    st.error(entry["analysis_error"])
                if document is None:
                    st.caption("Upload a document to begin.")

    if document is not None and entry is not None and entry["analysis"]:
        render_results(entry, document, file_name, pdf_bytes, image)


st.set_page_config(page_title="PatraAI | Indian document understanding", page_icon="📄", layout="wide", initial_sidebar_state="collapsed")
apply_theme()
preload_translation_model()
masthead()
st.markdown('<div class="mast-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">INDIAN DOCUMENT UNDERSTANDING</div><h1>Understand Indian Documents with AI.</h1><p>Upload a document to extract key information, get a quick summary and ask questions in your preferred language.</p></div>', unsafe_allow_html=True)

home_page()

st.markdown('<div class="site-footer"><div class="footer-brand"><span class="brandmark">P</span>PatraAI <span class="sub">| Indian Document Understanding</span></div>'
            '<div class="footer-disclaimer">AI output may contain errors. Verify names, dates and amounts against the original document.</div></div>', unsafe_allow_html=True)
privacy_column, terms_column = st.columns(2)
with privacy_column.expander("Privacy Policy", icon=":material/shield:"):
    render_privacy()
with terms_column.expander("Terms & Conditions", icon=":material/gavel:"):
    render_terms()
