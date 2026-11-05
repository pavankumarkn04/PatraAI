"""Multilingual question answering over one uploaded document.

Steps for every question:

1. Detect the question's language (native script or romanised wording).
2. Choose the answer language: an explicitly selected response language,
   otherwise the question's own language (never the document's).
3. Find the evidence: the whole document when it is short, otherwise the
   passages retrieved for the question and its other wordings.
4. Ask the model for a grounded answer in the target language.
5. Check the answer's language in code. An English answer that should be in
   an Indian language is translated with IndicTrans2; any other mismatch is
   rewritten once by the analysis model. If the answer still is not in the
   target language, an error is raised instead of showing a wrong-language
   answer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from document_processing import (
    QA_FULL_CONTEXT_CHARS,
    QA_FULL_CONTEXT_PAGES,
    _as_text,
    _char_budget,
    answer_document_question,
    number_tokens,
    question_in_english,
    rewrite_answer_language,
    strip_page_citations,
)
from language_detection import detect_question_language, is_in_language, resolve_answer_language
from translation import translate_segments, translate_structured_text, translation_service_description

logger = logging.getLogger("patraai")


class AnswerLanguageError(RuntimeError):
    """The answer could not be produced in the required language."""


def fits_whole(page_texts: list[str]) -> bool:
    """True when the Q&A model is given the whole document (no retrieval)."""
    total = sum(len(text.strip()) for text in page_texts)
    return total <= _char_budget(QA_FULL_CONTEXT_CHARS, "".join(page_texts[:3])) and len(page_texts) <= QA_FULL_CONTEXT_PAGES


def ensure_language(text: str, target_language: str) -> dict[str, Any]:
    """Return ``text`` in ``target_language``, converting it only when needed.

    Returns {"text", "method", "converted"}. Raises AnswerLanguageError when
    no route produces text in the target language.
    """
    if is_in_language(text, target_language):
        return {"text": text, "method": "", "converted": False}
    attempts: list[str] = []
    if target_language != "English" and is_in_language(text, "English"):
        try:
            translated = translate_structured_text(text, target_language)
            if is_in_language(translated["text"], target_language):
                service = translated.get("service") or "the translation service"
                return {"text": translated["text"], "method": f"translated from English by {service}", "converted": True}
            attempts.append(f"{translated.get('service') or 'translation'} did not return {target_language} text")
        except Exception as exc:  # noqa: BLE001 - reported to the user below
            attempts.append(f"translation failed: {exc}")
    # IndicTrans2 here translates from English only, so any other pair (for
    # example a Hindi answer that must be in Kannada) is rewritten by the model.
    try:
        rewritten = rewrite_answer_language(text, target_language)
        candidate = strip_page_citations(_as_text(rewritten.get("answer")))
        if candidate and is_in_language(candidate, target_language):
            return {"text": candidate, "method": f"rewritten in {target_language} by {rewritten.get('_model') or 'the analysis model'}", "converted": True}
        attempts.append(f"the analysis model did not return {target_language} text")
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"rewriting failed: {exc}")
    raise AnswerLanguageError(f"The answer could not be produced in {target_language} ({'; '.join(attempts)}). "
                              "Please ask again, or choose another response language.")


def _search_wordings(question: str, detected: dict[str, Any], document_language: str) -> tuple[list[str], str]:
    """Other wordings of the question for searching a long document.

    English documents are searched with an English translation of the
    question; Indian-language documents also with the question translated
    into the document's language by IndicTrans2.
    """
    wordings: list[str] = []
    method = ""
    english = question
    if detected["language"] != "English":
        english, method = question_in_english(question, romanised=detected.get("script") == "romanised")
        if english != question:
            wordings.append(english)
    have_english = detected["language"] == "English" or english != question
    indian_document = document_language not in {"English", "Unknown"}
    if indian_document and detected["language"] != document_language and have_english:
        if translation_service_description().get("is_indictrans2"):
            try:
                in_document_language = translate_segments([english], document_language)["texts"][0]
                if in_document_language:
                    wordings.append(in_document_language)
                    method = (method + "; " if method else "") + f"searched in {document_language} with IndicTrans2"
            except Exception as exc:  # noqa: BLE001 - searching still works with the other wordings
                logger.info("question translation into %s failed: %s", document_language, type(exc).__name__)
    return wordings, method


def answer_question(
    page_texts: list[str], question: str, *, response_choice: str, automatic: str,
    previous_language: str | None = None, document_language: str = "English", summary: str = "",
) -> dict[str, Any]:
    """Answer one question in the right language, with page references and timings."""
    started = time.perf_counter()
    detected = detect_question_language(question)
    target = resolve_answer_language(response_choice, automatic, detected, previous_language)
    timings: dict[str, float] = {}

    wordings: list[str] = []
    question_method = ""
    asked = question
    if not fits_whole(page_texts):
        step = time.perf_counter()
        wordings, question_method = _search_wordings(question, detected, document_language)
        timings["question translation for search"] = time.perf_counter() - step
    elif detected.get("script") == "romanised":
        # Measured: the small model misreads romanised Kannada and abstains, so
        # it also gets the question's validated English meaning.
        step = time.perf_counter()
        english, question_method = question_in_english(question, romanised=True)
        if english != question:
            wordings = [english]
        timings["question translation"] = time.perf_counter() - step
    if detected.get("script") == "romanised" and wordings:
        asked = f"{question}\n(Meaning of this romanised {detected['language']} question in English: {wordings[0]})"

    step = time.perf_counter()
    result = answer_document_question(page_texts, asked, summary=summary, target_language=target, extra_queries=wordings)
    timings["answer generation"] = time.perf_counter() - step
    model_answer = strip_page_citations(str(result.get("answer") or ""))
    if not model_answer:
        raise RuntimeError("The model returned an empty answer. Please ask again.")

    step = time.perf_counter()
    final = ensure_language(model_answer, target)
    if final["converted"]:
        timings["answer language conversion"] = time.perf_counter() - step
    warning = result.get("grounding_warning") or ""
    lost = number_tokens(model_answer) - number_tokens(final["text"])
    if final["converted"] and lost:
        warning = (warning + " " if warning else "") + f"Some numbers of the original answer were not kept in {target} ({', '.join(sorted(lost))}); check them against the document."
    timings["total"] = time.perf_counter() - started
    if final["method"]:
        language_method = final["method"]
    elif result.get("answer_method") == "source-text match":
        language_method = "quoted from the document text"
    else:
        language_method = f"written in {target} by {result.get('_model') or 'the analysis model'}"
    logger.info("question answered: question_language=%s target=%s method=%s converted=%s seconds=%.1f",
                detected["language"], target, result.get("answer_method"), final["converted"], timings["total"])
    return {
        "question": question.strip(),
        "question_language": detected["language"], "question_script": detected.get("script"),
        "answer_language": target, "answer": final["text"], "model_answer": model_answer,
        "language_method": language_method,
        "found": bool(result.get("found", True)), "answer_pages": result.get("answer_pages") or [],
        "answer_method": result.get("answer_method") or "", "question_method": question_method,
        "model": result.get("_model") or "", "backend": result.get("_backend") or "",
        "grounding_warning": warning, "timings": timings, "translations": {target: final["text"]},
    }


def answer_in_language(record: dict[str, Any], target_language: str) -> dict[str, Any]:
    """An existing answer shown in another language (cached in the record)."""
    translations = record.setdefault("translations", {record.get("answer_language", "English"): record["answer"]})
    if target_language not in translations:
        converted = ensure_language(record["answer"], target_language)
        translations[target_language] = converted["text"]
        record.setdefault("translation_methods", {})[target_language] = converted["method"]
    return {"text": translations[target_language],
            "method": record.get("translation_methods", {}).get(target_language) or record.get("language_method", "")}
