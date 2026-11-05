"""Coverage for page classification, scanned-only OCR and staged analysis."""

from __future__ import annotations

import io
import os
import unittest
from unittest.mock import Mock, patch

import fitz
from PIL import Image, ImageDraw

import document_processing as processing


def _page_png(label: str) -> bytes:
    image = Image.new("RGB", (900, 400), "white")
    ImageDraw.Draw(image).text((40, 60), label, fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def make_pdf(kinds: str) -> bytes:
    """Build a PDF from a string of page kinds: t=text, s=scanned image, b=blank."""
    document = fitz.open()
    for number, kind in enumerate(kinds, start=1):
        page = document.new_page(width=1200, height=500)
        if kind == "t":
            page.insert_text((40, 60), f"Page {number} has a unique searchable fact. " * 5)
        elif kind == "s":
            page.insert_image(page.rect, stream=_page_png(f"Scanned page {number}"))
    payload = document.tobytes()
    document.close()
    return payload


def fake_model(images, text, question, **kwargs):
    mode = "reduce" if kwargs.get("reduce") else kwargs.get("mode", "document")
    if mode == "ocr":
        return {"extracted_text": f"OCR text for {text}", "_backend": "Local Ollama", "_model": "vision"}
    return {
        "document_type": "Notice", "language": "English", "summary": f"{mode} summary",
        "fields": {"document_title": "Sample notice"}, "field_sources": {}, "answer": "Found it. [Page 2]",
        "_backend": "Local Ollama", "_model": "text",
    }


class PageClassificationTests(unittest.TestCase):
    def test_extracts_text_from_every_page_without_model_calls(self) -> None:
        extraction = processing.extract_pdf_pages(make_pdf("t" * 11))
        self.assertEqual(extraction["page_count"], 11)
        self.assertEqual(len(extraction["page_texts"]), 11)
        self.assertEqual(extraction["text_pages"], list(range(1, 12)))
        self.assertEqual(extraction["scanned_pages"], [])
        self.assertIn("Page 11 has a unique searchable fact", extraction["page_texts"][-1])
        self.assertGreater(extraction["preview"].width, 0)

    def test_blank_page_is_not_treated_as_scanned(self) -> None:
        extraction = processing.extract_pdf_pages(make_pdf("tbt"))
        self.assertEqual(extraction["page_kinds"], ["text", "blank", "text"])

    def test_full_page_image_is_scanned(self) -> None:
        extraction = processing.extract_pdf_pages(make_pdf("tst"))
        self.assertEqual(extraction["scanned_pages"], [2])

    def test_text_page_with_small_image_stays_text(self) -> None:
        doc = fitz.open()
        page = doc.new_page(width=1200, height=500)
        page.insert_text((40, 60), "This searchable text is long enough to be extracted from the page. " * 3)
        page.insert_image(fitz.Rect(40, 100, 300, 200), stream=_page_png("logo"))
        pdf_bytes = doc.tobytes()
        doc.close()
        self.assertEqual(processing.extract_pdf_pages(pdf_bytes)["page_kinds"], ["text"])

    def test_short_selectable_text_without_image_is_text(self) -> None:
        doc = fitz.open()
        doc.new_page().insert_text((40, 60), "Annexure A")
        pdf_bytes = doc.tobytes()
        doc.close()
        self.assertEqual(processing.extract_pdf_pages(pdf_bytes)["page_kinds"], ["text"])


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        # These tests cover the vision-model OCR path; a locally installed
        # Tesseract would otherwise read the pages first.
        patcher = patch.object(processing, "local_ocr", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("document_processing._request_analysis", side_effect=fake_model)
    def test_only_scanned_pages_reach_the_vision_model(self, request_analysis) -> None:
        kinds = "t" * 60 + "s" + "b" + "t" * 65  # 127 pages: 125 text, 1 scanned, 1 blank
        pdf_bytes = make_pdf(kinds)
        extraction = processing.extract_pdf_pages(pdf_bytes)
        stages = []
        result = processing.analyze_document_all_pages(
            extraction["page_texts"], "", pdf_bytes=pdf_bytes, page_kinds=extraction["page_kinds"],
            stage_callback=lambda stage, done, total: stages.append((stage, done, total)),
        )
        image_calls = [call for call in request_analysis.call_args_list if call.args[0]]
        self.assertEqual(len(image_calls), 1)
        self.assertEqual(len(image_calls[0].args[0]), 1)
        self.assertIn("[Page 61]", image_calls[0].args[1])
        self.assertIn(("ocr", 1, 1), stages)
        self.assertEqual(result["ocr_pages"], [61])
        self.assertIn("OCR text for [Page 61]", result["retrieval_pages"][60])
        self.assertEqual(result["page_count"], 127)
        # Bounded number of text calls: fields + section summaries + merge.
        text_calls = [call for call in request_analysis.call_args_list if not call.args[0]]
        self.assertLessEqual(len(text_calls), 1 + processing.SUMMARY_MAX_SECTIONS + 1)

    @patch("document_processing._request_analysis", side_effect=fake_model)
    def test_short_document_uses_one_combined_call(self, request_analysis) -> None:
        pages = processing.extract_pdf_pages(make_pdf("t" * 3))["page_texts"]
        result = processing.analyze_document_all_pages(pages, "")
        self.assertEqual(request_analysis.call_count, 1)
        self.assertEqual(request_analysis.call_args.kwargs.get("mode"), "document")
        for number in range(1, 4):
            self.assertIn(f"[Page {number}]", request_analysis.call_args.args[1])
        self.assertEqual(result["summary"], "document summary")

    @patch("document_processing._request_analysis")
    def test_empty_fields_are_not_a_successful_extraction(self, request_analysis) -> None:
        # Observed with qwen3.5:4b: the fields object leaked into main_points as
        # 'Required_documents:[],' and "fields" came back empty.
        combined = {"document_type": "Press Release", "summary": "CBSE issued guidelines.",
                    "main_points": ["Class X keeps two languages.", "Required_documents:[],"], "fields": {}, "_model": "m"}
        fields = {"document_type": "Press Release", "fields": {"issuing_department": "CBSE", "date": "29-06-2026"},
                  "field_sources": {"issuing_department": [1]}, "_model": "m"}
        request_analysis.side_effect = [combined, fields]
        result = processing.analyze_document_text(["PRESS RELEASE New Delhi, 29-06-2026. CBSE issues guidelines."])
        self.assertEqual([call.kwargs.get("mode") for call in request_analysis.call_args_list], ["document", "fields"])
        self.assertEqual(result["fields"]["issuing_department"], "CBSE")
        self.assertEqual(result["main_points"], ["Class X keeps two languages."])
        self.assertTrue(result["fields_retried"])
        # One document type for the overview and the extracted information.
        self.assertEqual(result["document_type"], result["fields"]["document_type"])

    @patch("document_processing._request_analysis")
    def test_empty_summary_is_requested_again(self, request_analysis) -> None:
        # Observed live: fields come first, and the reply sometimes ends before the summary.
        combined = {"document_type": "Press Release", "fields": {"issuing_department": "CBSE"}, "summary": "", "_model": "m"}
        summary = {"summary": "CBSE issued language guidelines.", "main_points": ["Class X keeps two languages."], "_model": "m"}
        request_analysis.side_effect = [combined, summary]
        result = processing.analyze_document_text(["PRESS RELEASE. CBSE issues guidelines."])
        self.assertEqual([call.kwargs.get("mode") for call in request_analysis.call_args_list], ["document", "summary"])
        self.assertEqual(result["summary"], "CBSE issued language guidelines.")
        self.assertEqual(result["fields"]["issuing_department"], "CBSE")
        self.assertTrue(result["summary_retried"])

    def test_raw_structures_never_reach_display_text(self) -> None:
        output = {"fields": {"required_documents": [{"required_documents": []}, "['Aadhaar card', 'Photo']"],
                             "amount": {"general": "Rs 500", "reserved": None}, "eligibility_criteria": []},
                  "main_points": ["Apply online. eligibility_criteria: [], fees: null"], "summary": {"overview": "A notice."}}
        processing.clean_fields(output)
        self.assertEqual(output["fields"]["required_documents"], ["Aadhaar card; Photo"])
        self.assertEqual(output["fields"]["amount"], "General: Rs 500")
        self.assertNotIn("eligibility_criteria", output["fields"])
        self.assertEqual(output["main_points"], ["Apply online."])
        # The structured summary's overview is the summary text, without its key name.
        self.assertEqual(output["summary"], "A notice.")

    @patch("document_processing._request_analysis")
    def test_question_translation_that_answers_instead_is_rejected(self, request_analysis) -> None:
        request_analysis.return_value = {"english": "The document does not contain information about who issued this document."}
        self.assertEqual(processing.question_in_english("ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")[0], "ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")
        request_analysis.return_value = {"english": "Who issued this document?", "_model": "m"}
        self.assertEqual(processing.question_in_english("ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")[0], "Who issued this document?")

    @patch("document_processing._request_analysis")
    def test_short_document_question_gets_whole_document_and_target_language(self, request_analysis) -> None:
        request_analysis.return_value = {"found": True, "answer": "ಸಿಬಿಎಸ್ಇ ಇದನ್ನು ಹೊರಡಿಸಿದೆ. [Page 2]", "_model": "m"}
        pages = ["PRESS RELEASE dated 29-06-2026.", "Issued by CBSE, Delhi."]
        answer = processing.answer_document_question(pages, "ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?", target_language="Kannada")
        self.assertEqual(request_analysis.call_args.kwargs["target_language"], "Kannada")
        self.assertIn("[Page 1]", request_analysis.call_args.args[1])
        self.assertIn("[Page 2]", request_analysis.call_args.args[1])
        self.assertEqual(answer["answer_pages"], [2])

    @patch.dict(os.environ, {"HF_CHAT_COMPLETIONS_URL": ""})
    @patch("document_processing.requests.Session.post")
    def test_question_prompt_puts_document_before_question_and_requires_language(self, post) -> None:
        reply = Mock(status_code=200, ok=True)
        reply.json.return_value = {"model": "qwen3.5:4b", "message": {"content": '{"found":true,"answer":"ಉತ್ತರ"}'}}
        post.return_value = reply
        processing._request_analysis([], "[Page 1]\nNotice text", "ಪ್ರಶ್ನೆ?", mode="question", target_language="Kannada")
        system, user = post.call_args.kwargs["json"]["messages"]
        self.assertIn("PatraAI", system["content"])
        self.assertLess(user["content"].index("Notice text"), user["content"].index("ಪ್ರಶ್ನೆ?"))
        self.assertIn("The required response language is Kannada", user["content"])
        self.assertIn("MUST generate the complete answer in Kannada", user["content"])

    def test_found_false_is_an_abstention_in_any_language(self) -> None:
        with patch("document_processing._request_analysis", return_value={"found": False, "answer": "ಈ ಮಾಹಿತಿ ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ. [Page 1]"}):
            answer = processing.answer_document_question(["Some notice text."], "ಶುಲ್ಕ ಎಷ್ಟು?", target_language="Kannada")
        self.assertFalse(answer["found"])

    def test_issue_date_not_printed_in_source_is_removed(self) -> None:
        output = {"fields": {"date": "2026-07-01", "document_title": "Notice"}, "field_sources": {"date": [1]}}
        processing.remove_unsupported_inferred_dates(output, ["Notice dated 29-06-2026"])
        self.assertIsNone(output["fields"]["date"])

    def test_qa_chunks_overlap_within_a_page(self) -> None:
        chunks = processing._page_chunks(["word " * 400], 400, overlap=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0][1].endswith(chunks[1][1][:40].strip()))

    def test_indian_language_retrieval_matches_inflected_words(self) -> None:
        pages = ["ಸಾಮಾನ್ಯ ಮಾಹಿತಿ " * 50, "ಈ ದಾಖಲೆ ಶಿಕ್ಷಣ ಇಲಾಖೆಯಿಂದ ಹೊರಡಿಸಲಾಗಿದೆ " * 10, "ಇತರೆ ವಿಷಯ " * 50]
        excerpts = processing.retrieve_relevant_chunks(pages, "ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ", top_k=1)
        self.assertTrue(excerpts[0].startswith("[Page 2]"))

    @patch("document_processing._request_analysis", side_effect=fake_model)
    def test_ocr_cache_resumes_without_repeating_pages(self, request_analysis) -> None:
        pdf_bytes = make_pdf("ss")
        cache = {1: {"text": "already read", "method": "Local Ollama", "model": "vision"}}
        results = processing.ocr_scanned_pages(pdf_bytes, [1, 2], cache=cache)
        self.assertEqual(request_analysis.call_count, 1)
        self.assertEqual(results[1]["text"], "already read")
        self.assertIn(2, cache)
        processing.ocr_scanned_pages(pdf_bytes, [1, 2], cache=cache)
        self.assertEqual(request_analysis.call_count, 1)

    def test_summary_sections_account_for_every_page(self) -> None:
        pages = [f"Section text for page {number}. " * 80 for number in range(1, 128)]
        pages[4] = ""
        sections = processing.plan_summary_sections(pages)
        self.assertLessEqual(len(sections), processing.SUMMARY_MAX_SECTIONS)
        self.assertEqual(sections[0]["first_page"], 1)
        self.assertEqual(sections[-1]["last_page"], 127)
        for previous, following in zip(sections, sections[1:]):
            self.assertEqual(previous["last_page"] + 1, following["first_page"])
        for section in sections:
            self.assertLessEqual(sum(len(text) for _, text in section["chunks"]), processing.SUMMARY_SECTION_CHARS)

    def test_field_evidence_keeps_first_page_and_finds_deadline(self) -> None:
        pages = ["Department of Higher Education\nAdmission Notice 2026"] + ["General guidance about campus life. " * 60] * 80
        pages.append("The last date to submit the application is 30 November 2026.")
        evidence = processing.select_field_evidence(pages, budget=6000)
        evidence_pages = {page for page, _ in evidence}
        self.assertIn(1, evidence_pages)
        self.assertIn(82, evidence_pages)

    @patch("document_processing._request_analysis")
    def test_question_retrieval_sends_relevant_page_chunks_and_uses_model_citation(self, request_analysis) -> None:
        request_analysis.return_value = {
            "answer": "Submit the form by 30 November 2026. [Page 101]",
            "fields": {}, "field_sources": {}, "_backend": "Local Ollama", "_model": "gemma4:e4b",
        }
        pages = [f"Routine guidance on applications, section {number}." for number in range(1, 101)]
        pages.append("Special exception: late applications are accepted only for documented emergencies.")

        answer = processing.answer_document_question(pages, "What special exception applies?")

        self.assertEqual(request_analysis.call_count, 1)
        source_text = request_analysis.call_args.args[1]
        self.assertIn("[Page 101]", source_text)
        self.assertNotIn("[Page 1]\n", source_text)
        self.assertEqual(answer["answer_pages"], [101])

    @patch("document_processing._request_analysis_once")
    def test_copied_template_placeholders_are_retried_and_never_shown(self, once) -> None:
        template = {"document_type": "Press Release", "summary": "...", "main_points": ["..."],
                    "fields": {"document_title": "...", "issuing_department": "...", "date": "..."}, "_model": "m"}
        real = {"document_type": "Press Release", "summary": "CBSE issued language guidelines.", "main_points": ["Two languages must be Indian."],
                "fields": {"document_title": "Three-language policy", "issuing_department": "CBSE", "date": "..."}, "_model": "m"}
        once.side_effect = [template, real]
        result = processing.analyze_document_all_pages(["Press release dated 29-06-2026 by CBSE on the three-language policy."], "")
        self.assertEqual(once.call_count, 2)
        self.assertIn("previous reply copied the template", once.call_args.kwargs["retry_note"])
        self.assertEqual(result["summary"], "CBSE issued language guidelines.")
        self.assertEqual(result["fields"]["issuing_department"], "CBSE")
        self.assertNotIn("date", result["fields"])  # a leftover placeholder is dropped, not shown

    def test_literal_amount_question_uses_exact_source_value_and_page(self) -> None:
        answer = processing.answer_document_question(
            ["Invoice amount: Rs 25,000. Please pay within 15 days."],
            "What is the amount due?",
        )
        self.assertEqual(answer["answer_pages"], [1])
        self.assertIn("Rs 25,000", answer["answer"])
        self.assertEqual(answer["answer_method"], "source-text match")

    def test_amount_shortcut_ignores_unlabelled_income_limits(self) -> None:
        pages = ["Reservation applies when family income is below Rs. 8 Lakh per annum."] * 3
        pages.append("Application fee for General Category\n₹ 1300/-\nReserved Category ₹ 1000/-")
        self.assertEqual(processing._labelled_amount(pages[0]), "")
        answer = processing.answer_document_question(pages, "What is the application fee?")
        self.assertIn("₹ 1300/-", answer["answer"])
        self.assertEqual(answer["answer_pages"], [4])

    @patch("document_processing._request_analysis")
    def test_truncated_answer_is_retried_and_abstention_has_no_pages(self, request_analysis) -> None:
        request_analysis.side_effect = [
            {"answer": "The document mentions a ", "field_sources": {}},
            {"answer": "The document does not provide enough information.", "field_sources": {"answer": [3, 5]}},
        ]
        answer = processing.answer_document_question(["Some guidance about hostels."] * 5, "What is the hostel deadline policy?")
        self.assertEqual(request_analysis.call_count, 2)
        self.assertTrue(request_analysis.call_args.kwargs["use_fallback"])
        self.assertEqual(answer["answer_pages"], [])

    def test_placeholder_field_values_are_removed_and_pages_normalized(self) -> None:
        output = {"fields": {"date": "N/A", "amount": "Rs 500", "important_dates": ["Not specified"]},
                  "field_sources": {"date": [], "amount": ["Page 4"], "important_dates": [1]}}
        processing.clean_fields(output)
        self.assertEqual(output["fields"], {"amount": "Rs 500"})
        self.assertEqual(output["field_sources"], {"amount": [4]})

    def test_specific_final_page_code_beats_an_earlier_reference_number(self) -> None:
        pages = ["Reference Number: LONG-2026-0001"] + ["Continuation guidance."] * 9
        pages.append("Additional review code: END-PAGE-7319")
        answer = processing.answer_document_question(pages, "What is the review code on the final page?")
        self.assertIn("END-PAGE-7319", answer["answer"])
        self.assertNotIn("LONG-2026-0001", answer["answer"])
        self.assertEqual(answer["answer_pages"], [11])

    @patch.dict(os.environ, {"OLLAMA_TEXT_MODEL": "gemma4:e4b", "OLLAMA_FALLBACK_MODEL": "qwen3.5:4b", "HF_CHAT_COMPLETIONS_URL": ""})
    @patch("document_processing.requests.Session.post")
    def test_repeat_limit_retries_and_reports_actual_local_fallback(self, post) -> None:
        failed = Mock(status_code=500, ok=False, text='{"error":"prediction aborted, token repeat limit reached"}')
        succeeded = Mock(status_code=200, ok=True)
        succeeded.json.return_value = {
            "model": "qwen3.5:4b",
            "message": {"content": '{"summary":"Observed","fields":{},"field_sources":{},"answer":""}'},
        }
        post.side_effect = [failed, succeeded]

        result = processing._request_analysis([], "[Page 1] Example notice", "")

        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen3.5:4b")
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_ctx"], processing.OLLAMA_NUM_CTX)
        self.assertEqual(result["_backend"], "Local Ollama fallback")
        self.assertEqual(result["_model"], "qwen3.5:4b")

    @patch.dict(os.environ, {"OLLAMA_TEXT_MODEL": "gemma4:e4b", "OLLAMA_FALLBACK_MODEL": "qwen3.5:4b", "HF_CHAT_COMPLETIONS_URL": ""})
    @patch("document_processing.requests.Session.post")
    def test_output_limit_retries_with_alternate_local_model(self, post) -> None:
        limited = Mock(status_code=200, ok=True)
        limited.json.return_value = {"done_reason": "length", "model": "gemma4:e4b", "message": {"content": ""}}
        succeeded = Mock(status_code=200, ok=True)
        succeeded.json.return_value = {
            "model": "qwen3.5:4b",
            "message": {"content": '{"summary":"Observed","fields":{},"field_sources":{},"answer":""}'},
        }
        post.side_effect = [limited, succeeded]

        result = processing._request_analysis([], "[Page 1] Example notice", "")

        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen3.5:4b")
        self.assertEqual(result["_backend"], "Local Ollama fallback")
        self.assertEqual(result["_model"], "qwen3.5:4b")

    @patch.dict(os.environ, {"HF_CHAT_COMPLETIONS_URL": "", "OLLAMA_BASE_URL": "http://127.0.0.1:9"})
    def test_unavailable_model_service_fails_fast_without_exposing_url(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            processing.check_model_service()
        self.assertIn("not running", str(raised.exception))
        self.assertNotIn("127.0.0.1", str(raised.exception))

    @patch("document_processing._request_analysis")
    def test_removes_model_calculated_deadline_not_printed_in_source(self, request_analysis) -> None:
        request_analysis.return_value = {
            "summary": "Notice issued 12 October 2026.",
            "fields": {
                "date": "2026-10-12", "deadline": "2026-11-07",
                "important_dates": [
                    {"value": "2026-10-12", "description": "Notice date"},
                    {"value": "2026-11-07", "description": "Calculated from 15 days"},
                ],
            },
            "field_sources": {"date": [1], "deadline": [1], "important_dates": [1]},
        }
        result = processing.analyze_document_all_pages(
            ["Date: 12/10/2026. Respond within 15 days."], "",
        )
        self.assertEqual(result["fields"]["date"], "2026-10-12")
        self.assertNotIn("deadline", result["fields"])
        self.assertEqual(result["fields"]["important_dates"], ["Notice date: 2026-10-12"])
        self.assertNotIn("deadline", result["field_sources"])


if __name__ == "__main__":
    unittest.main()
