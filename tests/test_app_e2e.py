"""Streamlit end-to-end tests of the website with local model-service test doubles."""

from __future__ import annotations

import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


REPO = Path(__file__).resolve().parents[1]
INTERNAL_DETAILS = ("127.0.0.1", "localhost", "11434", "OLLAMA_", "HF_CHAT_COMPLETIONS_URL", "INDIC_TRANS2_URL", "/api/chat")


# The fake translation writes each Latin letter as a Devanagari letter, so its
# output is in the Hindi script (the app checks the script of translations).
_DEVANAGARI = str.maketrans({chr(base + offset): chr(0x0915 + offset) for base in (65, 97) for offset in range(26)})


def to_hindi(text: str) -> str:
    return "HI: " + text.translate(_DEVANAGARI)


def analysis_for(prompt: str) -> dict:
    """Answers depend on the document in the prompt, like a real model."""
    if "Question: " in prompt and "The required response language is Kannada" in prompt:
        return {"found": True, "answer": "ಈ ಸೂಚನೆಯನ್ನು Accounts Office ಹೊರಡಿಸಿದೆ. [Page 1]"}
    if "Rewrite the given answer completely in Hindi" in prompt:
        return {"answer": "यह सूचना Accounts Office ने जारी की है।"}
    if "Leave Policy" in prompt:
        return {"document_type": "Office guideline", "language": "English",
                "summary": "The guideline explains how staff apply for leave.", "main_points": ["Apply ten days in advance."],
                "fields": {"issuing_department": "Sample Office", "document_title": "Leave Policy"},
                "field_sources": {"document_title": [1]}, "additional_fields": [{"label": "Approver", "value": "Section Head", "page": 2}]}
    no_summary = ServiceHandler.no_summary
    return {"document_type": "Invoice notice", "language": "English",
            "summary": "" if no_summary else {"overview": "The notice asks for payment of an invoice. It states the amount and the reference number.",
                                              "key_points": ["Payment is due within 15 days."],
                                              "important_dates": [{"date": "12/10/2026", "event": "Notice dated"},
                                                                  {"date": "01/01/2030", "event": "Invented hearing"}],
                                              "applies_to": "Customers of the Accounts Office", "actions": ["Pay the invoice."]},
            "fields": {"reference_number": "INV-2026-0042", "issuing_department": "Accounts Office",
                       "required_documents": [["Invoice copy"], ["Payment receipt"]]},
            "field_sources": {"reference_number": [1], "issuing_department": ["Page 1"]},
            "additional_fields": [{"label": "Invoice number", "value": "INV-2026-0042", "page": 1}]}


class ServiceHandler(BaseHTTPRequestHandler):
    received: dict[str, object] = {}
    analysis_calls = 0
    translation_calls = 0
    empty_model_output = False
    no_summary = False

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        type(self).received = json.loads(self.rfile.read(length))
        if "text" in type(self).received:
            type(self).translation_calls += 1
            text = str(type(self).received["text"])
            payload = {"translation": "\n".join(to_hindi(line) if line.strip() else line for line in text.split("\n"))}
        else:
            type(self).analysis_calls += 1
            prompt = type(self).received["messages"][0]["content"][0]["text"]
            content = "" if type(self).empty_model_output else json.dumps(analysis_for(prompt))
            payload = {"choices": [{"message": {"content": content}}]}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def all_text(at: AppTest) -> str:
    parts = [item.value for item in at.markdown] + [item.value for item in at.caption]
    parts += [str(item.value) for item in at.info] + [str(item.value) for item in at.error]
    parts += [str(item.value) for item in at.success] + [str(item.value) for item in at.warning]
    return "\n".join(parts)


def word_document() -> bytes:
    import docx

    document = docx.Document()
    document.add_heading("Leave Policy", level=1)
    document.add_paragraph("Staff must apply for leave in advance.")
    document.add_paragraph("Apply ten days before the leave starts", style="List Bullet")
    document.add_heading("Approval", level=1)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Approver", "Section Head"
    table.cell(1, 0).text, table.cell(1, 1).text = "Time limit", "3 days"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class AppEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        import streamlit as st

        # Each test starts with an empty document store (it persists per process).
        st.cache_resource.clear()
        st.cache_data.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ServiceHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        ServiceHandler.analysis_calls = ServiceHandler.translation_calls = 0
        ServiceHandler.empty_model_output = ServiceHandler.no_summary = False
        self.env = patch.dict(os.environ, {
            "INDIC_TRANS2_URL": f"http://127.0.0.1:{self.server.server_port}/translate",
            "HF_CHAT_COMPLETIONS_URL": f"http://127.0.0.1:{self.server.server_port}/chat/completions",
        })
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def analyze(self, at: AppTest, name: str, payload: bytes, mime: str) -> None:
        at.file_uploader[0].upload(name, payload, mime).run()
        next(button for button in at.button if button.label == "Analyze document").click().run()
        self.assertFalse(at.exception)

    def test_results_sections_language_translation_and_questions(self) -> None:
        at = AppTest.from_file(str(REPO / "Indian.py"), default_timeout=30).run()
        self.analyze(at, "sample-check.pdf", (REPO / "sample-check.pdf").read_bytes(), "application/pdf")
        self.assertEqual(ServiceHandler.analysis_calls, 1)
        visible = all_text(at)
        positions = [visible.find(title) for title in ("Document Overview", "Extracted Information", "Document Summary", "Translation", "Ask Questions", "Processing Details")]
        self.assertTrue(all(position >= 0 for position in positions) and positions == sorted(positions), positions)
        self.assertIn('href="#sec-translation"', visible)
        for expected in ("Invoice notice", "Accounts Office", "INV-2026-0042", "<li>Invoice copy</li><li>Payment receipt</li>",
                         "Not found in document", "p. 1", "The notice asks for payment of an invoice.", "Payment is due within 15 days.",
                         "<h4>Key points</h4>", "<h4>Who does this apply to?</h4>", "Customers of the Accounts Office",
                         "<h4>Important actions</h4>", "ANALYSIS READY"):
            self.assertIn(expected, visible)
        # An additional field that repeats an extracted value is shown once.
        self.assertNotIn("Invoice number", visible)
        # The summary has its own parts and does not repeat the extracted fields.
        summary_html = visible[visible.find('<div class="summary-text">'):]
        summary_html = summary_html[:summary_html.find("</div>")]
        self.assertNotIn("Required documents", summary_html)
        self.assertNotIn("Invoice copy", summary_html)
        self.assertIn("12/10/2026 — Notice dated", summary_html)
        self.assertNotIn("01/01/2030", summary_html)  # a summary date that is not printed in the document is removed
        self.assertIn("Detected document language: <b>English</b>", visible)
        self.assertEqual(ServiceHandler.translation_calls, 0)  # nothing is translated until asked

        # A question, answered from the stored pages.
        next(box for box in at.text_input if box.label.startswith("Your question")).set_value("What is the reference number?")
        next(button for button in at.button if button.label == "Ask").click().run()
        self.assertEqual(ServiceHandler.analysis_calls, 1)
        self.assertIn("What is the reference number?", all_text(at))
        self.assertIn("Source: p. 1", all_text(at))
        self.assertIn("Question: English · Answer: English", all_text(at))

        # Automatic mode: a Kannada question about this English document is
        # answered in Kannada (one model call, no re-analysis).
        next(box for box in at.text_input if box.label.startswith("Your question")).set_value("ಈ ಸೂಚನೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")
        next(button for button in at.button if button.label == "Ask").click().run()
        self.assertFalse(at.exception)
        visible = all_text(at)
        self.assertEqual(ServiceHandler.analysis_calls, 2)
        self.assertIn("ಈ ಸೂಚನೆಯನ್ನು Accounts Office ಹೊರಡಿಸಿದೆ.", visible)
        self.assertIn("Question: Kannada · Answer: Kannada", visible)
        # Asking it again reuses the stored answer.
        next(box for box in at.text_input if box.label.startswith("Your question")).set_value("ಈ ಸೂಚನೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")
        next(button for button in at.button if button.label == "Ask").click().run()
        self.assertEqual(ServiceHandler.analysis_calls, 2)

        # Response language: the structured results and the answers in Hindi,
        # without analyzing the document again.
        at.selectbox(key="response-language").select("Hindi").run()
        self.assertFalse(at.exception)
        visible = all_text(at)
        self.assertEqual(ServiceHandler.analysis_calls, 3)  # the Kannada answer is rewritten in Hindi; IndicTrans2 translates from English only
        self.assertEqual(ServiceHandler.received.get("target_language_code"), "hin_Deva")
        self.assertIn(to_hindi("The notice asks for payment of an invoice."), visible)
        self.assertIn(to_hindi("Accounts Office"), visible)
        self.assertIn("INV-2026-0042", visible)  # reference numbers are not translated
        self.assertIn(to_hindi("Issuing authority"), visible)  # field names too
        self.assertIn("showing results in <b>Hindi</b>", visible)
        self.assertIn("Answer: Hindi", visible)
        self.assertIn("यह सूचना Accounts Office ने जारी की है।", visible)
        self.assertNotIn("ಈ ಸೂಚನೆಯನ್ನು Accounts Office ಹೊರಡಿಸಿದೆ.", visible)  # no answer left in the wrong language

        # The Translation section reuses the Hindi results (no new call) and is structured.
        calls = ServiceHandler.translation_calls
        at.selectbox(key="translation-language").select("Hindi").run()
        next(button for button in at.button if button.label == "Translate").click().run()
        visible = all_text(at)
        self.assertEqual(ServiceHandler.translation_calls, calls)
        self.assertIn("Hindi · IndicTrans2 service", visible)
        self.assertNotIn("{'", visible)
        self.assertNotIn('["', visible)

        # Reruns keep everything without new model calls.
        before = (ServiceHandler.analysis_calls, ServiceHandler.translation_calls)
        at.run()
        self.assertEqual((ServiceHandler.analysis_calls, ServiceHandler.translation_calls), before)
        for detail in INTERNAL_DETAILS:
            self.assertNotIn(detail, all_text(at))

    def test_word_document_switching_documents_and_clear(self) -> None:
        at = AppTest.from_file(str(REPO / "Indian.py"), default_timeout=30).run()
        self.analyze(at, "sample-check.pdf", (REPO / "sample-check.pdf").read_bytes(), "application/pdf")
        self.assertIn("Accounts Office", all_text(at))

        # A different document: none of the first document's results appear.
        at.file_uploader[0].upload("leave-policy.docx", word_document(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document").run()
        visible = all_text(at)
        self.assertNotIn("Accounts Office", visible)
        self.assertNotIn("Results</h2>", visible)
        self.assertIn("2 sections · 1 table(s)", visible)
        next(button for button in at.button if button.label == "Analyze document").click().run()
        self.assertFalse(at.exception)
        visible = all_text(at)
        self.assertIn("Leave Policy", visible)
        self.assertIn("Section 1 · Leave Policy", visible)       # references by section, not invented pages
        self.assertIn("Section 2 · Approval", visible)
        self.assertNotIn("Accounts Office", visible)
        self.assertNotIn("INV-2026-0042", visible)

        next(button for button in at.button if button.label == "Clear document").click().run()
        visible = all_text(at)
        self.assertNotIn("Results</h2>", visible)
        self.assertNotIn("Leave Policy", visible)

    def test_empty_model_response_shows_an_error_not_done(self) -> None:
        ServiceHandler.empty_model_output = True
        at = AppTest.from_file(str(REPO / "Indian.py"), default_timeout=30).run()
        pdf = (REPO / "evaluation" / "samples" / "long-searchable-11-pages.pdf").read_bytes()
        self.analyze(at, "eleven-pages.pdf", pdf, "application/pdf")
        visible = all_text(at)
        self.assertTrue(any("failed" in str(error.value) and "empty response" in str(error.value) for error in at.error), visible)
        self.assertNotIn("Results</h2>", visible)
        self.assertIn('class="failed"', visible)
        at.run()
        self.assertTrue(any("empty response" in str(error.value) for error in at.error))
        self.assertTrue(any(button.label == "Analyze document" for button in at.button))

    def test_summary_that_stays_empty_is_reported_not_shown_as_success(self) -> None:
        ServiceHandler.no_summary = True
        at = AppTest.from_file(str(REPO / "Indian.py"), default_timeout=30).run()
        self.analyze(at, "sample-check.pdf", (REPO / "sample-check.pdf").read_bytes(), "application/pdf")
        self.assertEqual(ServiceHandler.analysis_calls, 2)  # the combined call, then one targeted summary request
        visible = all_text(at)
        self.assertIn("PARTIAL RESULTS", visible)
        self.assertIn("INV-2026-0042", visible)  # the extracted information is still shown
        self.assertTrue(any("summary could not be generated" in str(error.value) for error in at.error), visible)
        self.assertFalse(any("Analysis complete" in str(item.value) for item in at.success))
        self.assertTrue(any("summary" in str(item.value) for item in at.warning))

    def test_public_site_has_only_document_understanding_and_no_evaluation(self) -> None:
        at = AppTest.from_file(str(REPO / "Indian.py"), default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.tabs), 0)
        visible = all_text(at)
        for absent in ("Evaluation", "evaluation", "accuracy", "Test case", "test case", "Failure analysis", "dataset"):
            self.assertNotIn(absent, visible)
        self.assertEqual(len(at.file_uploader), 1)
        self.assertIn('class="footer-brand"', visible)
        self.assertEqual({"Privacy Policy", "Terms & Conditions"}, {expander.label for expander in at.expander})
        for detail in INTERNAL_DETAILS:
            self.assertNotIn(detail, visible)


if __name__ == "__main__":
    unittest.main()
