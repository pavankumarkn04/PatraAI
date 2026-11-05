"""The "Download all results (PDF)" file: every results section, in the shown
language, with sources and without model artefacts (no model needed)."""

from __future__ import annotations

import importlib
import sys
import unittest

import fitz


def results_app():
    # Importing the page in Streamlit's bare mode defines its functions.
    sys.argv = sys.argv[:1]
    return importlib.import_module("Indian")


class ResultsPdfTests(unittest.TestCase):
    def test_results_pdf_has_every_section_in_the_shown_language(self) -> None:
        app = results_app()
        analysis = {"fields": {"document_title": "Budget release order", "document_type": "Government Order",
                               "issuing_department": "Backward Classes Welfare Department", "reference_number": "G.O.Rt.No. 39"},
                    "field_sources": {"reference_number": [1]}, "additional_fields": [{"label": "Scheme", "value": "Study Circle", "pages": [1]}],
                    "summary": "The department sanctions funds.", "main_points": [f"Point {index} about the order." for index in range(60)],
                    "applies_to": "", "actions": [], "summary_dates": []}
        hindi = {"fields": {**analysis["fields"], "document_title": "बजट रिलीज़ आदेश"}, "additional_fields": analysis["additional_fields"],
                 "labels": app.english_labels(), "summary": "विभाग ने धनराशि स्वीकृत की।", "main_points": [], "summary_dates": [],
                 "applies_to": "", "actions": [], "service": "IndicTrans2"}
        entry = {"analysis": analysis, "language": {"language": "English"}, "localized": {"Hindi": hindi},
                 "qa_order": [("q", "English")], "answers": {("q", "English"): {
                     "question": "What is the reference number?", "answer": "**G.O.Rt.No. 39** [Page 1]", "answer_language": "English",
                     "question_language": "English", "answer_pages": [1], "found": True, "timings": {"total": 2.0}}}}
        document = {"page_kinds": ["text"], "unit": "page", "page_texts": ["text"]}
        pdf = app.results_pdf(entry, document, "order.pdf", None, "English", "Automatic")
        with fitz.open(stream=pdf, filetype="pdf") as opened:
            self.assertGreater(opened.page_count, 1)  # long content continues on further pages
            text = "".join(page.get_text() for page in opened).replace("\xa0", " ")
        for expected in ("PatraAI results", "Document Overview", "1 page", "Extracted Information", "G.O.Rt.No. 39", "p. 1",
                         "Document Summary", "Point 59 about the order.", "Translation: Hindi", "Questions and Answers",
                         "What is the reference number?", "Source: p. 1"):
            self.assertIn(expected, text)
        for artefact in ("[Page", "**", "{", "1 pages"):
            self.assertNotIn(artefact, text)


if __name__ == "__main__":
    unittest.main()
