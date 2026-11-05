"""Evaluation scoring rules, curated dataset integrity and report generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import benchmark, scoring
from evaluation.evaluate import evaluate


class ScoringTests(unittest.TestCase):
    def test_harmless_format_differences_are_accepted(self) -> None:
        self.assertTrue(scoring.field_matches("State Common Entrance Test Cell", "STATE COMMON ENTRANCE TEST CELL"))
        self.assertTrue(scoring.field_matches("26.05.2026", "26 May 2026"))
        self.assertTrue(scoring.field_matches("24 October 2025", "24 अक्टूबर, 2025"))
        self.assertTrue(scoring.field_matches("₹ 1300/-", "Rs. 1,300"))
        self.assertTrue(scoring.field_matches("2-5/2020", "मि. सं. 2-5/2020-प्रशासन (सीएचडी)"))
        self.assertTrue(scoring.field_matches({"any_of": ["केंद्रीय हिंदी निदेशालय", "Central Hindi Directorate"]}, "Central Hindi Directorate, New Delhi"))
        self.assertTrue(scoring.field_matches(["SSC (Std. X) mark sheet"], ["SSC (Std. X) mark sheet", "HSC mark sheet"]))

    def test_wrong_numbers_dates_and_values_are_rejected(self) -> None:
        self.assertFalse(scoring.field_matches("₹ 1300/-", "₹ 13000"))
        self.assertFalse(scoring.field_matches("26.05.2026", "26.06.2026"))
        self.assertFalse(scoring.field_matches("2-5/2020", "2-5/2021"))
        self.assertFalse(scoring.field_matches("E-110911/DSERT/NEP/ECCE/15/2026-TECD", "E-110911"))
        self.assertFalse(scoring.field_matches("State Common Entrance Test Cell", None))
        self.assertFalse(scoring.answer_matches("1300", "The fee is ₹ 13000. [Page 75]"))
        self.assertFalse(scoring.answer_matches("56 years", "The document does not provide enough information."))

    def test_answers_match_in_context_and_other_formats(self) -> None:
        self.assertTrue(scoring.answer_matches({"any_of": ["₹ 1300", "Rs. 1300"]}, "The document states: ₹ 1300/-. [Page 75]"))
        self.assertTrue(scoring.answer_matches("07/08/2026", "The circular is dated 7 August 2026."))
        self.assertTrue(scoring.answer_matches({"any_of": ["2nd Saturday", "second Saturday"]}, "It is held on the second Saturday of every month."))

    def test_failure_categories(self) -> None:
        self.assertEqual(scoring.field_error_category("26.05.2026", None), "Missing extracted field")
        self.assertEqual(scoring.field_error_category("26.05.2026", "1 June 2026"), "Incorrect extracted value")
        self.assertEqual(scoring.answer_error_category("56 years", "Not enough information is given; the document does not specify it."), "Answer not found in document (model abstained)")
        self.assertEqual(scoring.answer_error_category("56 years", "The age limit is 60 years."), "Incorrect answer")

    def test_longer_answers_accept_paraphrase_but_not_a_different_fact(self) -> None:
        expected = "Students in Class IX must study three languages with two Bhartiya Bhashas"
        self.assertTrue(scoring.answer_matches(expected, "Class IX students must study three languages, two of which are Bhartiya Bhashas."))
        self.assertFalse(scoring.answer_matches(expected, "Class X students study two languages."))

    def test_legacy_jsonl_scorer_reads_utf8_bom(self) -> None:
        case = {"id": "smoke", "expected_fields": {"date": "12 May 2026"}, "predicted_fields": {"date": "12 May 2026"},
                "expected_answer": "Reply in 30 days", "predicted_answer": "Reply in 30 days"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps(case) + "\n", encoding="utf-8-sig")
            self.assertEqual(evaluate(path)["field_extraction_accuracy"], 1.0)


class DatasetTests(unittest.TestCase):
    def test_curated_dataset_meets_project_requirements(self) -> None:
        cases = benchmark.load_cases()
        self.assertTrue(30 <= len(cases) <= 50)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for case in cases:
            self.assertTrue((benchmark.DATASET / "documents" / case["document"]).is_file(), case["document"])
            self.assertTrue(case["question"] and case["expected_answer"] and case["answer_pages"], case["id"])
            self.assertTrue(case["expected_fields"], case["id"])
            self.assertEqual(set(case["field_pages"]), set(case["expected_fields"]), case["id"])
            self.assertTrue(case["verification"] and case["source_reference"], case["id"])
            self.assertIn(case["source"], {"real", "derived"})  # no constructed documents
        documents = {case["document"] for case in cases}
        self.assertGreaterEqual(len(documents), 15)  # many independent documents, not one PDF split up
        self.assertLessEqual(max(sum(case["document"] == name for case in cases) for name in documents), 4)
        self.assertGreaterEqual(len({case["language"] for case in cases}), 5)
        self.assertGreaterEqual(len({case["document_type"] for case in cases}), 10)
        formats = " ".join(case["format"] for case in cases)
        for kind in ("Searchable PDF", "Scanned PDF", "Image", "DOCX"):
            self.assertIn(kind, formats)


class ReportTests(unittest.TestCase):
    def test_scores_cases_and_writes_csv_json_markdown_reports(self) -> None:
        cases = benchmark.load_cases()[:2]
        document = {"analysis": {"fields": {"issuing_department": "State Common Entrance Test Cell", "document_type": "Report"}}}
        rows = {}
        for case, answer in zip(cases, ({"answer": "Issued by the State Common Entrance Test Cell. [Page 1]", "answer_pages": [1]},
                                        {"answer": "The document does not provide enough information.", "answer_pages": []})):
            rows[case["id"]] = {"case": case, "fingerprint": "x", "error": "", "predicted_answer": answer["answer"],
                                "predicted_pages": answer["answer_pages"], **benchmark.score_case(case, document, answer, "")}
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "RESULTS", Path(directory)):
            report = benchmark.write_outputs(cases, rows, {"model": "test", "analysis_service": "test"})
            self.assertTrue((Path(directory) / benchmark.RESULTS_CSV).is_file())
            markdown = (Path(directory) / benchmark.REPORT_FILE).read_text(encoding="utf-8")
            self.assertTrue((Path(directory) / benchmark.RESULTS_JSON).is_file())
        metrics = report["metrics"]
        self.assertEqual((metrics["field_correct"], metrics["field_checks"]), (1, 2))
        self.assertEqual((metrics["qa_correct"], metrics["qa_checks"]), (1, 2))
        self.assertIn("English", metrics["by_language"])
        self.assertIn("Answer not found in document (model abstained)", metrics["failure_categories"])
        self.assertIn("**Field-extraction accuracy** | **50.0%**", markdown)
        self.assertIn("## 3. Results by document language", markdown)
        self.assertIn("## 10. Observations", markdown)
        self.assertIn("## 7. Correct and incorrect predictions", markdown)
        self.assertIn("## 11. Limitations", markdown)
        self.assertIn(f"{len(cases)} test cases over", markdown)


if __name__ == "__main__":
    unittest.main()
