"""Question-language detection and answer-language control (no live models)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import question_answering as qa  # noqa: E402
from document_profiles import document_category, field_rows, suggested_questions  # noqa: E402
from language_detection import detect_question_language, is_in_language, resolve_answer_language  # noqa: E402

AUTO = "Automatic"
ENGLISH_DOCUMENT = ["PRESS RELEASE. New Delhi, 29-06-2026. CBSE issues guidelines on the three-language policy."]


class QuestionLanguageTests(unittest.TestCase):
    def test_native_script_questions(self) -> None:
        cases = {
            "ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?": "Kannada", "इस दस्तावेज़ को किसने जारी किया?": "Hindi",
            "இந்த ஆவணத்தை யார் வெளியிட்டார்?": "Tamil", "ఈ పత్రాన్ని ఎవరు జారీ చేశారు?": "Telugu",
            "ഈ രേഖ ആരാണ് പുറപ്പെടുവിച്ചത്?": "Malayalam", "Who issued this document?": "English",
        }
        for question, language in cases.items():
            self.assertEqual(detect_question_language(question)["language"], language, question)

    def test_native_question_with_english_terms(self) -> None:
        self.assertEqual(detect_question_language("CBSE circular ಯಾವಾಗ ಬಿಡುಗಡೆಯಾಯಿತು?")["language"], "Kannada")

    def test_romanised_questions(self) -> None:
        cases = {
            "ee dakhaleyannu yaaru horadisiddare?": "Kannada", "is dastavez ko kisne jari kiya hai?": "Hindi",
            "indha aavanathai yaar veliyittathu?": "Tamil", "ee patram evaru jaari chesaru?": "Telugu",
            "ee rekha aaraanu purappeduvichathu?": "Malayalam", "what is the last date to apply?": "English",
        }
        for question, language in cases.items():
            detected = detect_question_language(question)
            self.assertEqual(detected["language"], language, question)
            self.assertTrue(detected["reliable"], question)

    def test_resolution_rules(self) -> None:
        kannada = detect_question_language("ಕೊನೆಯ ದಿನಾಂಕ ಯಾವುದು?")
        self.assertEqual(resolve_answer_language(AUTO, AUTO, kannada), "Kannada")          # automatic follows the question
        self.assertEqual(resolve_answer_language("English", AUTO, kannada), "English")     # explicit choice wins
        self.assertEqual(resolve_answer_language("Tamil", AUTO, detect_question_language("Who?")), "Tamil")
        unclear = detect_question_language("2026?")
        self.assertEqual(resolve_answer_language(AUTO, AUTO, unclear, previous="Kannada"), "Kannada")  # follow-up keeps language

    def test_answer_language_check(self) -> None:
        self.assertTrue(is_in_language("ಈ ದಾಖಲೆಯನ್ನು CBSE ಹೊರಡಿಸಿದೆ (29-06-2026).", "Kannada"))
        self.assertFalse(is_in_language("The document was issued by CBSE.", "Kannada"))
        self.assertFalse(is_in_language("यह दस्तावेज़ CBSE द्वारा जारी किया गया है।", "Kannada"))
        self.assertTrue(is_in_language("The document was issued by CBSE.", "English"))
        self.assertTrue(is_in_language("29-06-2026", "Tamil"))  # numbers only: nothing to convert


def model_answer(answer: str, found: bool = True):
    return {"found": found, "answer": answer, "answer_pages": [1], "answer_method": "whole document + model", "_model": "qwen3.5:4b", "_backend": "Local Ollama"}


class AnswerQuestionTests(unittest.TestCase):
    def ask(self, question: str, choice: str = AUTO, pages: list[str] | None = None, **kwargs):
        return qa.answer_question(pages or ENGLISH_DOCUMENT, question, response_choice=choice, automatic=AUTO, **kwargs)

    @patch("question_answering.answer_document_question", return_value=model_answer("ಈ ದಾಖಲೆಯನ್ನು CBSE ಹೊರಡಿಸಿದೆ. [Page 1]"))
    def test_english_document_kannada_question_kannada_answer(self, answer) -> None:
        record = self.ask("ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")
        self.assertEqual(answer.call_args.kwargs["target_language"], "Kannada")
        self.assertEqual((record["question_language"], record["answer_language"]), ("Kannada", "Kannada"))
        self.assertEqual(record["answer"], "ಈ ದಾಖಲೆಯನ್ನು CBSE ಹೊರಡಿಸಿದೆ.")
        self.assertNotIn("translated", record["language_method"])  # already correct: not translated again

    @patch("question_answering.translate_structured_text", return_value={"text": "ಈ ದಾಖಲೆಯನ್ನು CBSE ಹೊರಡಿಸಿದೆ.", "service": "IndicTrans2 (test)"})
    @patch("question_answering.answer_document_question", return_value=model_answer("The document was issued by **CBSE**. [Page 1]"))
    def test_english_reply_to_kannada_question_is_translated_with_indictrans2(self, _answer, translate) -> None:
        record = self.ask("ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")
        translate.assert_called_once_with("The document was issued by **CBSE**.", "Kannada")
        self.assertEqual(record["answer"], "ಈ ದಾಖಲೆಯನ್ನು CBSE ಹೊರಡಿಸಿದೆ.")
        self.assertIn("IndicTrans2", record["language_method"])

    @patch("question_answering.rewrite_answer_language", return_value={"answer": "Still in English."})
    @patch("question_answering.translate_structured_text", side_effect=RuntimeError("model unavailable"))
    @patch("question_answering.answer_document_question", return_value=model_answer("The document was issued by CBSE."))
    def test_failed_translation_raises_instead_of_showing_wrong_language(self, *_mocks) -> None:
        with self.assertRaises(qa.AnswerLanguageError) as raised:
            self.ask("इस दस्तावेज़ को किसने जारी किया?")
        self.assertIn("Hindi", str(raised.exception))

    @patch("question_answering.rewrite_answer_language", return_value={"answer": "ಈ ದಾಖಲೆಯನ್ನು CBSE ಹೊರಡಿಸಿದೆ."})
    @patch("question_answering.answer_document_question", return_value=model_answer("यह दस्तावेज़ CBSE ने जारी किया है।"))
    def test_hindi_reply_to_kannada_question_is_rewritten(self, _answer, rewrite) -> None:
        record = self.ask("ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?", pages=["यह सूचना सीबीएसई द्वारा जारी की गई है।"], document_language="Hindi")
        rewrite.assert_called_once()
        self.assertEqual(record["answer_language"], "Kannada")
        self.assertIn("rewritten", record["language_method"])

    @patch("question_answering.answer_document_question", return_value=model_answer("The document was issued by CBSE."))
    def test_explicit_english_with_kannada_question_gives_english(self, answer) -> None:
        record = self.ask("ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?", choice="English")
        self.assertEqual(answer.call_args.kwargs["target_language"], "English")
        self.assertEqual(record["answer_language"], "English")

    @patch("question_answering.answer_document_question", return_value=model_answer("The notice was issued by the Commissioner."))
    def test_kannada_document_english_question_gives_english(self, answer) -> None:
        record = self.ask("Who issued this circular?", pages=["ಈ ಸುತ್ತೋಲೆಯನ್ನು ಆಯುಕ್ತರು ಹೊರಡಿಸಿದ್ದಾರೆ."], document_language="Kannada")
        self.assertEqual(answer.call_args.kwargs["target_language"], "English")
        self.assertEqual(record["answer_language"], "English")

    @patch("question_answering.answer_document_question", return_value=model_answer("ಶುಲ್ಕದ ಮಾಹಿತಿ ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ.", found=False))
    def test_not_found_stays_in_target_language(self, _answer) -> None:
        record = self.ask("ಅರ್ಜಿ ಶುಲ್ಕ ಎಷ್ಟು?")
        self.assertFalse(record["found"])
        self.assertTrue(is_in_language(record["answer"], "Kannada"))

    @patch("question_answering.translate_segments", return_value={"texts": ["ಯಾರು ಹೊರಡಿಸಿದರು"]})
    @patch("question_answering.question_in_english", return_value=("Who issued this document?", "translated"))
    @patch("question_answering.answer_document_question", return_value=model_answer("ಆಯುಕ್ತರು. [Page 40]"))
    def test_long_document_searches_with_other_wordings(self, answer, english, to_document_language) -> None:
        pages = ["ಸಾಮಾನ್ಯ ಮಾಹಿತಿ " * 200] * 40
        with patch("question_answering.translation_service_description", return_value={"is_indictrans2": True}):
            self.ask("इसे किसने जारी किया?", pages=pages, document_language="Kannada")
        english.assert_called_once()
        self.assertEqual(answer.call_args.kwargs["extra_queries"], ["Who issued this document?", "ಯಾರು ಹೊರಡಿಸಿದರು"])
        self.assertEqual(answer.call_args.kwargs["target_language"], "Hindi")

    @patch("question_answering.question_in_english")
    @patch("question_answering.answer_document_question", return_value=model_answer("ಸಿಬಿಎಸ್ಇ."))
    def test_short_document_needs_no_question_translation_call(self, _answer, english) -> None:
        self.ask("ಈ ದಾಖಲೆಯನ್ನು ಯಾರು ಹೊರಡಿಸಿದ್ದಾರೆ?")
        english.assert_not_called()

    @patch("question_answering.question_in_english", return_value=("Who issued this document?", "translated"))
    @patch("question_answering.answer_document_question", return_value=model_answer("ಈ ದಾಖಲೆಯನ್ನು CBSE ಹೊರಡಿಸಿದೆ."))
    def test_romanised_question_is_given_with_its_english_meaning(self, answer, english) -> None:
        record = self.ask("ee dakhaleyannu yaaru horadisiddare?")
        english.assert_called_once_with("ee dakhaleyannu yaaru horadisiddare?", romanised=True)
        self.assertIn("Who issued this document?", answer.call_args.args[1])
        self.assertEqual(record["answer_language"], "Kannada")
        self.assertEqual(record["question"], "ee dakhaleyannu yaaru horadisiddare?")

    def test_existing_answer_shown_in_another_language_is_cached(self) -> None:
        record = {"answer": "Issued by CBSE.", "answer_language": "English"}
        with patch("question_answering.translate_structured_text", return_value={"text": "CBSE द्वारा जारी।", "service": "IndicTrans2"}) as translate:
            self.assertEqual(qa.answer_in_language(record, "Hindi")["text"], "CBSE द्वारा जारी।")
            qa.answer_in_language(record, "Hindi")
        translate.assert_called_once()


class DocumentProfileTests(unittest.TestCase):
    def test_categories_and_relevant_fields(self) -> None:
        self.assertEqual(document_category({"document_type": "Caste Certificate"}), "certificate")
        self.assertEqual(document_category({"fields": {"document_type": "Scholarship notification"}}), "scholarship")
        self.assertEqual(document_category({"document_type": "Press Release"}), "notice")
        self.assertEqual(document_category({"document_type": "Information Brochure"}), "admission")
        # The title wins over a mixed model-written type, and a department name is not a scheme.
        self.assertEqual(document_category({"fields": {"document_title": "Guidelines on the three-language policy", "document_type": "Scheme/Notice"}}), "notice")
        self.assertEqual(document_category({"fields": {"document_title": "Budget release order", "document_type": "Government Order",
                                                       "issuing_department": "Backward Classes Welfare Department"}}), "notice")
        self.assertEqual(document_category({"fields": {"document_title": "Income certificate application", "document_type": "Application form"}}), "form")
        rows = dict(field_rows("certificate", {"required_documents": ["Photo"]}, ("required_documents", "eligibility_criteria")))
        self.assertTrue(rows["reference_number"])           # expected: shown even when missing
        self.assertFalse(rows["required_documents"])        # optional: shown because it is present
        self.assertNotIn("eligibility_criteria", rows)      # irrelevant and absent: hidden

    def test_suggestions_are_relevant(self) -> None:
        notice = suggested_questions("notice", {})
        self.assertNotIn("Which documents are required?", notice)
        self.assertIn("Which documents are required?", suggested_questions("notice", {"required_documents": ["Photo"]}))


if __name__ == "__main__":
    unittest.main()
