"""Regression tests for local translation and the optional API override."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import requests

from translation import TRANSLATION_LANGUAGES, make_translation_source, translate_text


class TranslationTests(unittest.TestCase):
    def test_only_requested_languages_are_available(self) -> None:
        self.assertEqual(list(TRANSLATION_LANGUAGES), ["English", "Kannada", "Hindi", "Tamil", "Telugu", "Malayalam"])
        self.assertEqual(TRANSLATION_LANGUAGES["Tamil"][0], "tam_Taml")
        self.assertEqual(TRANSLATION_LANGUAGES["Malayalam"][0], "mal_Mlym")

    def test_english_returns_source_text_without_service_call(self) -> None:
        with patch("translation.requests.Session") as session:
            self.assertEqual(translate_text("Hello", "English"), "Hello")
            session.assert_not_called()

    def test_rejects_unsupported_language_and_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            translate_text("Hello", "French")
        with self.assertRaisesRegex(ValueError, "no analysis text"):
            translate_text("  ", "Hindi")

    @patch.dict(os.environ, {"INDIC_TRANS2_URL": "http://translation.test/translate"})
    @patch("translation.requests.post")
    def test_optional_api_override_uses_explicit_language_pair(self, post: Mock) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"translation": "నమస్కారం"}
        post.return_value = response
        self.assertEqual(translate_text("Hello", "Telugu"), "నమస్కారం")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["source_language_code"], "eng_Latn")
        self.assertEqual(payload["target_language_code"], "tel_Telu")
        self.assertEqual(payload["target_language"], "Telugu")

    @patch.dict(os.environ, {"INDICTRANS2_LOCAL": "0"}, clear=True)
    @patch("translation.requests.Session")
    def test_local_ollama_request_and_response(self, session_class: Mock) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"message": {"content": "ನಮಸ್ಕಾರ"}}
        session = session_class.return_value.__enter__.return_value
        session.post.return_value = response

        self.assertEqual(translate_text("Hello", "Kannada"), "ನಮಸ್ಕಾರ")
        self.assertFalse(session.trust_env)
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:11434/api/chat")
        body = kwargs["json"]
        self.assertEqual(body["model"], "qwen3.5:4b")
        self.assertFalse(body["think"])
        self.assertIn("Kannada", body["messages"][0]["content"])
        self.assertEqual(body["messages"][1]["content"], "Hello")

    @patch.dict(os.environ, {"INDICTRANS2_LOCAL": "0"}, clear=True)
    @patch("translation.requests.Session")
    def test_local_service_unavailable_is_actionable(self, session_class: Mock) -> None:
        session = session_class.return_value.__enter__.return_value
        session.post.side_effect = requests.exceptions.ConnectionError("offline")
        with self.assertRaisesRegex(RuntimeError, "Start Ollama"):
            translate_text("Hello", "Hindi")

    @patch.dict(os.environ, {"INDICTRANS2_LOCAL": "0"}, clear=True)
    @patch("translation.requests.Session")
    def test_long_local_translation_is_chunked_with_progress(self, session_class: Mock) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"message": {"content": "अनुवाद"}}
        session = session_class.return_value.__enter__.return_value
        session.post.return_value = response
        progress: list[tuple[int, int]] = []

        result = translate_text("This is a long document. " * 100, "Hindi", lambda done, total: progress.append((done, total)))

        self.assertEqual(result, "अनुवाद\nअनुवाद")
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(progress, [(1, 2), (2, 2)])

    def test_translation_source_includes_summary_fields_and_answer(self) -> None:
        text = make_translation_source({
            "summary": "A notice was issued.",
            "fields": {"date": "12 May", "amount": None},
            "answer": "Reply within 30 days.",
        })
        self.assertIn("A notice was issued.", text)
        self.assertIn("Date: 12 May", text)
        self.assertIn("Reply within 30 days.", text)
        self.assertNotIn("Amount", text)


if __name__ == "__main__":
    unittest.main()
