"""Concise structured summaries, clean extracted values, resumable analysis,
structured model output and faster IndicTrans2 batching (no model needed)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import document_processing as processing
import translation
from localization import localize


def reply(**values):
    return {"_backend": "Local Ollama", "_model": "test-model", "_mode": values.pop("_mode", "document"), "_seconds": 1.0, **values}


class ExtractedValueTests(unittest.TestCase):
    def test_page_references_move_to_the_source_column(self) -> None:
        output = {"fields": {
            "important_instructions": [{"condition": "Students already in Class X continue with the old system.", "page": 2},
                                       "Schools must report by 30 June; Page: 3"],
            "document_title": "Three-language policy [Page 1]",
            "eligibility_criteria": ["Candidates aged 18 or more (p. 4)"],
        }, "field_sources": {}}
        processing.clean_fields(output)
        fields = output["fields"]
        self.assertEqual(fields["important_instructions"], ["Students already in Class X continue with the old system.", "Schools must report by 30 June"])
        self.assertEqual(fields["document_title"], "Three-language policy")
        self.assertEqual(fields["eligibility_criteria"], ["Candidates aged 18 or more"])
        self.assertEqual(output["field_sources"], {"important_instructions": [2], "document_title": [1], "eligibility_criteria": [4]})
        for value in fields.values():
            for item in value if isinstance(value, list) else [value]:
                self.assertNotRegex(item, r"(?i)\bpage\b|\bp\.\s*\d|Condition:")

    def test_a_page_named_inside_a_sentence_is_kept(self) -> None:
        output = {"fields": {"application_procedure": ["Fill in page 2 of the form before signing."]}}
        processing.clean_fields(output)
        self.assertEqual(output["fields"]["application_procedure"], ["Fill in page 2 of the form before signing."])

    def test_page_and_section_numbers_are_not_required_documents(self) -> None:
        output = {"fields": {"required_documents": ["Page 1", "p. 3", "Page 2 of 5", "Section 4", "1/4", "Income certificate", "Annexure-II form"]}}
        processing.clean_fields(output)
        self.assertEqual(output["fields"]["required_documents"], ["Income certificate", "Annexure-II form"])

    def test_meaningful_single_key_labels_are_kept(self) -> None:
        output = {"fields": {"amount": {"general": "Rs 500"}, "important_instructions": [{"instruction": "Bring the hall ticket."}]}}
        processing.clean_fields(output)
        self.assertEqual(output["fields"]["amount"], "General: Rs 500")
        self.assertEqual(output["fields"]["important_instructions"], ["Bring the hall ticket."])

    def test_duplicates_are_removed_within_and_across_fields(self) -> None:
        output = {"fields": {"document_title": "Scholarship notice",
                             "application_procedure": ["Apply online.", "apply online"],
                             "important_instructions": ["Apply online", "Keep a printout."]},
                  "additional_fields": [{"label": "Title", "value": "Scholarship notice"}, {"label": "Document title", "value": "Other"},
                                        {"label": "Helpline", "value": "1800-11-0001; Page: 4"}]}
        processing.clean_fields(output)
        self.assertEqual(output["fields"]["application_procedure"], ["Apply online."])
        self.assertEqual(output["fields"]["important_instructions"], ["Keep a printout."])
        self.assertEqual(output["additional_fields"], [{"label": "Helpline", "value": "1800-11-0001", "pages": [4]}])


class SummaryTests(unittest.TestCase):
    def test_structured_summary_is_normalised_and_empty_parts_dropped(self) -> None:
        output = {"summary": {"overview": "The board explains the new language rule [Page 1].",
                              "key_points": ["Two of the three languages must be Indian.", "Two of the three languages must be Indian.", ""],
                              "important_dates": [{"date": "1 July 2026", "event": "The rule starts"}],
                              "applies_to": ["Schools affiliated to the board"], "actions": []}}
        processing.clean_fields(output)
        self.assertEqual(output["summary"], "The board explains the new language rule.")
        self.assertEqual(output["main_points"], ["Two of the three languages must be Indian."])
        self.assertEqual(output["summary_dates"], [{"date": "1 July 2026", "event": "The rule starts"}])
        self.assertEqual(output["applies_to"], "Schools affiliated to the board")
        self.assertEqual(output["actions"], [])
        text = processing.summary_text(output)
        self.assertIn("1 July 2026 — The rule starts", text)
        self.assertNotIn("{", text)

    def test_json_fragments_absence_notes_and_near_duplicates_are_removed(self) -> None:
        output = {"summary": {"overview": "The department sanctions funds for the study circle scheme.",
                              "key_points": ["Funds are drawn by the concerned DDO through a grants-in-aid bill.",
                                             "Funds are drawn by the concerned DDO through a grants-in-aid bill only.",
                                             'Important dates:[{"',
                                             "The commissioner must act as per Finance Department instructions. Dates: [{\"date",
                                             "No specific eligibility criteria or application procedures were detailed in this administrative order.",
                                             "The document does not mention any deadline.",
                                             "No fee is charged to students of the scheme.",
                                             "important_dates”:[]",
                                             "The session starts in June 2026. The document does not specify a submission deadline."],
                              "actions": ["Page 2: Submit the bill to the treasury."]}}
        processing.clean_fields(output)
        # Kept once: near-identical points; dropped: a bare JSON fragment and notes
        # about what the document does not say; a real "No ..." rule is kept.
        self.assertEqual(output["main_points"], ["Funds are drawn by the concerned DDO through a grants-in-aid bill.",
                                                 "The commissioner must act as per Finance Department instructions.",
                                                 "No fee is charged to students of the scheme.",
                                                 "The session starts in June 2026."])
        self.assertEqual(output["actions"], ["Submit the bill to the treasury."])

    def test_summary_keys_written_into_the_key_points_are_moved_to_their_part(self) -> None:
        output = {"summary": {"overview": "The department sanctions funds for the study circle.",
                              "key_points": ["Funds are drawn through a grants-in-aid bill.", "applies_to",
                                             "Treasury officers handling the scheme funds", "actions",
                                             "Overview: the order covers the third quarter only."]}}
        processing.clean_fields(output)
        self.assertEqual(output["main_points"], ["Funds are drawn through a grants-in-aid bill."])
        self.assertEqual(output["applies_to"], "Treasury officers handling the scheme funds")
        self.assertEqual(output["actions"], ["Overview: the order covers the third quarter only."])

        output = {"summary": {"overview": "A circular inviting applications on deputation.",
                              "key_points": ["Applicants need eight years of service.",
                                             'applies_to":"Government officers eligible for deputation","actions']}}
        processing.clean_fields(output)
        self.assertEqual(output["main_points"], ["Applicants need eight years of service."])
        self.assertEqual(output["applies_to"], "Government officers eligible for deputation")

    def test_source_pages_outside_the_document_are_replaced(self) -> None:
        output = {"fields": {"amount": "Rs 25,00,00,000", "document_type": "Government Order"},
                  "field_sources": {"amount": [1, 4], "document_type": [0]},
                  "additional_fields": [{"label": "Scheme", "value": "Telangana Study Circle", "pages": [3]}]}
        pages = ["Government Order. Sanction of Rs 25,00,00,000 for the Telangana Study Circle."]
        processing.clean_fields(output)
        processing.add_missing_sources(output, pages)
        self.assertEqual(output["field_sources"], {"amount": [1], "document_type": [1]})
        self.assertEqual(output["additional_fields"][0]["pages"], [1])

    def test_summary_date_not_printed_in_the_document_is_removed(self) -> None:
        output = {"summary": {"overview": "A notice about the new session.", "key_points": ["Classes begin soon."],
                              "important_dates": [{"date": "1 July 2026", "event": "Classes begin"}, {"date": "5 May 2031", "event": "Invented"}]}}
        processing.clean_fields(output)
        processing.ground_numbers(output, ["Classes begin on 1 July 2026."])
        self.assertEqual([item["date"] for item in output["summary_dates"]], ["1 July 2026"])
        self.assertIn({"field": "Summary date", "value": "5 May 2031 — Invented"}, output["ungrounded_values"])

    def test_summary_length_adapts_to_document_size(self) -> None:
        self.assertEqual(processing.summary_length("word " * 300)["words"], "50 to 80")
        self.assertEqual(processing.summary_length("word " * 2500)["words"], "70 to 110")
        self.assertEqual(processing.summary_length("word " * 20000)["words"], "100 to 150")

    @patch("document_processing._request_analysis_once")
    def test_combined_call_asks_for_the_documents_summary_length(self, once) -> None:
        once.return_value = reply(fields={"document_title": "Notice"}, summary={"overview": "A short notice about fees.", "key_points": ["Pay the fee."]})
        processing.analyze_document_text(["NOTICE. Students must pay the fee."])
        self.assertEqual(once.call_args.kwargs["task_values"], {"words": "50 to 80", "points": "3"})

    @patch("document_processing._request_analysis")
    def test_summary_that_stays_empty_is_marked_failed(self, request_analysis) -> None:
        request_analysis.side_effect = [reply(fields={"issuing_department": "CBSE"}, summary=""), reply(_mode="summary", summary="")]
        result = processing.analyze_document_text(["PRESS RELEASE. CBSE issues guidelines."])
        self.assertEqual(result["summary_status"], "failed")
        self.assertEqual(result["fields_status"], "ok")
        self.assertEqual(result["fields"]["issuing_department"], "CBSE")

    @patch("document_processing._request_analysis")
    def test_model_calls_are_recorded_with_their_measured_time(self, request_analysis) -> None:
        request_analysis.return_value = reply(fields={"document_title": "Notice"}, summary={"overview": "A short notice about fees.", "key_points": ["Pay."]},
                                              _usage={"prompt_eval_count": 900, "eval_count": 200})
        result = processing.analyze_document_text(["NOTICE. Students must pay the fee."])
        self.assertEqual(result["model_calls"], [{"mode": "document", "seconds": 1.0, "model": "test-model", "prompt_eval_count": 900, "eval_count": 200}])
        self.assertEqual(result["analysis_path"], "combined")


class ResumeAndPathTests(unittest.TestCase):
    def long_pages(self) -> list[str]:
        return [f"Page {number} explains rule {number} of the scheme in detail. " * 40 for number in range(1, 40)]

    @patch("document_processing._request_analysis")
    def test_failed_long_analysis_resumes_without_repeating_finished_calls(self, request_analysis) -> None:
        pages = self.long_pages()
        sections = processing.plan_summary_sections(pages)
        self.assertGreater(len(sections), 2)
        fields = reply(_mode="fields", fields={"document_title": "Scheme rules"})
        section = reply(_mode="section", summary="This section explains several rules of the scheme.")
        merged = reply(_mode="reduce", summary={"overview": "The scheme has many rules.", "key_points": ["Rules apply to all."]})
        request_analysis.side_effect = [fields, section, RuntimeError("timed out")]
        partial: dict = {}
        with self.assertRaises(RuntimeError):
            processing.analyze_document_text(pages, partial=partial)
        self.assertEqual(len(partial["sections"]), 1)
        request_analysis.reset_mock()
        request_analysis.side_effect = [section] * (len(sections) - 1) + [merged]
        result = processing.analyze_document_text(pages, partial=partial)
        modes = [call.kwargs.get("mode") for call in request_analysis.call_args_list]
        self.assertEqual(modes, ["section"] * (len(sections) - 1) + ["reduce"])  # no second field extraction, no repeated section
        self.assertEqual(result["summary"], "The scheme has many rules.")
        self.assertEqual(result["analysis_path"], "sections")

    def test_indian_language_documents_are_budgeted_by_tokens(self) -> None:
        hindi_page = "यह परिपत्र सभी विद्यालयों के लिए है और इसमें नियम दिए गए हैं। " * 60  # about 3,800 characters
        self.assertTrue(processing.fits_single_call([hindi_page]))
        self.assertGreater(processing.estimate_tokens(hindi_page), len(hindi_page) / processing.LATIN_CHARS_PER_TOKEN)

    @patch.dict(os.environ, {"HF_CHAT_COMPLETIONS_URL": ""})
    @patch("document_processing.requests.Session.post")
    def test_ollama_reply_is_decoded_against_the_output_schema(self, post) -> None:
        response = Mock(status_code=200, ok=True)
        response.json.return_value = {"model": "qwen3.5:4b", "prompt_eval_count": 1200, "eval_count": 300,
                                      "message": {"content": '{"document_type":"Notice","fields":{},"field_sources":{},"additional_fields":[],'
                                                             '"summary":{"overview":"A notice for students.","key_points":["Pay the fee."]}}'}}
        post.return_value = response
        result = processing._request_analysis([], "[Page 1]\nNotice text", "", mode="document", task_values={"words": "50 to 80", "points": "3"})
        body = post.call_args.kwargs["json"]
        self.assertIsInstance(body["format"], dict)
        self.assertIn("summary", body["format"]["required"])
        self.assertIn("50 to 80", body["messages"][1]["content"])
        self.assertEqual(result["_usage"], {"prompt_eval_count": 1200, "eval_count": 300})


class FakeTokenizer:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, batch, **kwargs):
        import torch

        self.batches.append(list(batch))
        return {"input_ids": torch.tensor([[index] for index in range(len(batch))])}

    def batch_decode(self, generated, **kwargs):
        # Echo the source sentence, so the order of the output can be checked.
        return [self.batches[-1][int(row[0])].split(" ", 2)[-1] for row in generated]


class FakeModel:
    def generate(self, input_ids, **kwargs):
        return input_ids


class FakeProcessor:
    def preprocess_batch(self, sentences, src_lang, tgt_lang, protect=True):
        return [f"{src_lang} {tgt_lang} {sentence}" for sentence in sentences]

    def postprocess_batch(self, sentences, lang):
        return sentences


class TranslationSpeedTests(unittest.TestCase):
    def test_long_sentences_are_split_instead_of_truncated(self) -> None:
        sentence = ", ".join(f"clause number {index} of a very long rule" for index in range(40))
        pieces = translation._split_long(sentence, 120)
        self.assertTrue(all(len(piece) <= 120 for piece in pieces))
        self.assertEqual(" ".join(pieces).replace(" ", ""), sentence.replace(" ", ""))

    def test_batches_are_sorted_by_length_and_order_is_restored(self) -> None:
        tokenizer = FakeTokenizer()
        text = "\n".join(["Short one.", "This is a much longer sentence than the others in the list.", "Mid length line here.", "Tiny."])
        with patch("translation.load_indictrans2", return_value=(FakeModel(), tokenizer, FakeProcessor())), \
             patch("translation.INDICTRANS2_BATCH", 2):
            output = translation._translate_indictrans2(text, "kan_Knda", None)
        self.assertEqual(output, text)  # every line back in its place
        lengths = [[len(sentence) for sentence in batch] for batch in tokenizer.batches]
        self.assertLessEqual(max(lengths[0]), min(lengths[1]))  # shortest sentences batched together

    def test_localize_sends_only_reader_text_and_keeps_dates_exact(self) -> None:
        analysis = {"summary": "A notice.", "main_points": ["Pay the fee."], "summary_dates": [{"date": "30-11-2026", "event": "Last date"}],
                    "applies_to": "All students", "actions": ["Apply online."],
                    "fields": {"reference_number": "No. 12/2026", "document_title": "Fee notice"}, "additional_fields": []}
        sent: list[str] = []

        def fake_segments(segments, language, progress_callback=None):
            sent.extend(segments)
            return {"texts": [f"KN:{segment}" for segment in segments], "service": "IndicTrans2", "is_indictrans2": True, "cached": 0}

        with patch("localization.translate_segments", side_effect=fake_segments):
            result = localize(analysis, "Kannada")
        self.assertEqual(result["summary_dates"], [{"date": "30-11-2026", "event": "KN:Last date"}])
        self.assertEqual(result["applies_to"], "KN:All students")
        self.assertEqual(result["actions"], ["KN:Apply online."])
        self.assertEqual(result["fields"]["reference_number"], "No. 12/2026")
        self.assertNotIn("30-11-2026", sent)
        self.assertNotIn("reference_number", " ".join(sent))


if __name__ == "__main__":
    unittest.main()
