"""IndicTrans2 routing and the ported IndicProcessor (model replaced by a test double)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import translation
from indictrans2_processor import IndicProcessor


class FakeTokenizer:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def __call__(self, batch, **kwargs):
        self.inputs.extend(batch)
        return {"input_ids": torch.zeros((len(batch), 3), dtype=torch.long)}

    def batch_decode(self, generated, **kwargs):
        # The real model writes Indic output in Devanagari with placeholders.
        return ["शुल्क <ID1> है ।" for _ in generated]


class FakeModel:
    def generate(self, input_ids, **kwargs):
        return torch.zeros((input_ids.shape[0], 3), dtype=torch.long)


class IndicTrans2Tests(unittest.TestCase):
    def test_processor_tags_protects_numbers_and_restores_target_script(self) -> None:
        processor = IndicProcessor()
        prepared = processor.preprocess_batch(["Pay Rs 1300 by 30/11/2026."], "eng_Latn", "kan_Knda")
        self.assertTrue(prepared[0].startswith("eng_Latn kan_Knda "))
        self.assertNotIn("30/11/2026", prepared[0])
        restored = processor.postprocess_batch(["शुल्क <ID1> है ।"], "kan_Knda")[0]
        self.assertIn("30/11/2026", restored)
        self.assertIn("ಶುಲ್ಕ", restored)  # Kannada script, not Devanagari

    def test_colon_rendered_as_visarga_is_restored_only_where_source_had_colons(self) -> None:
        self.assertEqual(translation._restore_colons("Application fee: Rs 1300", "ಅರ್ಜಿ ಶುಲ್ಕಃ 1300 ರೂ."), "ಅರ್ಜಿ ಶುಲ್ಕ: 1300 ರೂ.")
        self.assertEqual(translation._restore_colons("Contact: x", "தொடர்புஃ x"), "தொடர்பு: x")
        # No colon in the source: a genuine word-final visarga stays.
        self.assertEqual(translation._restore_colons("Again.", "पुनः ।"), "पुनः ।")
        # Visarga inside a word is never touched.
        self.assertEqual(translation._restore_colons("Grief: x", "ದುಃಖಃ x"), "ದುಃಖ: x")

    def test_respelled_placeholder_is_still_restored(self) -> None:
        processor = IndicProcessor()
        processor.preprocess_batch(["Students in Class X in 2026-27 are not affected."], "eng_Latn", "kan_Knda")
        restored = processor.postprocess_batch(["<ऐ. डी. 1> में कक्षा दस के छात्र प्रभावित नहीं होते हैं।"], "kan_Knda")[0]
        self.assertTrue(restored.startswith("2026-27"))
        self.assertNotIn("<", restored)

    def test_local_indictrans2_is_used_when_available(self) -> None:
        tokenizer = FakeTokenizer()
        with patch("translation.local_indictrans2_status", return_value=(True, "downloaded")), \
             patch("translation.load_indictrans2", return_value=(FakeModel(), tokenizer, IndicProcessor())):
            result = translation.translate_with_details("Pay by 30/11/2026.\n\nSecond line.", "Kannada")
        self.assertTrue(result["is_indictrans2"])
        self.assertIn("IndicTrans2", result["service"])
        self.assertTrue(all(item.startswith("eng_Latn kan_Knda") for item in tokenizer.inputs))
        self.assertEqual(len(result["text"].split("\n")), 3)  # line structure kept

    def test_dropped_number_is_retranslated_so_no_fact_is_lost(self) -> None:
        class DroppingTokenizer(FakeTokenizer):
            def batch_decode(self, generated, **kwargs):
                # Like the real model: with a placeholder the year is dropped,
                # with plain digits it is copied.
                return ["शुल्क 2026-27 है ।" if "2026-27" in self.inputs[-1] else "शुल्क है ।" for _ in generated]

        tokenizer = DroppingTokenizer()
        with patch("translation.local_indictrans2_status", return_value=(True, "downloaded")), \
             patch("translation.load_indictrans2", return_value=(FakeModel(), tokenizer, IndicProcessor())):
            result = translation.translate_with_details("The session is 2026-27.", "Kannada")
        self.assertIn("2026-27", result["text"])
        self.assertEqual(len(tokenizer.inputs), 2)  # first pass + one retry

    def test_fallback_is_reported_with_reason(self) -> None:
        with patch.dict("os.environ", {"INDIC_TRANS2_URL": ""}), \
             patch("translation.local_indictrans2_status", return_value=(False, "needs a token")), \
             patch("translation._translate_legacy", return_value="fallback text"):
            result = translation.translate_with_details("Hello.", "Hindi")
        self.assertFalse(result["is_indictrans2"])
        self.assertEqual(result["fallback_reason"], "needs a token")
        self.assertEqual(result["text"], "fallback text")


if __name__ == "__main__":
    unittest.main()
