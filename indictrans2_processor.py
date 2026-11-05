"""Pure-Python port of IndicTransToolkit's IndicProcessor (v1.1.1).

IndicTrans2 Hugging Face models expect inputs that are normalized,
tokenized, tagged with source and target language codes and, for Indic
sources, transliterated into Devanagari. The models also produce every Indic
output in Devanagari, which must be transliterated back into the target
script. The official toolkit is a Cython extension that needs a C compiler on
Windows; this port keeps its logic so the app installs with pip alone.

Original: https://github.com/VarunGumma/IndicTransToolkit (MIT License,
Copyright (c) Varun Gumma). Only inference-time behaviour is ported.
"""

from __future__ import annotations

import regex as re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_detokenize, indic_tokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

FLORES_TO_ISO = {
    "asm_Beng": "as", "awa_Deva": "hi", "ben_Beng": "bn", "bho_Deva": "hi", "brx_Deva": "hi",
    "doi_Deva": "hi", "eng_Latn": "en", "gom_Deva": "kK", "gon_Deva": "hi", "guj_Gujr": "gu",
    "hin_Deva": "hi", "hne_Deva": "hi", "kan_Knda": "kn", "kas_Arab": "ur", "kas_Deva": "hi",
    "kha_Latn": "en", "lus_Latn": "en", "mag_Deva": "hi", "mai_Deva": "hi", "mal_Mlym": "ml",
    "mar_Deva": "mr", "mni_Beng": "bn", "mni_Mtei": "hi", "npi_Deva": "ne", "ory_Orya": "or",
    "pan_Guru": "pa", "san_Deva": "hi", "sat_Olck": "or", "snd_Arab": "ur", "snd_Deva": "hi",
    "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur", "unr_Deva": "hi",
}

# Indic digits (Bengali, Gujarati, Kannada, Devanagari, Arabic, Meetei, Oriya,
# Gurmukhi, Ol Chiki, Persian, Telugu) to ASCII.
_DIGIT_BLOCKS = (0x09E6, 0x0AE6, 0x0CE6, 0x0966, 0x0660, 0xABF0, 0x0B66, 0x0A66, 0x1C50, 0x06F0, 0x0C66)
DIGITS = {block + value: str(value) for block in _DIGIT_BLOCKS for value in range(10)}

_MULTISPACE = re.compile(r"[ ]{2,}")
_DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
_DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
_DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
_END_BRACKET_SPACE_PUNC = re.compile(r"\) ([\.!:?;,])")
_URL = re.compile(r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b")
_NUMERAL = re.compile(r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
_OTHER = re.compile(r"[A-Za-z0-9]*[#|@]\w+")
_LEFTOVER_PLACEHOLDER = re.compile(r"[<\[]\s*[^<>\[\]\d]{0,20}?\s*(\d{1,3})\s*[>\]]")
_PUNC_REPLACEMENTS = [
    (re.compile(r"\r"), ""), (re.compile(r"\(\s*"), "("), (re.compile(r"\s*\)"), ")"),
    (re.compile(r"\s:\s?"), ":"), (re.compile(r"\s;\s?"), ";"), (re.compile(r"[`´‘‚’]"), "'"),
    (re.compile(r"[„“”«»]"), '"'), (re.compile(r"[–—]"), "-"), (re.compile(r"\.\.\."), "..."),
    (re.compile(r" %"), "%"), (re.compile(r"nº "), "nº "), (re.compile(r" ºC"), " ºC"),
    (re.compile(r" [?!;]"), lambda m: m.group(0).strip()), (re.compile(r", "), ", "),
]
# Spellings the model sometimes uses when it translates the "ID" placeholder.
_INDIC_FAILURE_CASES = [
    "آی ڈی ", "ꯑꯥꯏꯗꯤ", "आईडी", "आई . डी . ", "आई . डी .", "आई. डी. ", "आई. डी.", "आय. डी. ",
    "आय. डी.", "आय . डी . ", "आय . डी .आइ . डी . ", "आइ . डी .", "आइ. डी. ", "आइ. डी.", "ऐटि",
    "آئی ڈی ", "ᱟᱭᱰᱤ ᱾", "आयडी", "ऐडि", "आइडि", "ᱟᱭᱰᱤ",
]


class IndicProcessor:
    """Prepare IndicTrans2 inputs and restore its outputs (inference only)."""

    def __init__(self) -> None:
        self._en_tok = MosesTokenizer(lang="en")
        self._en_normalizer = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")
        self._placeholder_maps: list[dict[str, str]] = []

    @staticmethod
    def _punc_norm(text: str) -> str:
        for pattern, replacement in _PUNC_REPLACEMENTS:
            text = pattern.sub(replacement, text)
        text = _MULTISPACE.sub(" ", text)
        text = _END_BRACKET_SPACE_PUNC.sub(r")\1", text)
        text = _DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = _DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = _DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    @staticmethod
    def _wrap_with_placeholders(text: str) -> tuple[str, dict[str, str]]:
        """Protect emails, URLs and numbers from being translated."""
        serial = 1
        mapping: dict[str, str] = {}
        for pattern in (_EMAIL, _URL, _NUMERAL, _OTHER):
            for match in set(pattern.findall(text)):
                if pattern is _URL and len(match.replace(".", "")) < 4:
                    continue
                if pattern is _NUMERAL and len(match.replace(" ", "").replace(".", "").replace(":", "")) < 4:
                    continue
                for tag in ("ID", "id"):
                    for form in ("<{t}{n}>", "< {t}{n} >", "[{t}{n}]", "[ {t}{n} ]", "[{t} {n}]", "<{t}{n}]", "< {t}{n}]", "<{t}{n} ]"):
                        mapping[form.format(t=tag, n=serial)] = match
                for case in _INDIC_FAILURE_CASES:
                    for form in ("<{c}{n}>", "< {c}{n} >", "< {c} {n} >", "<{c} {n}]", "< {c} {n} ]", "[{c}{n}]", "[{c} {n}]", "[ {c}{n} ]", "[ {c} {n} ]", "{c} {n}", "{c}{n}"):
                        mapping[form.format(c=case, n=serial)] = match
                text = text.replace(match, f"<ID{serial}>")
                serial += 1
        text = re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]")
        return text, mapping

    def preprocess_batch(self, batch: list[str], src_lang: str, tgt_lang: str, protect: bool = True) -> list[str]:
        """protect=False leaves numbers and emails as plain text (used for a retry
        when the model dropped a placeholder)."""
        iso = FLORES_TO_ISO.get(src_lang, "hi")
        normalizer = None if src_lang == "eng_Latn" else IndicNormalizerFactory().get_normalizer(iso)
        transliterate = src_lang.split("_")[1] not in {"Arab", "Aran", "Olck", "Mtei", "Latn"}
        processed = []
        for sentence in batch:
            sentence = self._punc_norm(sentence).translate(DIGITS)
            sentence, mapping = self._wrap_with_placeholders(sentence) if protect else (sentence, {})
            self._placeholder_maps.append(mapping)
            if iso == "en":
                tokens = self._en_tok.tokenize(self._en_normalizer.normalize(sentence.strip()), escape=False)
                sentence = " ".join(tokens)
            else:
                sentence = " ".join(indic_tokenize.trivial_tokenize(normalizer.normalize(sentence.strip()), iso))
                if transliterate:
                    sentence = UnicodeIndicTransliterator.transliterate(sentence, iso, "hi").replace(" ् ", "्")
            processed.append(f"{src_lang} {tgt_lang} {sentence.strip()}")
        return processed

    def postprocess_batch(self, sentences: list[str], lang: str) -> list[str]:
        maps, self._placeholder_maps = self._placeholder_maps[:len(sentences)], self._placeholder_maps[len(sentences):]
        iso = FLORES_TO_ISO.get(lang, "hi")
        lang_code, script = lang.split("_", 1)
        results = []
        for sentence, mapping in zip(sentences, maps):
            if script in {"Arab", "Aran"}:
                sentence = sentence.replace(" ؟", "؟").replace(" ۔", "۔").replace(" ،", "،").replace("ٮ۪", "ؠ")
            if lang_code == "ory":
                sentence = sentence.replace("ଯ଼", "ୟ")
            for placeholder, value in mapping.items():
                sentence = sentence.replace(placeholder, value)
            # The model sometimes rewrites "<ID1>" in a spelling the list above
            # does not cover (for example "<ऐ. डी. 1>"); restore any bracketed
            # token that ends in a known placeholder number.
            sentence = _LEFTOVER_PLACEHOLDER.sub(
                lambda match: mapping.get(f"<ID{match.group(1)}>", match.group(0)), sentence)
            if lang == "eng_Latn":
                results.append(self._en_detok.detokenize(sentence.split(" ")))
            else:
                results.append(indic_detokenize.trivial_detokenize(UnicodeIndicTransliterator.transliterate(sentence, "hi", iso), iso))
        return results
