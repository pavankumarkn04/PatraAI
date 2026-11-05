"""Scoring rules for field extraction and question answering.

Accepted as equal: differences in case, spacing and punctuation; the same date
written differently (26.05.2026 = 26 May 2026 = 26 मई 2026); the same rupee
amount written differently (Rs 1,300 = ₹1300/-); and a prediction that contains
the expected value as whole words with exactly the same numbers ("Central Hindi
Directorate, New Delhi" for "Central Hindi Directorate").

Longer verified answers (five or more content words) also accept a paraphrase
whose word overlap with the verified answer has F1 >= 0.7 and recall >= 0.75
(at least three quarters of the verified content words appear), provided every
number in the verified answer appears in the prediction.

Never accepted: a different number, date, amount or reference; a partial number
match (1300 vs 13000); a missing value; or an abstention.

Expected values may be a string, a list (every item must appear), or
{"any_of": [...]} listing verified alternatives, for example the same office
name in Hindi and in English.
"""

from __future__ import annotations

import re
from typing import Any

from evaluation.evaluate import normalize

_INDIC_DIGITS = {block + value: str(value) for block in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66) for value in range(10)}
_HINDI_MONTHS = {
    "जनवरी": "January", "फरवरी": "February", "फ़रवरी": "February", "मार्च": "March", "अप्रैल": "April", "मई": "May",
    "जून": "June", "जुलाई": "July", "अगस्त": "August", "सितंबर": "September", "सितम्बर": "September",
    "अक्टूबर": "October", "अक्तूबर": "October", "नवंबर": "November", "नवम्बर": "November", "दिसंबर": "December", "दिसम्बर": "December",
}
ABSTAIN = re.compile(r"does not (?:provide|contain|state|mention|specify)|not enough information|cannot be determined|incomplete answer", re.I)


def flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(flatten(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    return "" if value is None else str(value)


def alternatives(expected: Any) -> list[Any]:
    if isinstance(expected, dict) and "any_of" in expected:
        return list(expected["any_of"])
    return [expected]


def _prepare(value: Any) -> str:
    text = flatten(value).translate(_INDIC_DIGITS)
    for hindi, english in _HINDI_MONTHS.items():
        text = text.replace(hindi, english)
    return text


def dates(value: Any) -> set[str]:
    from document_processing import _normalized_dates

    return _normalized_dates(_prepare(value))


def amounts(value: Any) -> set[float]:
    found = re.findall(r"(?:rs\.?|inr|₹|रु\.?|ரூ\.?|రూ\.?|രൂ\.?|ರೂ\.?)\s*([\d][\d,]*(?:\.\d+)?)|([\d][\d,]*(?:\.\d+)?)\s*(?:/-|rupees?|रुपये|രൂപ)", _prepare(value), re.I)
    return {float((first or second).replace(",", "")) for first, second in found}


def numbers(value: Any) -> set[str]:
    """Numbers as integers-as-strings, ignoring thousands separators and leading zeros."""
    text = re.sub(r"(?<=\d),(?=\d)", "", _prepare(value))
    return {str(int(token)) for token in re.findall(r"\d+", text)}


def _contains_words(expected: str, predicted: str) -> bool:
    return bool(expected) and f" {expected} " in f" {predicted} "


def _single_field_match(expected: Any, predicted: Any) -> bool:
    predicted_text = normalize(_prepare(predicted))
    if not predicted_text:
        return False
    if isinstance(expected, list):
        return bool(expected) and all(_contains_words(normalize(_prepare(item)), predicted_text) for item in expected)
    expected_text = normalize(_prepare(expected))
    if expected_text == predicted_text:
        return True
    expected_dates = dates(expected)
    if expected_dates and expected_dates == dates(predicted):
        return True
    expected_amounts = amounts(expected)
    if expected_amounts and expected_amounts == amounts(predicted):
        return True
    return _contains_words(expected_text, predicted_text) and numbers(expected) == numbers(predicted)


def field_matches(expected: Any, predicted: Any) -> bool:
    return any(_single_field_match(option, predicted) for option in alternatives(expected))


def _single_answer_match(expected: Any, predicted: str) -> bool:
    expected_text, predicted_text = normalize(_prepare(expected)), normalize(_prepare(predicted))
    if not expected_text or not predicted_text:
        return False
    if _contains_words(expected_text, predicted_text):
        return True
    expected_dates = dates(expected)
    if expected_dates and expected_dates <= dates(predicted):
        return True
    expected_amounts = amounts(expected)
    if expected_amounts and expected_amounts <= amounts(predicted):
        return True
    # Longer verified answers (five or more content words): accept a paraphrase
    # with word-overlap F1 >= 0.7 and recall >= 0.75, provided every number is
    # the same. Recall guards against answers that share only generic words.
    if len(_content_words(str(expected))) >= 5:
        f1, recall = token_overlap(str(expected), predicted)
        return f1 >= 0.7 and recall >= 0.75 and numbers(expected) <= numbers(predicted)
    return False


_STOPWORDS = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be", "by", "on", "with", "as", "at", "it", "this", "that", "from", "will", "shall", "must"}


def _content_words(text: str) -> list[str]:
    return [word for word in normalize(_prepare(text)).split() if word not in _STOPWORDS]


def token_overlap(expected: str, predicted: str) -> tuple[float, float]:
    """(F1, recall) of content-word overlap between verified and predicted answers (SQuAD-style)."""
    expected_words, predicted_words = _content_words(expected), _content_words(predicted)
    if not expected_words or not predicted_words:
        return 0.0, 0.0
    common = sum(min(expected_words.count(word), predicted_words.count(word)) for word in set(expected_words))
    if not common:
        return 0.0, 0.0
    precision, recall = common / len(predicted_words), common / len(expected_words)
    return 2 * precision * recall / (precision + recall), recall


def answer_matches(expected: Any, predicted: str) -> bool:
    if not predicted or ABSTAIN.search(predicted):
        return False
    return any(_single_answer_match(option, predicted) for option in alternatives(expected))


def field_error_category(expected: Any, predicted: Any) -> str:
    if not normalize(_prepare(predicted)):
        return "Missing extracted field"
    return "Incorrect extracted value"


def answer_error_category(expected: Any, predicted: str) -> str:
    if not predicted.strip() or ABSTAIN.search(predicted):
        return "Answer not found in document (model abstained)"
    return "Incorrect answer"


def display(value: Any) -> str:
    if isinstance(value, dict) and "any_of" in value:
        return " | ".join(display(item) for item in value["any_of"])
    if isinstance(value, list):
        return "; ".join(display(item) for item in value)
    return "" if value is None else str(value)
