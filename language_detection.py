"""Language detection for documents, questions and generated answers.

Native-script text is identified by Unicode block. Romanised Indian-language
questions ("ee dakhale yaaru horadisiddare?") are identified with small
lexicons of distinctive function words, which is reliable for questions and
needs no extra dependency or model call.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

SUPPORTED_LANGUAGES = ("English", "Kannada", "Hindi", "Tamil", "Telugu", "Malayalam")

# Unicode script ranges. Devanagari is reported as Hindi.
SCRIPTS = (
    ("Hindi", 0x0900, 0x097F), ("Kannada", 0x0C80, 0x0CFF), ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F), ("Malayalam", 0x0D00, 0x0D7F), ("Bengali", 0x0980, 0x09FF),
    ("Gujarati", 0x0A80, 0x0AFF), ("Punjabi", 0x0A00, 0x0A7F), ("Odia", 0x0B00, 0x0B7F),
)

# Distinctive words of romanised questions. Words shared by two languages
# (for example "illa") are listed under both and resolved by the other words.
_ROMANISED = {
    "Hindi": {
        "kya", "kaun", "kab", "kahan", "kaha", "kaise", "kitna", "kitni", "kitne", "hai", "hain", "kiya", "kis", "kisne",
        "kise", "kiske", "kaunsa", "kaunse", "kyon", "kyu", "kyun", "aur", "nahi", "nahin", "batao", "bataiye", "bataye",
        "dastavez", "dastavej", "liye", "karna", "karne", "chahiye", "tarikh", "tareekh", "yojana", "patrata", "jari",
        "sakta", "sakte", "hoga", "wala", "wali", "mein", "kaagaz", "zaroori", "jaruri", "avashyak", "antim",
    },
    "Kannada": {
        "yaaru", "yaru", "enu", "yenu", "yavaga", "yaavaga", "elli", "hege", "eshtu", "yake", "yaake", "ide", "ideya",
        "idhe", "illa", "beku", "bekagutte", "maadi", "madi", "dakhale", "dakhaleyannu", "dakhaleya", "hesaru", "dinanka",
        "koneya", "arhate", "nanage", "idu", "yavudu", "yaavudu", "horadisiddare", "horadisidavaru", "horadisalagide",
        "kodi", "heli", "helu", "sigutte", "agutte", "mattu", "athava", "yavuvu", "yaavuvu", "dakhalegalu", "suchanegalu",
    },
    "Tamil": {
        "yaar", "enna", "eppo", "eppothu", "eppodhu", "enge", "engu", "eppadi", "evvalavu", "irukku", "irukkiradhu",
        "illai", "vendum", "venum", "aavanam", "aavanathai", "aavanangal", "thethi", "theathi", "thakuthi",
        "vazhangiyavar", "vazhangiyathu", "veliyittavar", "veliyittathu", "sollunga", "enakku", "idhu", "indha", "intha",
        "endra", "pannunga", "yaarukku", "ennenna", "kadaisi", "mukkiyamana",
    },
    "Telugu": {
        "evaru", "emi", "enti", "eppudu", "ekkada", "ela", "enduku", "undi", "ledu", "kavali", "cheyandi", "cheppandi",
        "idi", "tedi", "thedi", "arhata", "chesaru", "chesindi", "chesina", "gurinchi", "patralu", "jaari", "evvaru",
        "emiti", "ivvandi", "avasaramaina", "mukhyamaina", "chivari",
    },
    "Malayalam": {
        "aaru", "aranu", "aaraanu", "enthu", "enthanu", "enthaanu", "eppol", "eppozhanu", "evide", "engane", "ethra",
        "aanu", "venam", "vendathu", "cheyyuka", "parayuka", "rekha", "rekhakal", "theeyathi", "yogyatha",
        "purappeduvichathu", "purappeduvichu", "ithu", "njan", "enikku", "ulla", "undu", "illa", "aavashyamaya", "avasana",
    },
}
_ENGLISH_WORDS = {
    "what", "who", "when", "where", "which", "why", "how", "is", "are", "was", "were", "the", "this", "that", "these",
    "document", "documents", "issued", "date", "dates", "required", "eligibility", "about", "does", "do", "of", "for",
    "to", "in", "and", "or", "a", "an", "it", "be", "can", "i", "my", "me", "please", "tell", "list", "give", "fee",
    "fees", "amount", "last", "important", "instructions", "apply", "application", "notice", "circular", "there", "any",
}


def script_of(char: str) -> str:
    """'English' for Latin letters, the language of an Indian script, or 'Other'."""
    code = ord(char)
    if code < 0x0250:
        return "English"
    return next((name for name, low, high in SCRIPTS if low <= code <= high), "Other")


def script_shares(text: str) -> dict[str, float]:
    """Share of letters in each script, largest first.

    Vowel signs and viramas of Indian scripts are combining marks, not
    letters, to str.isalpha(); they are counted so Indian-language text is
    not under-weighted against English names inside it.
    """
    counts: dict[str, int] = {}
    for char in text:
        if char.isalpha() or (ord(char) >= 0x0900 and unicodedata.category(char) in ("Mn", "Mc")):
            name = script_of(char)
            counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    return {name: count / total for name, count in sorted(counts.items(), key=lambda item: -item[1])} if total else {}


def detect_language(page_texts: list[str]) -> dict[str, Any]:
    """Dominant document language by script, with any other language above 15% of letters."""
    shares = script_shares("".join(page_texts))
    if not shares:
        return {"language": "Unknown", "shares": {}, "multilingual": False, "supported": False}
    # A few English words (headers, codes, URLs) are common in Indian-language
    # documents, so an Indian script wins if it has a substantial share.
    indian = [(name, share) for name, share in shares.items() if name not in {"English", "Other"} and share >= 0.25]
    dominant = indian[0][0] if indian else next(iter(shares))
    others = [name for name, share in shares.items() if name != dominant and name != "Other" and share >= 0.15]
    return {"language": dominant, "shares": shares, "multilingual": bool(others), "others": others,
            "supported": dominant in SUPPORTED_LANGUAGES}


def detect_question_language(question: str) -> dict[str, Any]:
    """Language of one question: native script first, then romanised wording.

    Returns {"language", "script" ('native' | 'romanised' | 'latin' | 'none'),
    "reliable"}. A question with no letters, or a very short Latin question
    with no recognisable words, is marked unreliable so the caller can keep
    the language of the previous question.
    """
    shares = script_shares(question)
    indian = {name: share for name, share in shares.items() if name in SUPPORTED_LANGUAGES and name != "English"}
    if indian:
        language, share = max(indian.items(), key=lambda item: item[1])
        if share >= 0.3:
            return {"language": language, "script": "native", "reliable": True}
    if not shares:
        return {"language": "English", "script": "none", "reliable": False}
    words = re.findall(r"[a-z]+", question.casefold())
    scores = {language: sum(1.0 if len(word) > 3 else 0.6 for word in words if word in lexicon)
              for language, lexicon in _ROMANISED.items()}
    english = sum(1 for word in words if word in _ENGLISH_WORDS)
    best, score = max(scores.items(), key=lambda item: item[1])
    tied = sum(1 for value in scores.values() if value == score) > 1
    if score >= 1 and score > english and not tied:
        return {"language": best, "script": "romanised", "reliable": True}
    return {"language": "English", "script": "latin", "reliable": english > 0 or len(words) >= 3}


def resolve_answer_language(choice: str, automatic: str, question_language: dict[str, Any], previous: str | None = None) -> str:
    """Target answer language.

    An explicitly selected response language always wins. In Automatic mode
    the answer follows the question's own language (never the document's);
    an ambiguous question keeps the language of the previous answer.
    """
    if choice != automatic and choice in SUPPORTED_LANGUAGES:
        return choice
    if not question_language.get("reliable") and previous in SUPPORTED_LANGUAGES:
        return previous  # type: ignore[return-value]
    language = question_language.get("language")
    return language if language in SUPPORTED_LANGUAGES else "English"


def is_in_language(text: str, language: str) -> bool:
    """True when most of the letters of `text` are written in `language`'s script.

    Names, codes and URLs in Latin script are common inside Indian-language
    answers, so an Indian-language answer passes when its script holds at
    least 45% of letters and no other Indian script is larger.
    """
    shares = script_shares(re.sub(r"https?://\S+|\[Pages?[^\]]*\]", " ", text))
    if not shares:
        return True  # Numbers or symbols only: nothing to translate.
    if language == "English":
        return shares.get("English", 0.0) >= 0.8
    target = shares.get(language, 0.0)
    rival = max((share for name, share in shares.items() if name not in {"English", "Other", language}), default=0.0)
    return target >= 0.45 and target >= rival
