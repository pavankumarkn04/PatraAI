"""Document categories: which fields matter and which questions to suggest.

Every document is analysed with the same field menu, but a certificate, a
scholarship notification and an admission brochure need different views.
The category decides which fields are always shown (with "Not found in
document" when absent) and which suggested questions are relevant. Fields
outside the expected set are shown only when the document states them.
"""

from __future__ import annotations

import re
from typing import Any

# Category: (keywords, expected fields, label overrides, suggested questions).
PROFILES: dict[str, dict[str, Any]] = {
    "form": {
        "keywords": ("application form", "form", "proforma", "performa", "application for", "ಅರ್ಜಿ ನಮೂನೆ", "आवेदन पत्र", "प्रपत्र"),
        "expected": ("document_title", "document_type", "issuing_department", "required_documents", "important_instructions"),
        "labels": {},
        "questions": ("What is this form for?", "Who should fill in this form?", "Which documents must be attached?",
                      "Where should the form be submitted?", "What are the important instructions?"),
    },
    "certificate": {
        "keywords": ("certificate", "pramaan", "praman patra", "ಪ್ರಮಾಣ ಪತ್ರ", "ಪ್ರಮಾಣಪತ್ರ", "प्रमाण पत्र", "प्रमाणपत्र", "சான்றிதழ்", "ధృవీకరణ పత్రం", "സർട്ടിഫിക്കറ്റ്"),
        "expected": ("document_title", "document_type", "issuing_department", "date", "reference_number"),
        "labels": {"document_title": "Certificate type", "reference_number": "Certificate number", "date": "Issue date"},
        "questions": ("Who issued this certificate?", "Who is this certificate issued to?", "What is the certificate number?",
                      "When was it issued?", "What is this certificate valid for?"),
    },
    "scholarship": {
        # Not "welfare": it names departments ("Backward Classes Welfare Department") that issue all kinds of orders.
        "keywords": ("scholarship", "scheme", "yojana", "stipend", "fellowship", "financial assistance", "ವಿದ್ಯಾರ್ಥಿ ವೇತನ", "छात्रवृत्ति", "योजना", "உதவித்தொகை", "ఉపకార వేతనం", "സ്കോളർഷിപ്പ്"),
        "expected": ("document_title", "document_type", "issuing_department", "date", "eligibility_criteria", "amount",
                     "important_dates", "required_documents", "application_procedure"),
        "labels": {"document_title": "Scheme or scholarship name", "amount": "Amount / income limit"},
        "questions": ("Who is eligible?", "What are the eligibility conditions, such as income or marks?", "What are the application dates?",
                      "Which documents are required?", "How do I apply?", "What benefit or amount is given?"),
    },
    "admission": {
        "keywords": ("admission", "brochure", "prospectus", "entrance", "counselling", "cet", "information bulletin", "ಪ್ರವೇಶ", "प्रवेश", "சேர்க்கை", "ప్రవేశ", "പ്രവേശന"),
        "expected": ("document_title", "document_type", "issuing_department", "eligibility_criteria", "application_procedure",
                     "important_dates", "amount"),
        "labels": {"issuing_department": "Institution or authority", "amount": "Fees"},
        "questions": ("Which courses does this cover?", "Who is eligible for admission?", "What is the admission process?",
                      "What are the important dates?", "What are the fees?"),
    },
    "recruitment": {
        "keywords": ("recruitment", "vacancy", "vacancies", "posts", "appointment", "deputation", "selection", "ನೇಮಕಾತಿ", "भर्ती", "रिक्ति", "ஆட்சேர்ப்பு", "నియామక", "നിയമന"),
        "expected": ("document_title", "document_type", "issuing_department", "date", "reference_number", "eligibility_criteria",
                     "important_dates", "application_procedure"),
        "labels": {},
        "questions": ("Which posts are available?", "Who is eligible to apply?", "What is the last date to apply?",
                      "How do I apply?", "Which documents are required?"),
    },
    "notice": {
        "keywords": ("notice", "circular", "order", "press release", "notification", "government order", "proceedings", "gazette",
                     "memorandum", "letter", "guidelines", "ಸುತ್ತೋಲೆ", "ಆದೇಶ", "ಅಧಿಸೂಚನೆ", "परिपत्र", "आदेश", "अधिसूचना", "सूचना", "சுற்றறிக்கை", "அரசாணை", "ఉత్తర్వు", "ఉత్తర్వులు", "సర్క్యులర్", "ഉത്തരവ്", "സർക്കുലർ"),
        "expected": ("document_title", "document_type", "issuing_department", "date", "reference_number", "important_instructions"),
        "labels": {},
        "questions": ("Who issued this document?", "What is this document about?", "What are the important instructions?",
                      "Who does this apply to?"),
    },
    "other": {
        "keywords": (),
        "expected": ("document_title", "document_type", "issuing_department", "date"),
        "labels": {},
        "questions": ("What is this document about?", "Who issued this document?", "What are the key points?"),
    },
}
# Categories decided from the document type and title alone, in this order.
_TYPE_ORDER = ("form", "certificate", "scholarship", "admission", "recruitment", "notice")
# Categories that may also be recognised from the opening text.
_TEXT_ORDER = ("scholarship", "admission", "recruitment")

# Generic questions, offered only when the document states the field.
_FIELD_QUESTIONS = (
    ("important_dates", "What are the important dates?"), ("deadline", "What is the last date?"),
    ("eligibility_criteria", "What are the eligibility requirements?"), ("required_documents", "Which documents are required?"),
    ("application_procedure", "How do I apply?"), ("important_instructions", "What are the important instructions?"),
    ("amount", "What amounts or fees are mentioned?"),
)


def _matches(text: str, keywords: tuple[str, ...]) -> bool:
    text = text.casefold()
    return any(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) if keyword.isascii() else keyword in text for keyword in keywords)


def document_category(analysis: dict[str, Any], page_texts: list[str] | None = None) -> str:
    """Category of an analysed document: form, certificate, scholarship,
    admission, recruitment, notice or other."""
    fields = analysis.get("fields") or {}
    title = str(fields.get("document_title") or "")
    document_type = " ".join(str(value or "") for value in (fields.get("document_type"), analysis.get("document_type")))
    # A form or a certificate is recognised from the title and type together
    # ("Income certificate application" typed "Application form" is a form).
    for category in _TYPE_ORDER[:2]:
        if _matches(f"{document_type} {title}", PROFILES[category]["keywords"]):
            return category
    # Otherwise the title first: it names the document itself, while a
    # model-written type can mix categories ("Scheme/Notice" for a CBSE press
    # release about language policy).
    for heading in (title, document_type):
        for category in _TYPE_ORDER[2:]:
            if _matches(heading, PROFILES[category]["keywords"]):
                return category
    opening = " ".join(page_texts[:2])[:2500] if page_texts else ""
    for category in _TEXT_ORDER:
        if _matches(opening, PROFILES[category]["keywords"]):
            return category
    return "notice" if _matches(opening, PROFILES["notice"]["keywords"]) else "other"


def field_rows(category: str, fields: dict[str, Any], optional_keys: tuple[str, ...]) -> list[tuple[str, bool]]:
    """(field key, shown even when missing) in display order.

    Expected fields of the category are always listed; any other field only
    when the document states it.
    """
    expected = PROFILES[category]["expected"]
    rows = [(key, True) for key in expected]
    rows += [(key, False) for key in optional_keys if key not in expected and fields.get(key)]
    return rows


def field_label(category: str, key: str, default: str, localized: bool) -> str:
    """Category-specific English label; translated generic labels are kept as they are."""
    if localized:
        return default
    return PROFILES[category]["labels"].get(key, default)


def suggested_questions(category: str, fields: dict[str, Any], limit: int = 6) -> list[str]:
    """Questions relevant to this document: the category's own, then generic
    ones for fields the document actually states."""
    questions = list(PROFILES[category]["questions"])
    for key, question in _FIELD_QUESTIONS:
        if fields.get(key) and question not in questions:
            questions.append(question)
    return questions[:limit]
