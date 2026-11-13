"""Conservative, local-only redaction helpers for resume-like JSON or text.

This is a risk-reduction utility, not an anonymizer. Pattern matching cannot
find every identity, indirect identifier, image/header content, or disguised
contact detail. Supply known applicant names explicitly; unlabelled names are
not inferred. Review output before any disclosure and keep source files secure.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_URL = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\]\[(){};,!?\"']+|"
    r"(?<![@\w])(?:[\w-]+\.)+[a-zа-я]{2,}"
    r"(?:/[^\s<>\]\[(){};,!?\"']*)?"
)
_TECHNOLOGIES = {"node.js", "vue.js", "next.js", "react.js", "asp.net"}
# Restrict to clearly phone-like forms (country prefix or common Russian 8-prefix)
# so versions such as Python 3.12 and skill counts are not mistaken for phones.
_PHONE = re.compile(r"(?<!\w)(?:\+7|8)[\s().-]*(?:\d[\s().-]*){10}(?!\w)")
_NAME_LABEL = re.compile(
    r"(?im)(\b(?:name|full\s*name|candidate|фио|имя|фамилия)\s*[:—-]\s*)[^\r\n]+"
)
_PII_KEYS = {
    "name",
    "fullname",
    "candidate",
    "candidatename",
    "applicant",
    "applicantname",
    "фио",
    "имя",
    "фамилия",
    "имяфамилия",
}
_CONTACT_KEYS = {
    "email": "[EMAIL]",
    "почта": "[EMAIL]",
    "электроннаяпочта": "[EMAIL]",
    "телефон": "[PHONE]",
    "phone": "[PHONE]",
    "mobile": "[PHONE]",
    "phonenumber": "[PHONE]",
    "номер": "[PHONE]",
    "url": "[LINK]",
    "website": "[LINK]",
    "linkedin": "[LINK]",
    "сайт": "[LINK]",
    "ссылка": "[LINK]",
    "ссылки": "[LINK]",
}


def _redact_text(text: str, names: tuple[str, ...]) -> str:
    """Redact known PII patterns and explicitly supplied applicant names."""
    text = _NAME_LABEL.sub(r"\1[NAME]", text)
    for name in sorted(names, key=len, reverse=True):
        if name.strip():
            text = re.sub(re.escape(name.strip()), "[NAME]", text, flags=re.IGNORECASE)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub(
        lambda match: (
            match.group() if match.group().lower() in _TECHNOLOGIES else "[LINK]"
        ),
        text,
    )
    return _PHONE.sub("[PHONE]", text)


def deidentify(value: Any, *, names: list[str] | tuple[str, ...] = ()) -> Any:
    """Return a recursively redacted copy of structured data or free text.

    Recognized name/contact fields are replaced as whole values; other strings
    have labeled names, explicit ``names`` entries, emails, URLs and common
    Russian phone forms redacted. Non-string scalars and non-PII text are kept.
    This does not guarantee complete anonymization: arbitrary unlabelled names,
    contextual/indirect identifiers, OCR/image content, and novel obfuscations
    may remain. Regexes can also miss unusual formats or redact phone-like text.
    """

    def key_name(key: Any) -> str:
        return re.sub(r"[\W_]", "", str(key).lower())

    def collect_names(item: Any) -> list[str]:
        if isinstance(item, Mapping):
            return [
                name
                for key, child in item.items()
                for name in (
                    [child]
                    if key_name(key) in _PII_KEYS and isinstance(child, str)
                    else collect_names(child)
                )
            ]
        if isinstance(item, (list, tuple)):
            return [name for child in item for name in collect_names(child)]
        return []

    known_names = tuple(
        name for name in (*names, *collect_names(value)) if isinstance(name, str)
    )

    def redact(item: Any) -> Any:
        if isinstance(item, str):
            return _redact_text(item, known_names)
        if isinstance(item, Mapping):
            cleaned = {}
            for key, child in item.items():
                normalized_key = key_name(key)
                if normalized_key in _PII_KEYS:
                    cleaned[key] = "[NAME]"
                elif normalized_key in _CONTACT_KEYS:
                    cleaned[key] = _CONTACT_KEYS[normalized_key]
                else:
                    cleaned[key] = redact(child)
            return cleaned
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, tuple):
            return tuple(redact(child) for child in item)
        return item

    return redact(value)
