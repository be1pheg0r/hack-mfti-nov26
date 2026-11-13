"""Tests for evidence-bound vocabulary mentions."""

from vrscore.vacancy_parser.skill_mentions import extract_mentions


def test_mentions_keep_exact_quote_priority_and_offsets():
    text = "Нужен Python.\nБудет плюсом: SQL или 1С."
    items = [
        {
            "text": "Нужен Python.",
            "start": 0,
            "end": 13,
            "priority": "unspecified",
            "scope": "vacancy",
        },
        {
            "text": "Будет плюсом: SQL или 1С.",
            "start": 14,
            "end": len(text),
            "priority": "preferred",
            "scope": "vacancy",
        },
    ]
    vocabulary = [
        {"label": "Python", "category": "IT", "patterns": [r"(?i)\bpython\b"]},
        {"label": "SQL", "category": "IT", "patterns": [r"(?i)\bsql\b"]},
    ]
    mentions = extract_mentions({"id": "v", "text": text}, items, vocabulary)
    assert [m["label"] for m in mentions] == ["Python", "SQL"]
    assert [m["priority"] for m in mentions] == ["unspecified", "preferred"]
    assert all(text[m["start"] : m["end"]] == m["matched_text"] for m in mentions)
    assert all(
        text[m["context_start"] : m["context_end"]] == m["context"] for m in mentions
    )
