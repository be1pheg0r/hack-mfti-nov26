"""Check evidence spans and uncertainty on parsed requirement items."""

from vacancy_parser.parser import parse_record, requirement_items


def test_requirement_items_preserve_verbatim_and_optional_scope():
    """Preferred requirements must not be labeled as mandatory."""
    text = "Требования:\nPython\nБудет плюсом:\nSQL\n"
    begin = text.index("Python")
    record = {
        "id": "vacancy_001",
        "text": text,
        "attributes": {"positionRequirements": text[begin:].rstrip()},
        "sections": [
            {
                "source_field": "positionRequirements",
                "label": "Требования",
                "start": begin,
                "end": len(text.rstrip()),
            }
        ],
        "source": {"source_id_kind": "mipt_hr_id"},
    }
    items = requirement_items(record)
    assert [x["text"] for x in items] == ["Python", "SQL"]
    assert [x["priority"] for x in items] == ["unspecified", "preferred"]
    assert all(text[item["start"] : item["end"]] == item["text"] for item in items)


def test_preferred_subsection_does_not_duplicate_requirement_items():
    """A nested preferred subsection gets a field but is not counted twice."""
    text = "Python\nБудет плюсом:\nSQL"
    record = {
        "id": "vacancy_002",
        "text": text,
        "attributes": {"positionRequirements": text},
        "sections": [
            {
                "source_field": "positionRequirements",
                "label": "Требования",
                "start": 0,
                "end": len(text),
            }
        ],
        "source": {},
    }
    parsed, _ = parse_record(record)
    assert parsed["attributes"]["qualifications"] == text
    assert parsed["attributes"]["additionalRequirements"] == "SQL"
    assert [x["text"] for x in requirement_items(parsed)] == ["Python", "SQL"]
