"""Attach separately verified source fields with explicit provenance."""

import copy
import hashlib
import json
import re

FIELD_NAMES = {
    "experience": ("experience", "Опыт из карточки источника"),
    "schedule": ("scheduleType", "График из карточки источника"),
    "employment": ("busyType", "Занятость из карточки источника"),
    "salary": ("salaryText", "Оплата из карточки источника"),
    "salary_object": ("sourceSalary", "Оплата из структурированной карточки hh"),
    "base_salary": ("baseSalaryText", "Должностной оклад из карточки источника"),
    "key_skills": ("skills", "Навыки, указанные источником"),
    "education": ("sourceEducation", "Образование из карточки источника"),
    "stake": ("sourceStake", "Ставка из карточки источника"),
    "work_location": ("workLocation", "Место работы из карточки источника"),
}
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}")
PHONE = re.compile(
    r"(?<!\w)(?:\+7|8)[\s(\-]*\d{3}[\s)\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"
)


def apply_source_evidence(original: dict, evidence: dict) -> tuple[dict, dict]:
    """Attach verified fields or restore a truncated description from its source.

    Args:
        original: Vacancy before enrichment.
        evidence: Matching, separately verified source fields and source URL.

    Returns:
        A copied enriched record and an audit containing source field evidence.

    Raises:
        ValueError: The evidence does not match the record or contains contacts.
    """
    record = copy.deepcopy(original)
    if (
        evidence["id"] != record["id"]
        or evidence["source_url"] != record["source"]["original_url"]
    ):
        raise ValueError("Evidence source does not match the vacancy")
    audit = {"restored_description": False, "fields": {}}
    restored = evidence.get("full_description")
    if restored is not None:
        if EMAIL.search(restored) or PHONE.search(restored):
            raise ValueError("Description contains contact information")
        section = next(
            (s for s in record["sections"] if s["source_field"] == "description"), None
        )
        if not section or section["end"] != len(record["text"]):
            raise ValueError("Cannot safely replace nonfinal description")
        previous = record["attributes"]["description"]
        if not restored.startswith(previous) or len(restored) <= len(previous):
            raise ValueError("Recovered description does not extend original text")
        record["text"] = record["text"][: section["start"]] + restored
        record["attributes"]["description"] = restored
        section["end"] = len(record["text"])
        audit["restored_description"] = True
    for key, item in evidence.get("fields", {}).items():
        if key not in FIELD_NAMES:
            raise ValueError(f"Unsupported evidence field: {key}")
        field, label = FIELD_NAMES[key]
        if field in record["attributes"]:
            continue
        value = item["value"]
        if key == "key_skills":
            if not isinstance(value, list) or not all(
                isinstance(x, str) for x in value
            ):
                raise ValueError("Skills must be a list of strings")
            rendered = "; ".join(value)
        elif key == "salary_object":
            if not isinstance(value, dict):
                raise ValueError("Structured salary must be an object")
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
            value = rendered
        else:
            if not isinstance(value, str):
                raise ValueError("Source field must be a string")
            rendered = value
        if not rendered.strip():
            continue
        if EMAIL.search(rendered) or PHONE.search(rendered):
            raise ValueError("Source field contains contact information")
        record["text"] += f"\n\n{label}:\n"
        start = len(record["text"])
        record["text"] += rendered
        end = len(record["text"])
        record["attributes"][field] = value
        record["sections"].append(
            {"source_field": field, "label": label, "start": start, "end": end}
        )
        audit["fields"][field] = {
            "start": start,
            "end": end,
            "evidence": rendered,
            "source_url": evidence["source_url"],
            "source_label": item["source_label"],
        }
        if key == "base_salary":
            amount = re.fullmatch(r"\s*([\d\s\u00a0]+)\s*руб\.\s*", rendered)
            if amount:
                number = int(re.sub(r"\s", "", amount.group(1)))
                if number > 0:
                    a, b = start + amount.start(1), start + amount.end(1)
                    while a < b and record["text"][b - 1].isspace():
                        b -= 1
                    record["attributes"]["baseSalary"] = number
                    audit["fields"]["baseSalary"] = {
                        "start": a,
                        "end": b,
                        "evidence": record["text"][a:b],
                        "source_url": evidence["source_url"],
                        "source_label": item["source_label"],
                    }
        if key == "salary_object":
            for source_key, target in (("from", "salaryMin"), ("to", "salaryMax")):
                amount = item["value"].get(source_key)
                if (
                    isinstance(amount, int)
                    and not isinstance(amount, bool)
                    and amount >= 0
                ):
                    if target in record["attributes"]:
                        continue
                    match = re.search(rf'"{source_key}":\s*({amount})\b', rendered)
                    if not match:
                        raise ValueError("Salary amount is not present in evidence")
                    a, b = start + match.start(1), start + match.end(1)
                    record["attributes"][target] = amount
                    audit["fields"][target] = {
                        "start": a,
                        "end": b,
                        "evidence": record["text"][a:b],
                        "source_url": evidence["source_url"],
                        "source_label": f"{item['source_label']}.{source_key}",
                    }
    record["text_sha256"] = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    return record, audit
