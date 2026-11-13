"""Extract explicit vacancy fields while preserving the source text."""

import copy
import hashlib
import re

HEADINGS = re.compile(
    r"(?im)^[ \t]*(?P<label>обязанности|должностные обязанности|"
    r"задачи|будущие задачи|какие задачи предстоит решать|"
    r"чем (?:вам )?предстоит заниматься|чем вы будете заниматься|"
    r"требования(?: к кандидату)?|основные требования к кандидатам на данную должность|"
    r"что мы ожидаем(?: от вас)?|наши ожидания от вас|"
    r"ты нам подходишь, если у тебя есть|ожидания от кандидата|"
    r"будет плюсом|будет преимуществом|преимуществом будет|"
    r"условия(?: работы)?|условия, которые мы вам предлагаем|"
    r"мы предлагаем|рады предложить|что мы предлагаем)[ \t]*:?[ \t]*$"
)
EXPERIENCE = re.compile(
    r"(?i)опыт[^\n]{0,80}?(?:от|не менее(?: чем)?)\s*"
    r"(?P<years>\d{1,2})(?:\s*[-–]\s*х)?"
    r"(?:\s*(?:до|[-–])\s*\d{1,2})?\s*(?:лет|год(?:а)?)\b"
)
OPTIONAL = re.compile(r"(?i)желател|преимуществ|будет плюсом|приветствуется")
EDUCATION = re.compile(r"(?i)(?:высш\w*|средн\w*)[^\n]{0,80}образован")
RESEARCH_HEADING = re.compile(
    r"(?im)^[ \t]*(?P<label>Научная тематика работы|"
    r"Квалификационные требования|Условия работы|Перечень трудовых функций|"
    r"Ожидаемые результаты|Наличие допусков, сертификатов, и т\.п\.)[ \t]*:"
)
SALARY = re.compile(r"(?i)зарплат|заработн\w*\s+плат")
MONEY = re.compile(r"(?<!\d)(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d{4,8})(?!\d)")
CURRENCY = re.compile(r"(?i)руб(?:\.|л(?:ей|я|ь)?)?|₽")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim only whitespace without altering the underlying evidence."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _label_field(label: str) -> str:
    label = label.casefold()
    if label.startswith(
        ("обязанности", "должностные", "задачи", "будущие задачи", "какие", "чем")
    ):
        return "responsibilities"
    if label.startswith(("будет", "преимуществом")):
        return "additionalRequirements"
    if label.startswith(
        (
            "требования",
            "основные требования",
            "что мы ожидаем",
            "наши ожидания",
            "ты нам",
            "ожидания",
        )
    ):
        return "positionRequirements"
    return "conditions"


def _description_sections(record: dict) -> list[tuple[str, int, int]]:
    text = record["text"]
    result = []
    for section in record.get("sections", []):
        if section.get("source_field") != "description":
            continue
        start, end = section["start"], section["end"]
        headings = list(HEADINGS.finditer(text, start, end))
        for index, match in enumerate(headings):
            stop = headings[index + 1].start() if index + 1 < len(headings) else end
            first, last = _trim_span(text, match.end(), stop)
            if first < last:
                result.append((_label_field(match.group("label")), first, last))
    return result


def _add_text_field(
    record: dict, audit: dict, field: str, start: int, end: int
) -> None:
    if field in record["attributes"]:
        return
    value = record["text"][start:end]
    record["attributes"][field] = value
    record["sections"].append(
        {"source_field": field, "label": field, "start": start, "end": end}
    )
    audit["derived"][field] = {"start": start, "end": end, "evidence": value}


def redact_scientific_contacts(original: dict) -> tuple[dict, bool]:
    """Remove a final scientific-contact line while preserving other sections."""
    record = copy.deepcopy(original)
    if record.get("source", {}).get("source_id_kind") != "competition_vacancy_id":
        return record, False
    text = record["text"]
    match = re.search(r"(?im)^[ \t]*Контакт для ответа[^\r\n]*[\r\n]*\Z", text)
    if not match:
        return record, False
    new_text = text[: match.start()].rstrip()
    for section in record["sections"]:
        if section["end"] > len(new_text):
            if section["start"] >= len(new_text):
                raise ValueError("Contact removal would erase an entire section")
            section["end"] = len(new_text)
            record["attributes"][section["source_field"]] = new_text[
                section["start"] : section["end"]
            ]
    record["text"] = new_text
    record["text_sha256"] = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
    return record, True


def parse_record(original: dict) -> tuple[dict, dict]:
    """Derive fields only from verbatim evidence in a vacancy's existing text.

    Args:
        original: A source vacancy record with text and section offsets.

    Returns:
        A copied record and an audit entry with derivation evidence offsets.
    """
    record = copy.deepcopy(original)
    audit = {
        "id": record["id"],
        "source_quality": record.get("source", {}).get("text_completeness"),
        "derived": {},
    }
    for field, start, end in _description_sections(record):
        _add_text_field(record, audit, field, start, end)
    original_requirements = next(
        (
            s
            for s in record["sections"]
            if s.get("source_field") == "positionRequirements"
        ),
        None,
    )
    if original_requirements:
        start, end = original_requirements["start"], original_requirements["end"]
        optional_heading = re.search(
            r"(?im)^[ \t]*(?:будет плюсом|будет преимуществом|"
            r"преимуществом будет)[ \t]*:",
            record["text"][start:end],
        )
        if optional_heading:
            a, b = _trim_span(record["text"], start + optional_heading.end(), end)
            if a < b:
                _add_text_field(record, audit, "additionalRequirements", a, b)
        headings = list(RESEARCH_HEADING.finditer(record["text"], start, end))
        research_fields = {
            "научная тематика работы": "researchTopic",
            "квалификационные требования": "qualifications",
            "условия работы": "conditions",
            "перечень трудовых функций": "specificResponsibilities",
            "ожидаемые результаты": "expectedResults",
            "наличие допусков, сертификатов, и т.п.": "certificates",
        }
        for index, match in enumerate(headings):
            field = research_fields[match.group("label").casefold()]
            stop = headings[index + 1].start() if index + 1 < len(headings) else end
            a, b = _trim_span(record["text"], match.end(), stop)
            if a < b:
                _add_text_field(record, audit, field, a, b)
        source = record.get("source", {})
        if "qualifications" not in record["attributes"] and (
            source.get("source_id_kind") != "competition_vacancy_id"
            or source.get("text_completeness") == "search_index_excerpt_incomplete"
        ):
            _add_text_field(record, audit, "qualifications", start, end)
    requirements = next(
        (s for s in record["sections"] if s.get("source_field") == "qualifications"),
        original_requirements,
    )
    if requirements:
        start, end = requirements["start"], requirements["end"]
        optional_section = False
        for line_match in re.finditer(r"[^\n]+", record["text"][start:end]):
            line_start = start + line_match.start()
            line_end = start + line_match.end()
            line = line_match.group()
            if OPTIONAL.search(line) and line.strip().endswith(":"):
                optional_section = True
            if optional_section:
                continue
            if OPTIONAL.search(line):
                continue
            experience = EXPERIENCE.search(line)
            if experience and "experienceRequirements" not in record["attributes"]:
                a = line_start + experience.start()
                b = line_start + experience.end()
                record["attributes"]["experienceRequirements"] = int(
                    experience.group("years")
                )
                audit["derived"]["experienceRequirements"] = {
                    "start": a,
                    "end": b,
                    "evidence": record["text"][a:b],
                }
            knowledge_only = re.match(
                r"(?i)^[\s•\-*]*(?:знани[ея]|понимани[ея])\b", line
            )
            if EDUCATION.search(line) and not knowledge_only:
                a, b = _trim_span(record["text"], line_start, line_end)
                education_label = re.search(r"(?i)\bобразование\s*:", line)
                if education_label:
                    a = line_start + education_label.start()
                    following = re.search(
                        r"(?i)\s+(?:ученая степень|опыт работы|"
                        r"примерный перечень)\s*:",
                        line[education_label.end() :],
                    )
                    if following:
                        b = line_start + education_label.end() + following.start()
                    a, b = _trim_span(record["text"], a, b)
                _add_text_field(record, audit, "educationRequirements", a, b)
    conditions = next(
        (s for s in record["sections"] if s.get("source_field") == "salaryText"),
        next(
            (s for s in record["sections"] if s.get("source_field") == "conditions"),
            None,
        ),
    )
    if conditions:
        start, end = conditions["start"], conditions["end"]
        for line_match in re.finditer(r"[^\n]+", record["text"][start:end]):
            line = line_match.group()
            amounts = list(MONEY.finditer(line))
            if (
                (conditions["source_field"] != "salaryText" and not SALARY.search(line))
                or not CURRENCY.search(line)
                or not 1 <= len(amounts) <= 2
            ):
                continue
            if len(amounts) == 2 and not re.search(
                r"(?i)до|[-–—]", line[amounts[0].end() : amounts[1].start()]
            ):
                continue
            a, b = _trim_span(
                record["text"], start + line_match.start(), start + line_match.end()
            )
            _add_text_field(record, audit, "salaryText", a, b)
            prefix = line[: amounts[0].start()].casefold()
            fields = ["salaryMin", "salaryMax"]
            if len(amounts) == 1:
                fields = (
                    ["salaryMax"] if re.search(r"\bдо\s*$", prefix) else ["salaryMin"]
                )
                if not re.search(r"\b(?:от|до)\s*$", prefix):
                    fields = ["salaryMin", "salaryMax"]
            for index, field in enumerate(fields):
                if field in record["attributes"]:
                    continue
                amount = amounts[min(index, len(amounts) - 1)]
                a = start + line_match.start() + amount.start()
                b = start + line_match.start() + amount.end()
                record["attributes"][field] = int(re.sub(r"\s", "", amount.group()))
                audit["derived"][field] = {
                    "start": a,
                    "end": b,
                    "evidence": record["text"][a:b],
                }
            break
    return record, audit


def requirement_items(record: dict) -> list[dict]:
    """Split explicit requirement sections into verbatim, evidence-linked lines.

    Priority remains unspecified unless the source explicitly marks a criterion
    as desirable. Scientific residual descriptions without a qualification
    heading are not assumed to be pure requirements.
    """
    sections = record.get("sections", [])
    preferred = next(
        (s for s in sections if s["source_field"] == "qualifications"), None
    )
    main = preferred or next(
        (s for s in sections if s["source_field"] == "positionRequirements"), None
    )
    source = record.get("source", {})
    if (
        main
        and not preferred
        and source.get("source_id_kind") == "competition_vacancy_id"
        and source.get("text_completeness") != "search_index_excerpt_incomplete"
    ):
        main = None
    additional = next(
        (s for s in sections if s["source_field"] == "additionalRequirements"), None
    )
    if (
        main
        and additional
        and main["start"] <= additional["start"]
        and additional["end"] <= main["end"]
    ):
        additional = None
    degree = (
        next((s for s in sections if s["source_field"] == "degree"), None)
        if main is None
        else None
    )
    scope = (
        "generic_role"
        if source.get("text_completeness")
        == "generic_role_requirements_not_department_specific"
        else "vacancy"
    )
    items = []
    for section, initial in (
        (main, "unspecified"),
        (additional, "preferred"),
        (degree, "unspecified"),
    ):
        if section is None:
            continue
        priority = initial
        start, end = section["start"], section["end"]
        for match in re.finditer(r"[^\n]+", record["text"][start:end]):
            a, b = _trim_span(
                record["text"], start + match.start(), start + match.end()
            )
            value = record["text"][a:b]
            if not value or value.strip("-• ") == "":
                continue
            if OPTIONAL.search(value) and value.endswith(":"):
                priority = "preferred"
                continue
            if value.endswith(":"):
                continue
            items.append(
                {
                    "text": value,
                    "start": a,
                    "end": b,
                    "source_field": section["source_field"],
                    "priority": "preferred" if OPTIONAL.search(value) else priority,
                    "scope": scope,
                }
            )
    return items
