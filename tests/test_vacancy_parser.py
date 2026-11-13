"""Tests for evidence-preserving vacancy field extraction."""

from vrscore.vacancy_parser.parser import (
    parse_record,
    redact_scientific_contacts,
)


def test_structure_unstructured_description_and_track_evidence():
    """Pull verbatim sections from a description without fabricating fields."""
    text = (
        "Должность:\nРазработчик\n\nОписание вакансии:\n"
        "Обязанности:\nПисать код.\n"
        "Требования:\nОпыт Python от 3 лет.\nВысшее техническое образование.\n"
        "Условия:\nУдалённая работа."
    )
    begin = text.index("Обязанности:")
    record = {
        "id": "vacancy_001",
        "kind": "vacancy",
        "title": "Разработчик",
        "text": text,
        "attributes": {"description": text[begin:]},
        "sections": [
            {
                "source_field": "description",
                "label": "Описание",
                "start": begin,
                "end": len(text),
            }
        ],
        "source": {"text_completeness": "public_page_description"},
    }
    output, audit = parse_record(record)
    assert output["text"] == text
    assert record["attributes"].keys() == {"description"}
    assert output["attributes"]["responsibilities"] == "Писать код."
    assert output["attributes"]["positionRequirements"] == (
        "Опыт Python от 3 лет.\nВысшее техническое образование."
    )
    assert output["attributes"]["conditions"] == "Удалённая работа."
    assert output["attributes"]["experienceRequirements"] == 3
    assert (
        output["attributes"]["educationRequirements"]
        == "Высшее техническое образование."
    )
    assert "salaryMin" not in output["attributes"]
    for section in output["sections"]:
        assert (
            text[section["start"] : section["end"]]
            == output["attributes"][section["source_field"]]
        )
    exp = audit["derived"]["experienceRequirements"]
    assert text[exp["start"] : exp["end"]] == exp["evidence"]


def test_keep_existing_fields_and_do_not_invent_experience():
    """Do not turn unspecified experience or preferred criteria into facts."""
    text = "Требования:\nЖелателен опыт работы от 5 лет.\nОбязанности:\nОбучать."
    start = text.index("Желателен")
    end = text.index("\nОбязанности")
    record = {
        "id": "vacancy_002",
        "kind": "vacancy",
        "title": "Преподаватель",
        "text": text,
        "attributes": {"positionRequirements": text[start:end]},
        "sections": [
            {
                "source_field": "positionRequirements",
                "label": "Требования",
                "start": start,
                "end": end,
            }
        ],
        "source": {
            "text_completeness": "generic_role_requirements_not_department_specific"
        },
    }
    output, audit = parse_record(record)
    assert "experienceRequirements" not in output["attributes"]
    assert (
        audit["source_quality"] == "generic_role_requirements_not_department_specific"
    )
    assert output["attributes"]["positionRequirements"] == text[start:end]


def test_optional_section_does_not_become_mandatory_experience():
    """An explicit desirable subsection scopes following requirement lines."""
    text = "Требования:\nОтветственность.\nБудет преимуществом:\nОпыт работы от 5 лет."
    start = text.index("Ответственность")
    record = {
        "id": "vacancy_003",
        "kind": "vacancy",
        "title": "Сотрудник",
        "text": text,
        "attributes": {"positionRequirements": text[start:]},
        "sections": [
            {
                "source_field": "positionRequirements",
                "label": "Требования",
                "start": start,
                "end": len(text),
            }
        ],
        "source": {},
    }
    output, _ = parse_record(record)
    assert "experienceRequirements" not in output["attributes"]


def test_extract_alternative_headings_and_separate_preferred_criteria():
    """Retain desired criteria separately from explicit main requirements."""
    text = (
        "Чем Вам предстоит заниматься:\nПисать код.\n"
        "Наши ожидания от Вас:\nЗнание Python.\n"
        "Будет плюсом:\nОпыт работы от 5 лет.\n"
        "Условия, которые мы Вам предлагаем:\nГибкий график."
    )
    record = {
        "id": "vacancy_004",
        "kind": "vacancy",
        "title": "Разработчик",
        "text": text,
        "attributes": {"description": text},
        "sections": [
            {
                "source_field": "description",
                "label": "Описание",
                "start": 0,
                "end": len(text),
            }
        ],
        "source": {},
    }
    output, _ = parse_record(record)
    assert output["attributes"]["positionRequirements"] == "Знание Python."
    assert output["attributes"]["additionalRequirements"] == "Опыт работы от 5 лет."
    assert "experienceRequirements" not in output["attributes"]


def test_research_qualifications_exclude_duties_and_conditions():
    """A research competition's residual field is not all qualifications."""
    text = (
        "Прочие требования и описание:\nНаучная тематика работы: Биофизика\n"
        "Квалификационные требования:\nОбразование: Высшее образование\n"
        "Опыт работы: от 3 лет\nУсловия работы: Полная занятость\n"
        "Перечень трудовых функций: Исследования."
    )
    start = text.index("Научная тематика")
    record = {
        "id": "vacancy_005",
        "kind": "vacancy",
        "title": "Научный сотрудник",
        "text": text,
        "attributes": {"positionRequirements": text[start:]},
        "sections": [
            {
                "source_field": "positionRequirements",
                "label": "Требования",
                "start": start,
                "end": len(text),
            }
        ],
        "source": {},
    }
    output, _ = parse_record(record)
    assert output["attributes"]["qualifications"] == (
        "Образование: Высшее образование\nОпыт работы: от 3 лет"
    )
    assert output["attributes"]["experienceRequirements"] == 3
    assert output["attributes"]["educationRequirements"] == (
        "Образование: Высшее образование"
    )


def test_education_does_not_misclassify_knowledge_of_education_law():
    """References to education systems are not education credentials."""
    text = "Знание нормативной базы российского высшего образования."
    record = {
        "id": "vacancy_006",
        "kind": "vacancy",
        "title": "Сотрудник",
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
    output, _ = parse_record(record)
    assert "educationRequirements" not in output["attributes"]


def test_excerpt_education_is_a_short_exact_subspan():
    """Do not use a whole one-line research excerpt as education value."""
    text = (
        "Образование: Высшее образование Ученая степень: кандидат наук "
        "Опыт работы: от 10"
    )
    record = {
        "id": "vacancy_007",
        "kind": "vacancy",
        "title": "Сотрудник",
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
    output, _ = parse_record(record)
    assert (
        output["attributes"]["educationRequirements"]
        == "Образование: Высшее образование"
    )


def test_redact_only_actual_scientific_contact_tail():
    """Remove recruiter contact tails without truncating ordinary job duties."""
    text = "Требования:\nОпыт работы.\nКонтакт для ответа: Имя Фамилия"
    start = text.index("Опыт")
    record = {
        "id": "vacancy_008",
        "kind": "vacancy",
        "title": "Научный сотрудник",
        "text": text,
        "attributes": {"positionRequirements": text[start:]},
        "sections": [
            {
                "source_field": "positionRequirements",
                "label": "Требования",
                "start": start,
                "end": len(text),
            }
        ],
        "source": {"source_id_kind": "competition_vacancy_id"},
    }
    output, changed = redact_scientific_contacts(record)
    assert changed and "Имя" not in output["text"]
    s = output["sections"][0]
    assert (
        output["text"][s["start"] : s["end"]] == output["attributes"][s["source_field"]]
    )
    assert record["text"] == text


def test_salary_range_and_unspecified_salary_are_distinct():
    """Only amounts explicitly associated with salary become numeric fields."""
    text = "Условия:\nЗаработная плата от 90 000 до 115 000 рублей до вычета НДФЛ."
    start = text.index("Заработная")
    record = {
        "id": "vacancy_009",
        "kind": "vacancy",
        "title": "Сотрудник",
        "text": text,
        "attributes": {"conditions": text[start:]},
        "sections": [
            {
                "source_field": "conditions",
                "label": "Условия",
                "start": start,
                "end": len(text),
            }
        ],
        "source": {},
    }
    output, audit = parse_record(record)
    assert output["attributes"]["salaryMin"] == 90000
    assert output["attributes"]["salaryMax"] == 115000
    assert output["attributes"]["salaryText"] == text[start:]
    assert audit["derived"]["salaryMin"]["evidence"] == "90 000"

    upper = "Заработная плата до 220 000 руб."
    record["text"] = upper
    record["attributes"] = {"conditions": upper}
    record["sections"][0].update(start=0, end=len(upper))
    output, _ = parse_record(record)
    assert output["attributes"]["salaryMax"] == 220000
    assert "salaryMin" not in output["attributes"]

    record["attributes"] = {"salaryText": "от 80 000 до 120 000 ₽ на руки"}
    record["text"] = record["attributes"]["salaryText"]
    record["sections"][0].update(
        source_field="salaryText", start=0, end=len(record["text"])
    )
    output, _ = parse_record(record)
    assert output["attributes"]["salaryMin"] == 80000
    assert output["attributes"]["salaryMax"] == 120000
