"""Build conservative, evidence-linked preliminary vacancy reports."""

import re


def _key(value):
    """Normalize a few unambiguous spelling variants, not arbitrary substrings."""
    text = re.sub(r"\s+", " ", str(value).casefold()).strip()
    aliases = {
        "tailwind css": "tailwind",
        "tailwindcss": "tailwind",
        "vue.js": "vue",
        "vue 3": "vue",
    }
    return aliases.get(text, text)


def build_report(vacancy: dict, resume: dict) -> dict:
    """Compare extracted facts; a keyword mention does not prove full suitability."""
    rows = []
    candidates = [
        entity
        for entity in resume.get("entities", [])
        if isinstance(entity, dict) and entity.get("evidence")
    ]
    for requirement in vacancy.get("entities", []):
        if (
            not isinstance(requirement, dict)
            or requirement.get("defaulted")
            or not requirement.get("evidence")
            or requirement.get("value") is None
            or requirement.get("category")
            in {"position_name", "typical_position", "salary_min", "salary_max"}
        ):
            continue
        category = requirement.get("category")
        value = requirement["value"]
        matches = [
            entity
            for entity in candidates
            if entity.get("category") == category
            and entity.get("value") is not None
            and _key(entity["value"]) == _key(value)
        ]
        negatives = [entity for entity in matches if entity.get("polarity") == "absent"]
        positives = [entity for entity in matches if entity.get("polarity") != "absent"]
        evidence = (negatives or positives or [None])[0]
        if negatives and positives:
            status = "partial"
            comment = "Есть противоречивые сведения; требуется уточнение."
        elif negatives:
            status = "contradicted"
            comment = "В резюме указано отрицание; проверьте его контекст."
        elif positives:
            status = "partial"
            comment = "Факт упомянут; уровень и контекст нужно проверить."
        else:
            status = "not_stated"
            comment = "Подтверждение не найдено; это не означает отсутствие навыка."
        rows.append(
            {
                "requirement": requirement.get("display_value", value),
                "category": category,
                "importance": requirement.get("importance"),
                "vacancy_evidence": requirement["evidence"],
                "vacancy_source": requirement,
                "resume_evidence": evidence.get("evidence") if evidence else None,
                "resume_source": evidence,
                "status": status,
                "comment": comment,
            }
        )
    strengths = [row for row in rows if row["status"] == "matched"]
    # Positive lexical matches are potential strengths, not confirmed full matches.
    strengths.extend(
        row
        for row in rows
        if row["status"] == "partial"
        and row["resume_source"].get("polarity") != "absent"
    )
    unresolved = [row for row in rows if row["status"] != "matched"]
    return {
        "vacancy_id": vacancy.get("id"),
        "resume_id": resume.get("id"),
        "preliminary": True,
        "disclaimer": "Предварительное сопоставление; контекст требует проверки.",
        "requirements": rows,
        "strengths": strengths,
        "risks": unresolved,
        "questions": [
            f"Как ваш опыт соотносится с требованием «{row['vacancy_evidence']}»?"
            for row in unresolved
        ],
    }
