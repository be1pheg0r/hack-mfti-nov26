"""Conservative, source-preserving comparison for preliminary vacancy reports."""

import math
import re
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "report.yaml"
NEGATION = re.compile(
    r"\b(?:нет|отсутств\w*|без\s+опыта|never|no\s+experience)\b"
    r"|\bне\s+(?:работал\w*|использовал\w*|занимал\w*|знаком\w*|име\w*)\b",
    re.IGNORECASE,
)
COMMERCIAL = re.compile(r"коммерческ|commercial", re.IGNORECASE)
QUALIFIED = re.compile(
    r"коммерческ|практическ|уверенн|хорош\w*\s+знан|"
    r"глубок|самостоятельн|уровн|\bопыт\b|expert|proficien",
    re.IGNORECASE,
)
EXCLUDED = {"position_name", "typical_position", "salary_min", "salary_max"}
IMPORTANCE = {"strict": 3, "good2have": 2, "honestly_desired": 1, None: 0}
ACTIONABLE = {"experience", "education", "work_format", "work_schedule", "hard_skill"}
GENERIC_EXPERIENCE_WORDS = {
    "опыт",
    "опыта",
    "опытом",
    "работа",
    "работы",
    "работе",
    "от",
    "до",
    "не",
    "менее",
    "более",
    "год",
    "года",
    "лет",
    "годов",
    "г",
    "месяц",
    "месяца",
    "месяцев",
    "общий",
    "общего",
    "коммерческий",
    "коммерческой",
    "коммерческого",
    "требуется",
    "требования",
    "и",
    "с",
    "в",
    "на",
    "по",
}


def _settings(config=None):
    """Resolve the report's configurable question limit and skill aliases."""
    defaults = OmegaConf.load(DEFAULT_CONFIG)
    merged = OmegaConf.merge(defaults, OmegaConf.create(config or {}))
    settings = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(settings, dict):
        raise ValueError("report configuration must be a mapping")
    limit = settings["max_questions"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("report.max_questions must be a non-negative integer")
    return settings


def _normal(value):
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _key(value, settings):
    value = _normal(value)
    return settings["skill_aliases"].get(value, value)


def _base(value, settings):
    key = _key(value, settings)
    return settings["related_skill_aliases"].get(key, key)


def _text(entity):
    return " ".join(str(entity.get(key) or "") for key in ("evidence", "context"))


def _commercial(entity):
    return bool(COMMERCIAL.search(_text(entity)))


def _unique_sources(sources):
    """Preserve distinct original citations in document order."""
    result = []
    for source in sources:
        if source not in result:
            result.append(source)
    return result


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _experience(requirement, candidates):
    """Compare evidenced durations, not unrelated skills or unverified sums."""
    threshold = _number(requirement.get("value"))
    valid = []
    ambiguous_sum = False
    for candidate in candidates:
        if candidate.get("polarity") == "absent":
            continue
        if candidate.get("derived"):
            ambiguous_sum = True
            sources = candidate.get("sources") or []
        else:
            sources = [candidate]
        for source in sources:
            number = _number(source.get("value"))
            if source.get("evidence") and number is not None:
                valid.append((number, source))
    if threshold is None or not valid:
        return "not_stated", "Подходящее подтверждение опыта не найдено.", [], False
    best, source = max(valid, key=lambda item: item[0])
    citations = _unique_sources([source, *(item[1] for item in valid)])
    if _commercial(requirement) and not _commercial(source):
        return (
            "partial",
            "Указан опыт, но его коммерческий характер нужно уточнить.",
            citations,
            False,
        )
    subject_words = set(re.findall(r"[^\W\d_]+", _text(requirement).casefold()))
    if subject_words - GENERIC_EXPERIENCE_WORDS:
        return (
            "partial",
            "Указан общий опыт; соответствие требуемому направлению нужно уточнить.",
            citations,
            False,
        )
    if best >= threshold:
        return (
            "matched",
            f"Указано {best:g} г. при требовании от {threshold:g} г.",
            citations,
            False,
        )
    if ambiguous_sum or len(valid) > 1:
        return (
            "partial",
            "Указаны периоды опыта; их пересечения не проверены.",
            citations,
            False,
        )
    return (
        "contradicted",
        f"Указано {best:g} г. при требовании от {threshold:g} г.; "
        "уточните актуальный опыт.",
        citations,
        True,
    )


def _skill_relation(requirement, candidate, settings):
    left, right = (
        _key(requirement["value"], settings),
        _key(candidate["value"], settings),
    )
    if _base(left, settings) != _base(right, settings):
        return None
    # A general mention of a technology does not establish its specific version.
    original = _normal(requirement["value"])
    candidate_original = _normal(candidate["value"])
    versioned_vue = original in {"vue 3", "vue3"} and candidate_original not in {
        "vue 3",
        "vue3",
    }
    return "related" if left != right or versioned_vue else "exact"


def _compare(requirement, candidates, settings):
    """Return status, explanation, citations and whether a real risk is evidenced."""
    category = requirement.get("category")
    same = [c for c in candidates if c.get("category") == category]
    if category == "experience":
        return _experience(requirement, same)
    if category == "hard_skill":
        matches = [(c, _skill_relation(requirement, c, settings)) for c in same]
        matches = [(c, relation) for c, relation in matches if relation]
    else:
        matches = [
            (c, "exact")
            for c in same
            if _normal(c["value"]) == _normal(requirement["value"])
        ]
    if not matches:
        return (
            "not_stated",
            "Подходящее подтверждение не найдено среди извлечённых данных; "
            "это не означает отсутствия навыка или опыта.",
            [],
            False,
        )
    sources = _unique_sources([item for item, _ in matches])
    positive = [
        (c, relation) for c, relation in matches if c.get("polarity") != "absent"
    ]
    negative = [c for c, _ in matches if c.get("polarity") == "absent"]
    confirmed_negative = [c for c in negative if NEGATION.search(_text(c))]
    uncertain_negative = len(confirmed_negative) != len(negative)
    broad_negative = [c for c in confirmed_negative if not _commercial(c)]
    scoped_negative = [c for c in confirmed_negative if _commercial(c)]
    applicable = broad_negative or (scoped_negative if _commercial(requirement) else [])
    if positive and applicable:
        return (
            "partial",
            "Есть положительные и отрицательные сведения; "
            "проверьте контекст и период опыта.",
            sources,
            True,
        )
    if applicable and not uncertain_negative:
        return (
            "contradicted",
            "В резюме указано отрицание по этому требованию; проверьте контекст.",
            sources,
            True,
        )
    if negative:
        if scoped_negative and not _commercial(requirement) and not broad_negative:
            return (
                "partial",
                "Коммерческий опыт отрицается; опыт в других проектах нужно уточнить.",
                sources,
                False,
            )
        return (
            "partial",
            "Отрицание требует проверки в исходном контексте.",
            sources,
            False,
        )
    exact = any(relation == "exact" for _, relation in positive)
    explicit = any(c.get("polarity") == "present" for c, _ in positive)
    structured = category in {"work_format", "work_schedule", "employment_type"}
    if (
        exact
        and (explicit or structured)
        and (structured or not QUALIFIED.search(_text(requirement)))
    ):
        return (
            "matched",
            "Указанное требование подтверждено фрагментом резюме.",
            sources,
            False,
        )
    return (
        "partial",
        "Есть связанное упоминание; требуемый уровень и контекст нужно уточнить.",
        sources,
        False,
    )


def _questions(rows, limit):
    """Prioritize material issues and avoid repeated questions about one source."""
    ordered = sorted(
        rows,
        key=lambda row: (
            {"contradicted": 0, "not_stated": 1, "partial": 2}.get(row["status"], 3),
            -IMPORTANCE.get(row["importance"], 0),
        ),
    )
    questions, seen = [], set()
    for row in ordered:
        if len(questions) >= limit:
            break
        if row["category"] not in ACTIONABLE:
            continue
        if row["importance"] != "strict" and not (
            row["importance"] == "good2have" and row["risk"]
        ):
            continue
        if row["status"] == "matched":
            continue
        source = row["vacancy_source"]
        reference = (
            source.get("document_id"),
            source.get("block_id") or _normal(source.get("evidence")),
        )
        if reference in seen:
            continue
        seen.add(reference)
        if row["category"] == "experience":
            question = (
                "Каков ваш актуальный релевантный опыт и какие периоды в него входят?"
            )
        elif row["category"] == "work_format":
            question = "Подходит ли вам указанный в вакансии формат работы?"
        elif row["category"] == "work_schedule":
            question = "Подходит ли вам указанный в вакансии график?"
        elif row["category"] == "education":
            question = "Какое образование вы завершили?"
        else:
            question = (
                f"Расскажите об опыте работы с «{row['requirement']}» и его контексте."
            )
        if question not in questions:
            questions.append(question)
    return questions


def build_report(vacancy: dict, resume: dict, config=None) -> dict:
    """Compare grounded facts without equating missing data with absence."""
    settings = _settings(config)
    candidates = [
        entity
        for entity in resume.get("entities", [])
        if isinstance(entity, dict)
        and entity.get("evidence")
        and entity.get("value") is not None
    ]
    grouped = {}
    for item in vacancy.get("entities", []):
        if (
            not isinstance(item, dict)
            or item.get("defaulted")
            or not item.get("evidence")
            or item.get("value") is None
            or item.get("category") in EXCLUDED
        ):
            continue
        category = item.get("category")
        value = (
            _key(item["value"], settings)
            if category == "hard_skill"
            else _normal(item["value"])
        )
        if category == "hard_skill" and _normal(item["value"]) in {"vue 3", "vue3"}:
            value = "vue 3"
        # Commercial experience is a separate requirement, not a synonym for use.
        group_key = (category, value, _commercial(item))
        grouped.setdefault(group_key, []).append(item)
    rows = []
    for requirements in grouped.values():
        requirements = _unique_sources(requirements)
        requirement = max(
            requirements, key=lambda item: IMPORTANCE.get(item.get("importance"), 0)
        )
        status, comment, sources, risk = _compare(requirement, candidates, settings)
        primary = sources[0] if sources else None
        rows.append(
            {
                "requirement": requirement.get("display_value", requirement["value"]),
                "category": requirement.get("category"),
                "importance": requirement.get("importance"),
                "vacancy_evidence": requirement["evidence"],
                "vacancy_source": requirement,
                "vacancy_sources": requirements,
                "resume_evidence": primary.get("evidence") if primary else None,
                "resume_source": primary,
                "resume_sources": sources,
                "status": status,
                "comment": comment,
                "risk": risk,
            }
        )
    strengths = [
        row
        for row in rows
        if row["status"] == "matched"
        or (
            row["status"] == "partial"
            and not row["risk"]
            and any(
                source.get("polarity") == "present" for source in row["resume_sources"]
            )
        )
    ]
    risks = [row for row in rows if row["risk"]]
    unconfirmed = [row for row in rows if row["status"] == "not_stated"]
    return {
        "vacancy_id": vacancy.get("id"),
        "resume_id": resume.get("id"),
        "preliminary": True,
        "disclaimer": "Предварительное сопоставление; контекст требует проверки.",
        "requirements": rows,
        "strengths": strengths,
        "risks": risks,
        "unconfirmed": unconfirmed,
        "questions": _questions(rows, settings["max_questions"]),
    }
