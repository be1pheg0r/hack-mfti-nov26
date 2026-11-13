"""Build separate, evidence-linked compatible vacancy datasets."""

import collections
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .evidence import apply_source_evidence
from .parser import (
    parse_record,
    redact_scientific_contacts,
    requirement_items,
)
from .skill_mentions import extract_mentions


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_dataset(
    source: Path,
    output: Path,
    *,
    evidence_path: Path | None = None,
    resolutions_path: Path | None = None,
    vocabulary_path: Path | None = None,
) -> dict:
    """Build a new corpus without modifying or overwriting the source.

    Args:
        source: Dataset directory containing manifest, schema and JSONL files.
        output: New dataset directory; must not already exist.
        evidence_path: Optional, source-verified per-record evidence JSONL.
        resolutions_path: Optional, hash-bound manual duplicate decisions.
        vocabulary_path: Optional, curated lexical vocabulary JSON.

    Returns:
        The generated dataset manifest.

    Raises:
        FileExistsError: Output already exists.
        ValueError: Input IDs, counts or source text offsets are inconsistent.
    """
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    originals = [
        json.loads(line)
        for line in (source / "vacancies.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(originals) != manifest["actual"]["vacancies"]:
        raise ValueError("Vacancy count differs from manifest")
    if len({r["id"] for r in originals}) != len(originals):
        raise ValueError("Duplicate vacancy ID")
    evidence = {}
    if evidence_path is not None:
        for line in Path(evidence_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in evidence:
                raise ValueError("Duplicate source evidence ID")
            evidence[item["id"]] = item
        if not set(evidence).issubset({r["id"] for r in originals}):
            raise ValueError("Source evidence contains an unknown vacancy ID")
    rows, audits, requirements = [], [], []
    for original in originals:
        if original.get("kind") != "vacancy":
            raise ValueError("Unexpected record kind")
        text = original["text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != original["text_sha256"]:
            raise ValueError(f"Invalid text hash: {original['id']}")
        for s in original["sections"]:
            if text[s["start"] : s["end"]] != original["attributes"][s["source_field"]]:
                raise ValueError(f"Invalid source section: {original['id']}")
        sanitized, redacted = redact_scientific_contacts(original)
        source_audit = {"restored_description": False, "fields": {}}
        if original["id"] in evidence:
            sanitized, source_audit = apply_source_evidence(
                sanitized, evidence[original["id"]]
            )
        record, audit = parse_record(sanitized)
        audit["contact_tail_removed"] = redacted
        audit["source_evidence"] = source_audit
        items = requirement_items(record)
        rows.append(record)
        audits.append(audit)
        requirements.append(
            {
                "id": record["id"],
                "source_quality": record.get("source", {}).get("text_completeness"),
                "items": items,
            }
        )
    alternatives = []
    parsed_by_id = {r["id"]: r for r in rows}
    if resolutions_path is not None:
        resolutions = json.loads(Path(resolutions_path).read_text(encoding="utf-8"))
        source_by_id = {r["id"]: r for r in originals}

        requirements_by_id = {r["id"]: r for r in requirements}
        removed = set()
        for decision in resolutions["merges"]:
            keep, drop = decision["keep"], decision["remove"]
            if (
                keep == drop
                or keep not in source_by_id
                or drop not in source_by_id
                or drop in removed
                or keep in removed
                or source_by_id[keep]["text_sha256"] != decision["keep_sha256"]
                or source_by_id[drop]["text_sha256"] != decision["remove_sha256"]
            ):
                raise ValueError("Invalid or stale duplicate resolution")
            alternatives.append(
                {
                    "duplicate_of": keep,
                    "record": parsed_by_id[drop],
                    "requirements": requirements_by_id[drop],
                    "reason": decision["reason"],
                }
            )
            removed.add(drop)
        if any(a["duplicate_of"] in removed for a in alternatives):
            raise ValueError("A kept duplicate target was also removed")
        rows = [r for r in rows if r["id"] not in removed]
        requirements = [r for r in requirements if r["id"] not in removed]
    mention_rows = []
    if vocabulary_path is not None:
        vocabulary = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))[
            "terms"
        ]
        mention_rows = [
            {
                "id": row["id"],
                "mentions": extract_mentions(row, req["items"], vocabulary),
            }
            for row, req in zip(rows, requirements, strict=True)
        ]
        for alternative in alternatives:
            alternative["skill_mentions"] = extract_mentions(
                alternative["record"], alternative["requirements"]["items"], vocabulary
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vacancy-parser-", dir=output.parent
    ) as tmp:
        stage = Path(tmp)
        _jsonl(stage / "vacancies.jsonl", rows)
        _jsonl(stage / "requirements.jsonl", requirements)
        _jsonl(stage / "parsing_audit.jsonl", audits)
        if vocabulary_path is not None:
            _jsonl(stage / "skill_mentions.jsonl", mention_rows)
            shutil.copy2(vocabulary_path, stage / "skill_vocabulary.json")
        if resolutions_path is not None:
            _jsonl(stage / "alternative_records.jsonl", alternatives)
            shutil.copy2(resolutions_path, stage / "duplicate_resolutions.json")
        if evidence_path is not None:
            shutil.copy2(evidence_path, stage / "source_evidence.jsonl")
        shutil.copy2(source / "schema.json", stage / "schema.json")
        duplicate_path = source / "duplicate_groups.jsonl"
        if duplicate_path.is_file():
            groups = [
                json.loads(line) for line in duplicate_path.read_text().splitlines()
            ]
            for alternate in alternatives:
                keep, dropped = alternate["duplicate_of"], alternate["record"]
                groups.append(
                    {
                        "canonical_id": keep,
                        "member_ids": [keep, dropped["id"]],
                        "sources": [
                            {"record_id": row["id"], "source": row["source"]}
                            for row in (parsed_by_id[keep], dropped)
                        ],
                        "evidence": [{"rationale": alternate["reason"]}],
                    }
                )
            _jsonl(stage / "duplicate_groups.jsonl", groups)
        uncertain_path = source / "uncertain_duplicates.jsonl"
        remaining_uncertain = None
        if uncertain_path.is_file():
            aliases = {a["record"]["id"]: a["duplicate_of"] for a in alternatives}
            uncertain = []
            for line in uncertain_path.read_text().splitlines():
                item = json.loads(line)
                item["canonical_ids"] = [
                    aliases.get(i, i) for i in item["canonical_ids"]
                ]
                if len(set(item["canonical_ids"])) > 1:
                    uncertain.append(item)
            _jsonl(stage / "uncertain_duplicates.jsonl", uncertain)
            remaining_uncertain = len(uncertain)
        os.symlink(
            os.path.relpath(source / "resumes.jsonl", output), stage / "resumes.jsonl"
        )
        result = copy.deepcopy(manifest)
        result["name"] = f"{manifest['name']}_structured"
        result["version"] = "0.4.0"
        result["actual"]["vacancies"] = len(rows)
        result["quality"] = dict(
            collections.Counter(r["source"].get("text_completeness") for r in rows)
        )
        if "deduplication" in result:
            result["deduplication"]["output_records"] = len(rows)
            result["deduplication"]["removed_as_duplicates"] += len(alternatives)
            result["deduplication"]["duplicate_groups"] += len(alternatives)
            result["deduplication"]["confirmed_pairs"] += len(alternatives)
            if remaining_uncertain is not None:
                result["deduplication"]["uncertain_pairs_retained"] = (
                    remaining_uncertain
                )
            result["deduplication"]["source_records"] = os.path.relpath(
                source / "vacancies_before_dedup.jsonl", output
            )
        result["parsing"] = {
            "input_vacancies_sha256": _hash(source / "vacancies.jsonl"),
            "processed": len(originals),
            "retained": len(rows),
            "additional_duplicates_resolved": len(alternatives),
            "records_with_lexical_skill_mentions": sum(
                bool(r["mentions"]) for r in mention_rows
            ),
            "lexical_skill_mentions": sum(len(r["mentions"]) for r in mention_rows),
            "records_with_requirement_items": sum(
                bool(r["items"]) for r in requirements
            ),
            "requirement_items": sum(len(r["items"]) for r in requirements),
            "derived_field_counts": dict(
                collections.Counter(key for audit in audits for key in audit["derived"])
            ),
            "scientific_contact_tails_removed": sum(
                audit["contact_tail_removed"] for audit in audits
            ),
            "source_evidence_records": len(evidence),
            "restored_descriptions": sum(
                audit["source_evidence"]["restored_description"] for audit in audits
            ),
            "method": "Exact source spans and conservative explicit-pattern extraction",
            "requirements_scope": (
                "PPS requirements are generic role-level, "
                "not specific to the department"
            ),
            "labels": "No candidate matching or relevance judgments",
        }
        (stage / "README.md").write_text(
            "# Структурированные вакансии МФТИ\n\n"
            "`vacancies.jsonl` — основной набор; `requirements.jsonl` — фрагменты "
            "требований с позициями в тексте. `parsing_audit.jsonl` содержит "
            "происхождение извлечённых полей и охватывает все входные записи.\n\n"
            "`skill_mentions.jsonl` — лексические совпадения со словарём внутри "
            "требований, а не перечень независимых обязательных навыков. Для них "
            "сохранены точный фрагмент, контекст и статус предпочтительности.\n\n"
            "`source_evidence.jsonl` хранит проверенные поля оригинальных карточек; "
            "`alternative_records.jsonl` — полные разобранные альтернативные "
            "публикации, исключённые как дубли. Оклад научной позиции указан отдельно "
            "как `baseSalary`, а не как полная зарплата. Период и налоговый статус "
            "не предполагаются, если они не указаны в источнике.\n\n"
            "Требования ППС являются общими для должности. Отдельные записи могут "
            "быть неполными выдержками. Некоторые карточки не содержат требований; "
            "отсутствие поля не означает отсутствия требования. Автоматические "
            "извлечения требуют ручной проверки, парных меток соответствия нет.\n\n"
            "Исходные данные, оставшиеся сомнительные дубли и ограничения "
            "перераспространения сохраняются в соседнем исходном наборе. "
            "Контактные строки научных карточек удалены из этой версии; проверка "
            "персональных данных не является исчерпывающей.\n",
            encoding="utf-8",
        )
        result["sha256"] = {
            path.name: _hash(path)
            for path in stage.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        (stage / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    return result
