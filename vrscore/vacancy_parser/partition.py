"""Partition a structured corpus into standard vacancies and an appendix."""

import collections
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .dataset import _hash, _jsonl

ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def partition_dataset(
    source: Path,
    main: Path,
    appendix: Path,
    *,
    main_kinds: list[str],
    appendix_kinds: list[str],
) -> None:
    """Write two new compatible corpora while preserving the structured input.

    Unknown or overlapping source categories fail closed. Cross-partition
    uncertain-duplicate references remain in both relevant review files.

    Args:
        source: Existing complete structured corpus.
        main: New directory for the main corpus.
        appendix: New directory for the academic appendix.
        main_kinds: Explicit source identifiers included in the main corpus.
        appendix_kinds: Explicit source identifiers included in the appendix.
    """
    source, main, appendix = (
        Path(source).resolve(),
        Path(main).absolute(),
        Path(appendix).absolute(),
    )
    if (
        main == appendix
        or main.exists()
        or main.is_symlink()
        or appendix.exists()
        or appendix.is_symlink()
    ):
        raise FileExistsError("Partition output already exists or outputs coincide")
    kinds = [set(main_kinds), set(appendix_kinds)]
    if not all(kinds) or kinds[0] & kinds[1]:
        raise ValueError("Partition source types must be nonempty and disjoint")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(source / "vacancies.jsonl")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) != manifest["actual"]["vacancies"]:
        raise ValueError("Source vacancies do not match manifest or IDs are duplicated")
    unknown = {r["source"].get("source_id_kind") for r in rows} - kinds[0] - kinds[1]
    if unknown:
        raise ValueError(f"Unknown source types: {sorted(map(str, unknown))}")
    requirements = _read_jsonl(source / "requirements.jsonl")
    mentions = _read_jsonl(source / "skill_mentions.jsonl")
    audits = _read_jsonl(source / "parsing_audit.jsonl")
    alternatives = _read_jsonl(source / "alternative_records.jsonl")
    evidence = _read_jsonl(source / "source_evidence.jsonl")
    groups = _read_jsonl(source / "duplicate_groups.jsonl")
    uncertain = _read_jsonl(source / "uncertain_duplicates.jsonl")
    for side, output, source_kinds in zip(
        ("main", "appendix"), (main, appendix), kinds, strict=True
    ):
        chosen = [r for r in rows if r["source"]["source_id_kind"] in source_kinds]
        chosen_ids = {r["id"] for r in chosen}
        selected_alts = [r for r in alternatives if r["duplicate_of"] in chosen_ids]
        evidence_ids = chosen_ids | {r["record"]["id"] for r in selected_alts}
        selected_audits = [r for r in audits if r["id"] in evidence_ids]
        selected_requirements = [r for r in requirements if r["id"] in chosen_ids]
        selected_mentions = [r for r in mentions if r["id"] in chosen_ids]
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".partition-", dir=output.parent
        ) as tmp:
            stage = Path(tmp)
            for name, data in {
                "vacancies.jsonl": chosen,
                "requirements.jsonl": selected_requirements,
                "skill_mentions.jsonl": selected_mentions,
                "parsing_audit.jsonl": selected_audits,
                "alternative_records.jsonl": selected_alts,
                "source_evidence.jsonl": [
                    r for r in evidence if r["id"] in evidence_ids
                ],
                "duplicate_groups.jsonl": [
                    r for r in groups if r["canonical_id"] in chosen_ids
                ],
                "uncertain_duplicates.jsonl": [
                    r for r in uncertain if chosen_ids.intersection(r["canonical_ids"])
                ],
            }.items():
                _jsonl(stage / name, data)
            shutil.copy2(source / "schema.json", stage / "schema.json")
            resolution_file = source / "duplicate_resolutions.json"
            if resolution_file.is_file():
                resolutions = json.loads(resolution_file.read_text(encoding="utf-8"))
                resolutions["merges"] = [
                    r for r in resolutions["merges"] if r["keep"] in chosen_ids
                ]
                (stage / "duplicate_resolutions.json").write_text(
                    json.dumps(resolutions, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if (source / "skill_vocabulary.json").is_file():
                shutil.copy2(
                    source / "skill_vocabulary.json", stage / "skill_vocabulary.json"
                )
            os.symlink(
                os.path.relpath((source / "resumes.jsonl").resolve(), output),
                stage / "resumes.jsonl",
            )
            description = (
                "Основной набор: вакансии HR-портала и hh."
                if side == "main"
                else "Приложение: научные конкурсы и группы должностей ППС."
            )
            (stage / "README.md").write_text(
                f"# {description}\n\n"
                "Схема и фрагменты требований совместимы с исходным "
                "структурированным набором. Резюме подключены символической "
                "ссылкой; переносить их нужно отдельно. "
                "Сомнительные дубли могут ссылаться на записи из другого раздела. "
                "Отсутствие требования в тексте не означает его отсутствия. "
                "Метки соответствия пар резюме–вакансия отсутствуют.\n\n"
                "Упоминания навыков — лексические совпадения с контекстом, "
                "не отдельные "
                "обязательные требования. Научные поисковые выдержки неполны; "
                "требования ППС общие, не индивидуальные. Права перераспространения "
                "и анонимизация не гарантированы.\n",
                encoding="utf-8",
            )
            result = {
                "name": "mfti_vacancies_standard"
                if side == "main"
                else "mfti_vacancies_appendix",
                "version": "1.0.0",
                "actual": {
                    "resumes": manifest["actual"]["resumes"],
                    "vacancies": len(chosen),
                },
                "labels": manifest.get("labels", {}),
                "provenance": manifest.get("provenance", {}),
                "quality": dict(
                    collections.Counter(
                        r["source"].get("text_completeness", "unknown") for r in chosen
                    )
                ),
                "partition": {
                    "role": side,
                    "source_id_kinds": sorted(source_kinds),
                    "parent_dataset": os.path.relpath(source, output),
                    "parent_vacancies_sha256": _hash(source / "vacancies.jsonl"),
                    "uncertain_duplicates_may_cross_partitions": True,
                },
                "parsing": {
                    "records_with_requirement_items": sum(
                        bool(r["items"]) for r in selected_requirements
                    ),
                    "requirement_items": sum(
                        len(r["items"]) for r in selected_requirements
                    ),
                    "records_with_lexical_skill_mentions": sum(
                        bool(r["mentions"]) for r in selected_mentions
                    ),
                    "lexical_skill_mentions": sum(
                        len(r["mentions"]) for r in selected_mentions
                    ),
                    "audit_records_including_alternatives": len(selected_audits),
                },
                "sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in stage.iterdir()
                    if p.is_file() and not p.is_symlink()
                },
            }
            (stage / "manifest.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            stage.rename(output)


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "configs"),
    config_name="vacancy_partition",
)
def main(config: DictConfig) -> None:
    """Run the configured split; existing output directories are never replaced."""

    def resolve(path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else ROOT / value

    partition_dataset(
        resolve(config.source_dir),
        resolve(config.main_dir),
        resolve(config.appendix_dir),
        main_kinds=list(config.main_kinds),
        appendix_kinds=list(config.appendix_kinds),
    )
    print(
        f"Partitioned {config.source_dir} into "
        f"{config.main_dir} and {config.appendix_dir}"
    )


if __name__ == "__main__":
    main()
