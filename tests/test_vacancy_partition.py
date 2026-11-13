"""Check evidence-preserving vacancy partitions."""

import json

from vacancy_parser.partition import partition_dataset


def test_partition_preserves_evidence_and_rejects_unknown_sources(tmp_path):
    """HR/hh and academic records are disjoint while cross-links survive."""
    source = tmp_path / "source"
    source.mkdir()
    records = [
        {"id": "hr", "source": {"source_id_kind": "mipt_hr_id"}},
        {"id": "hh", "source": {"source_id_kind": "hh_vacancy_id"}},
        {"id": "science", "source": {"source_id_kind": "competition_vacancy_id"}},
        {"id": "pps", "source": {"source_id_kind": "mipt_pps_department_position"}},
    ]

    def save(name, rows):
        (source / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    save("vacancies.jsonl", records)
    save("requirements.jsonl", [{"id": r["id"], "items": []} for r in records])
    save("skill_mentions.jsonl", [{"id": r["id"], "mentions": []} for r in records])
    save(
        "parsing_audit.jsonl",
        [
            {
                "id": r["id"],
                "derived": {},
                "source_evidence": {"restored_description": False},
                "contact_tail_removed": False,
            }
            for r in records
        ],
    )
    save("alternative_records.jsonl", [{"duplicate_of": "hr", "record": {"id": "alt"}}])
    (source / "duplicate_resolutions.json").write_text(
        json.dumps({"merges": [{"keep": "hr", "remove": "alt"}]})
    )
    save("source_evidence.jsonl", [{"id": "hr"}, {"id": "alt"}, {"id": "science"}])
    save(
        "duplicate_groups.jsonl", [{"canonical_id": "hr"}, {"canonical_id": "science"}]
    )
    save("uncertain_duplicates.jsonl", [{"canonical_ids": ["hr", "science"]}])
    (source / "manifest.json").write_text(
        json.dumps(
            {"name": "fixture", "actual": {"vacancies": 4, "resumes": 0}, "labels": {}}
        )
    )
    (source / "schema.json").write_text("{}")
    (source / "resumes.jsonl").write_text("")
    main, appendix = tmp_path / "main", tmp_path / "appendix"
    partition_dataset(
        source,
        main,
        appendix,
        main_kinds=["mipt_hr_id", "hh_vacancy_id"],
        appendix_kinds=["competition_vacancy_id", "mipt_pps_department_position"],
    )

    def read(p, n):
        return [json.loads(line) for line in (p / n).read_text().splitlines()]

    assert {r["id"] for r in read(main, "vacancies.jsonl")} == {"hr", "hh"}
    assert {r["id"] for r in read(appendix, "vacancies.jsonl")} == {"science", "pps"}
    assert {r["id"] for r in read(main, "source_evidence.jsonl")} == {"hr", "alt"}
    assert json.loads((main / "duplicate_resolutions.json").read_text())["merges"] == [
        {"keep": "hr", "remove": "alt"}
    ]
    assert {r["id"] for r in read(appendix, "source_evidence.jsonl")} == {"science"}
    assert len(read(main, "uncertain_duplicates.jsonl")) == 1
    assert len(read(appendix, "uncertain_duplicates.jsonl")) == 1
    assert json.loads((main / "manifest.json").read_text())["actual"]["vacancies"] == 2
    assert (main / "resumes.jsonl").resolve() == (source / "resumes.jsonl").resolve()
    records[0]["source"]["source_id_kind"] = "unknown"
    save("vacancies.jsonl", records)
    import pytest

    with pytest.raises(ValueError, match="Unknown source"):
        partition_dataset(
            source,
            tmp_path / "main2",
            tmp_path / "appendix2",
            main_kinds=["mipt_hr_id", "hh_vacancy_id"],
            appendix_kinds=["competition_vacancy_id", "mipt_pps_department_position"],
        )
