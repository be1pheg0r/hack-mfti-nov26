"""Prepare a local, minimally redacted resume and vacancy ranking input.

Regex redaction is imperfect and does not guarantee anonymity. Review input
and obtain authorization before passing its hash to the ranking command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from vrscore.deidentification import deidentify

# Heuristic only: Cyrillic family-name endings plus given name and patronymic.
_NAME = re.compile(
    r"\b[А-ЯЁ][а-яё]+(?:ов|ова|ев|ева|ин|ина|ский|ская|ко)\s+"
    r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+(?:вич|вна|ич|ична))?\b"
)


def prepare(corpus: Path, output: Path, resume_index: int, vacancy_count: int) -> str:
    """Prepare one resume and uniquely identified vacancies from JSONL files."""

    def rows(name: str) -> list[dict]:
        with (corpus / name).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    resumes, vacancies = rows("resumes.jsonl"), rows("vacancies.jsonl")
    if resume_index < 0 or resume_index >= len(resumes):
        raise ValueError("Invalid resume index")
    if vacancy_count < 1 or vacancy_count > len(vacancies):
        raise ValueError("Invalid vacancy count")
    selected = vacancies[:vacancy_count]
    if len({row["id"] for row in selected}) != vacancy_count:
        raise ValueError("Vacancy IDs are not unique")
    resume = resumes[resume_index]
    text = _NAME.sub("[NAME]", deidentify(resume["text"]))
    payload = {
        "resume": {"id": "resume", "text": text},
        "vacancies": [
            {"id": f"v{i:03d}", "text": row["text"]}
            for i, row in enumerate(selected, 1)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    mapping = {
        "resume": resume["id"],
        "vacancies": {f"v{i:03d}": row["id"] for i, row in enumerate(selected, 1)},
    }
    output.with_name(output.stem + ".mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> None:
    """Prepare input without making any external requests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(".data/trudvsem_medium"))
    parser.add_argument(
        "--output", type=Path, default=Path(".data/llm-ranking/input.json")
    )
    parser.add_argument("--resume-index", type=int, default=0)
    parser.add_argument("--vacancy-count", type=int, default=100)
    args = parser.parse_args()
    print(
        "Input SHA-256:",
        prepare(args.corpus, args.output, args.resume_index, args.vacancy_count),
    )


if __name__ == "__main__":
    main()
