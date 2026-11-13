"""Synthetic-only coverage for conservative resume de-identification."""

import json
import subprocess
import sys

from vrscore.deidentification import deidentify


def test_redacts_structured_and_labeled_russian_pii_without_touching_skills():
    resume = {
        "ФИО": "Марина Тестова",
        "email": "marina@example.test",
        "телефон": "+7 (999) 123-45-67",
        "skills": ["Python 3.12", "C++17", "SQL"],
        "summary": (
            "ФИО: Марина Тестова\nНавыки: Python 3.12\n"
            "Связь: marina@example.test, +7 (999) 123-45-67\n"
            "Профиль: https://example.test/u/marina"
        ),
    }

    result = deidentify(resume)

    assert result["ФИО"] == "[NAME]"
    assert result["email"] == "[EMAIL]"
    assert result["телефон"] == "[PHONE]"
    assert result["skills"] == ["Python 3.12", "C++17", "SQL"]
    assert "Марина Тестова" not in result["summary"]
    assert "marina@example.test" not in result["summary"]
    assert "+7 (999) 123-45-67" not in result["summary"]
    assert "https://example.test/u/marina" not in result["summary"]
    assert "Python 3.12" in result["summary"]


def test_redacts_common_russian_phone_forms_and_plain_links():
    text = "Тел.: 8 800 555-35-35; https://t.me/example; example.test/path"

    result = deidentify(text)

    assert result == "Тел.: [PHONE]; [LINK]; [LINK]"


def test_common_technology_names_are_not_redacted_as_domains():
    assert deidentify("Node.js, Vue.js, Next.js и ASP.NET; example.com/u") == (
        "Node.js, Vue.js, Next.js и ASP.NET; [LINK]"
    )


def test_explicit_names_are_redacted_but_unlabeled_words_are_not_guessed():
    text = "Алексей Образцов — специалист; команда Образцов использует Python."

    result = deidentify(text, names=["Алексей Образцов"])

    assert result == "[NAME] — специалист; команда Образцов использует Python."


def test_structured_name_is_redacted_in_nested_text():
    resume = {"candidate_name": "Марина Тестова", "text": "Марина Тестова, Python"}
    assert deidentify(resume) == {"candidate_name": "[NAME]", "text": "[NAME], Python"}


def test_hydra_cli_writes_redacted_json_with_path_overrides(tmp_path):
    source = tmp_path / "input.json"
    destination = tmp_path / "output.json"
    source.write_text(
        json.dumps({"email": "demo@example.test", "skills": ["Python 3.12"]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vrscore.deidentify_cli",
            f"input_path={source}",
            f"output_path={destination}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "email": "[EMAIL]",
        "skills": ["Python 3.12"],
    }
    assert "demo@example.test" not in completed.stdout + completed.stderr


def test_cli_refuses_to_overwrite_input(tmp_path):
    source = tmp_path / "input.json"
    original = json.dumps({"email": "test@example.test"})
    source.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vrscore.deidentify_cli",
            f"input_path={source}",
            f"output_path={source}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert source.read_text() == original
