"""Check that private project files are excluded from Git."""

import subprocess
from pathlib import Path


def test_private_files_are_ignored() -> None:
    """Protect credentials and private working documents from commits."""
    root = Path(__file__).resolve().parents[1]
    paths = [".env", ".env.local", ".docs/AGENTS.md", ".data/example.txt"]
    result = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(paths)
