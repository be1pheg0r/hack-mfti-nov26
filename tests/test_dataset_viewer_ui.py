"""Regression checks for deep-link presentation in the dataset viewer."""

from pathlib import Path

APP_JS = Path(__file__).resolve().parents[1] / "vrscore/web/dataset_viewer/app.js"


def test_deep_linked_resume_opens_ranked_matches_when_available():
    """A matching resume deep link must select matches and reveal its reader."""
    app = APP_JS.read_text(encoding="utf-8")

    assert "state.tab = revealMatches && hasMatches ? 'matches' : 'overview'" in app
    assert "selectRecord(hash,false,true)" in app
    assert "panel.scrollIntoView({ block: 'start' })" in app


def test_resume_without_pairs_and_non_resume_keep_overview_default():
    """Match-tab default is conditional and must not leak to other records."""
    app = APP_JS.read_text(encoding="utf-8")

    assert (
        "r.kind === 'resume' && state.pairs.some(pair => pair.resume_id === r.id)"
        in app
    )
    assert "state.tab = revealMatches && hasMatches ? 'matches' : 'overview'" in app
