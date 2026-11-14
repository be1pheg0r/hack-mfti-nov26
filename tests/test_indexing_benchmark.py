"""Tests for safe, repeatable indexing benchmark helpers."""

import json

import pytest

from vrscore.benchmarking.indexing_benchmark import (
    RecordingChat,
    build_summary,
    make_row,
    safe_write_json,
)


def test_rows_and_summary_keep_metadata_without_source_text():
    """Summaries count outcomes and artifact rows omit sensitive source text."""
    rows = [
        {
            **make_row("vacancy", 1, {"id": "case-1", "text": "private text"}),
            "status": "ok",
        },
        {
            "kind": "resume",
            "iteration": 1,
            "status": "error",
            "error_type": "ValueError",
        },
    ]
    summary = build_summary(rows)
    assert summary["total"] == 2
    assert summary["success"] == 1
    assert summary["failure"] == 1
    assert "private text" not in json.dumps(rows)
    assert rows[0]["source_sha256"]
    assert (
        build_summary([{"kind": "resume", "status": "error", "llm": None}])["failure"]
        == 1
    )


def test_artifact_writer_never_overwrites_existing_output(tmp_path):
    """A repeat run gets a distinct destination instead of replacing results."""
    path = tmp_path / "results.json"
    safe_write_json(path, {"run": 1})
    with pytest.raises(FileExistsError):
        safe_write_json(path, {"run": 2})
    assert json.loads(path.read_text()) == {"run": 1}


def test_recording_chat_adds_unique_nonce_and_retains_usage_metadata():
    """Every extraction request is distinct and response usage is recorded."""

    class Response:
        id = "request-id"
        model = "model"
        provider = "provider"
        usage = {"cost": 0.12, "prompt_tokens_details": {"cached_tokens": 0}}
        choices = [type("Choice", (), {"finish_reason": "stop"})()]

    class Chat:
        def send(self, **kwargs):
            self.messages = kwargs["messages"]
            return Response()

    original = Chat()
    recording = RecordingChat(original)
    messages = [{"role": "system", "content": "extract"}]
    recording.send(messages=messages)
    nonce1 = original.messages[0]["content"].split("\n", 1)[0]
    recording.send(messages=messages)
    nonce2 = original.messages[0]["content"].split("\n", 1)[0]
    assert nonce1 != nonce2
    assert recording.last is not None
    assert recording.last["usage"]["cost"] == 0.12
    assert recording.last["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert messages[0]["content"] == "extract"


def test_recording_chat_rejects_reported_prompt_cache_hits():
    """Do not treat a provider-reported cached prompt as an uncached sample."""

    class Response:
        id = "cached-request"
        model = "model"
        provider = "provider"
        usage = {"prompt_tokens_details": {"cached_tokens": 3}}
        choices = [type("Choice", (), {"finish_reason": "stop"})()]

    class Chat:
        def send(self, **kwargs):
            return Response()

    recording = RecordingChat(Chat())
    with pytest.raises(RuntimeError, match="cached prompt"):
        recording.send(messages=[{"role": "system", "content": "extract"}])
    assert recording.last is not None
    assert recording.last["cache_usage_status"] == "cache_hit_rejected"


def test_summary_by_kind_and_unverified_cache():
    """Do not conflate kinds or silently accept missing cache telemetry."""
    summary = build_summary(
        [
            {
                "kind": "vacancy",
                "status": "ok",
                "embedding_seconds": 2.0,
                "llm": {"usage": {"cost": 0.1}},
            },
            {
                "kind": "resume",
                "status": "ok",
                "embedding_seconds": 4.0,
                "llm": {"usage": {"cost": 0.2}},
            },
        ]
    )
    vacancy = summary["by_kind"]["vacancy"]
    assert vacancy["durations_seconds"]["embedding_seconds"]["mean"] == 2
    assert summary["by_kind"]["resume"]["llm_cost_reported_mean"] == 0.2

    class Response:
        id = "request"
        model = "model"
        provider = "provider"
        usage = {"cost": 0.1}
        choices = [type("Choice", (), {"finish_reason": "stop"})()]

    class Chat:
        def send(self, **kwargs):
            return Response()

    recording = RecordingChat(Chat())
    with pytest.raises(RuntimeError, match="cache telemetry"):
        recording.send(messages=[{"role": "system", "content": "extract"}])
    assert recording.last["usage"]["cost"] == 0.1
