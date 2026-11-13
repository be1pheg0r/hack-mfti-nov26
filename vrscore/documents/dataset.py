"""Bridge extraction contracts to the existing strict JSONL dataset schema.

Dataset ``sections`` are semantic, potentially overlapping spans; they are NOT
physical blocks. Import therefore exposes ``record.text`` as one original block
without renormalising/rebuilding it. Its source is the dataset representation,
not an assertion that the publisher's full original document is available.
"""

import copy
import hashlib
import re

from .contracts import ParsedDocument, SourceLocation, TextBlock


def from_dataset_record(record: dict) -> ParsedDocument:
    """Adapt a dataset record without changing its ID, text or section offsets.

    Args:
        record: Existing resume/vacancy JSONL record with a valid text hash.

    Returns:
        An immutable text view; semantic sections remain in the input record.

    Raises:
        ValueError: Text hash, record identity or source section is inconsistent.
    """
    text = record["text"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != record["text_sha256"] or not record["id"] or not text.strip():
        raise ValueError("Invalid dataset identity, text or text_sha256")
    for section in record["sections"]:
        start, end = section["start"], section["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or text[start:end]
            != _render_attribute(record["attributes"][section["source_field"]])
        ):
            raise ValueError("Dataset section differs from its source text")
    parser = "dataset-text-v1"
    version = hashlib.sha256(f"{parser}:{digest}".encode("utf-8")).hexdigest()
    block = TextBlock(
        record["id"],
        version,
        "b000000",
        "paragraph",
        text,
        text,
        0,
        len(text),
        SourceLocation("record.text"),
    )
    quality = record.get("source", {}).get("text_completeness")
    warnings = (f"source_quality:{quality}",) if quality else ()
    return ParsedDocument(
        record["id"], version, parser, "identity-v1", text, (block,), warnings
    )


def _render_attribute(value: str | int | float | list[str]) -> str:
    """Use the dataset's existing list/numeric rendering, never rebuild its text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "; ".join(value)
    if type(value) in (int, float):
        return str(value)
    raise ValueError("Unsupported dataset attribute representation")


def to_dataset_record(
    document: ParsedDocument,
    *,
    id: str,
    kind: str,
    sampling_group: str,
    title: str,
    source: dict,
) -> dict:
    """Export DOCX as a description in the MFTI mixed-source record schema.

    Metadata is required from the caller: role, group, publisher and identity
    are not guessed from text or file names. Extraction provenance is nested
    under ``source.document`` because top-level additional properties and extra
    section fields are forbidden. Physical blocks remain in the stored artifact.
    Historical-only trudvsem schemas cannot represent local DOCX provenance;
    their source constants must not be fabricated to force compatibility.

    Raises:
        ValueError: Metadata or text cannot satisfy the current dataset schema.
    """
    if kind not in ("resume", "vacancy") or not re.fullmatch(
        rf"{kind}_[0-9]{{3,}}", id
    ):
        raise ValueError("Dataset id must match its kind and existing ID format")
    if sampling_group not in {
        "it",
        "finance",
        "education",
        "engineering",
        "office_service",
    }:
        raise ValueError("Unknown dataset sampling_group")
    if not title.strip() or len(document.text) < 100:
        raise ValueError("Dataset requires a title and at least 100 text characters")
    required = {
        "publisher",
        "acquisition",
        "original_url",
        "source_id",
        "source_id_kind",
    }
    if not required <= source.keys() or not all(
        isinstance(source[k], str) for k in required
    ):
        raise ValueError("Dataset requires explicit source provenance strings")
    provenance = copy.deepcopy(source)
    if "document" in provenance:
        raise ValueError("source.document is reserved for extraction provenance")
    provenance["document"] = {
        "document_id": document.document_id,
        "text_version": document.text_version,
        "parser_version": document.parser_version,
        "normalization": document.normalization,
        "warnings": list(document.warnings),
    }
    return {
        "id": id,
        "kind": kind,
        "sampling_group": sampling_group,
        "title": title,
        "text": document.text,
        "text_sha256": document.text_sha256,
        "attributes": {"description": document.text},
        "sections": [
            {
                "source_field": "description",
                "label": "Описание",
                "start": 0,
                "end": len(document.text),
            }
        ],
        "source": provenance,
    }
