"""Immutable, format-neutral text and evidence contracts.

Offsets are Python Unicode code-point indices, half-open [start, end), in
``ParsedDocument.text`` of the specified version, never byte or DOCX offsets.
Blocks are joined by exactly one LF. Source locations refer to whole original
blocks: normalisation does not claim a character-level mapping to Word XML.
"""

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol


class DocumentError(ValueError):
    """An actionable ingestion failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        """Attach a stable code without returning fabricated document content."""
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SourceLocation:
    """Whole-block source coordinates; DOCX paths include table/row/cell indices."""

    part: str
    xml_path: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class ListInfo:
    """Word list identity/level, not a fabricated rendered numbering label."""

    numbering_id: int
    level: int


@dataclass(frozen=True)
class TextBlock:
    """One paragraph, including a list item or paragraph inside a table cell."""

    document_id: str
    text_version: str
    block_id: str
    kind: Literal["paragraph", "list_item"]
    text: str
    original_text: str
    start: int
    end: int
    source: SourceLocation
    list_info: ListInfo | None = None


@dataclass(frozen=True)
class Evidence:
    """An exact literal quote tied to a document, text version and single block."""

    document_id: str
    text_version: str
    block_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """Immutable extraction; warnings disclose content outside the text scope.

    A changed block remains traceable by ``source`` and ``original_text``, but
    cannot supply literal quotes: use the whole-block reference instead.
    """

    document_id: str
    text_version: str
    parser_version: str
    normalization: str
    text: str
    blocks: tuple[TextBlock, ...]
    warnings: tuple[str, ...] = ()

    @property
    def text_sha256(self) -> str:
        """Hash UTF-8 text using the existing dataset's text_sha256 convention."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        """Reject inconsistent offsets, duplicate identities and foreign blocks."""
        cursor = 0
        identities = set()
        for block in self.blocks:
            if (
                block.document_id != self.document_id
                or block.text_version != self.text_version
                or block.block_id in identities
                or block.start != cursor
                or block.end != cursor + len(block.text)
                or self.text[block.start : block.end] != block.text
            ):
                raise ValueError("Inconsistent document block or offsets")
            identities.add(block.block_id)
            cursor = block.end + 1
        if self.text != "\n".join(block.text for block in self.blocks):
            raise ValueError("Document text must be the ordered block representation")

    def block(self, block_id: str) -> TextBlock:
        """Resolve a whole-block reference, rejecting nonexistent blocks."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise ValueError(f"Unknown block: {block_id}")

    def quote(self, block_id: str, start: int, end: int, text: str) -> Evidence:
        """Validate proposed literal text; never trim, repair or truncate it."""
        evidence = Evidence(
            self.document_id, self.text_version, block_id, start, end, text
        )
        self.resolve(evidence)
        return evidence

    def resolve(self, evidence: Evidence) -> str:
        """Validate all evidence identities, boundaries and original literal text."""
        if (
            evidence.document_id != self.document_id
            or evidence.text_version != self.text_version
        ):
            raise ValueError("Evidence refers to a different document or text version")
        block = self.block(evidence.block_id)
        if block.original_text != block.text:
            raise ValueError(
                "Changed block cannot supply a literal quote; cite the block"
            )
        if (
            not block.start <= evidence.start < evidence.end <= block.end
            or self.text[evidence.start : evidence.end] != evidence.text
        ):
            raise ValueError("Evidence text or offsets do not match the source block")
        return evidence.text


class DocumentReader(Protocol):
    """Small reader boundary for future, separately implemented formats."""

    supported_suffixes: tuple[str, ...]

    def read(self, data: bytes, *, filename: str) -> ParsedDocument:
        """Extract supported bytes or raise DocumentError; never run implicit OCR."""
        ...
