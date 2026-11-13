"""Local document ingestion and immutable evidence contracts."""

from .contracts import (
    DocumentError,
    DocumentReader,
    Evidence,
    ListInfo,
    ParsedDocument,
    SourceLocation,
    TextBlock,
)
from .dataset import from_dataset_record, to_dataset_record
from .docx import DocxReader, ReaderOptions
from .storage import DocumentStore, StoredDocument

__all__ = [
    "DocumentError",
    "DocumentReader",
    "DocxReader",
    "Evidence",
    "ListInfo",
    "ParsedDocument",
    "ReaderOptions",
    "SourceLocation",
    "TextBlock",
    "from_dataset_record",
    "to_dataset_record",
    "DocumentStore",
    "StoredDocument",
]
