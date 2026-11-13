"""Separate, content-addressed local originals and immutable extraction versions."""

import json
import os
import tempfile
from pathlib import Path

from .contracts import DocumentError, ImmutableContract, ParsedDocument
from .docx import DocxReader, ReaderOptions


class StoredDocument(ImmutableContract):
    """Extraction and its separate original/versioned artifact locations."""

    document: ParsedDocument
    original_path: Path
    parsed_path: Path


class DocumentStore:
    """Ingest local DOCX without external APIs or replacement of stored content."""

    def __init__(self, root: Path, options: ReaderOptions | None = None) -> None:
        """Configure a storage root and extraction policy, without creating files."""
        self.root = Path(root)
        self.reader = DocxReader(options)

    def ingest(self, source: Path) -> StoredDocument:
        """Validate/extract before publishing separate original and JSON artifacts.

        Args:
            source: Local DOCX path; reads are bounded by the configured size limit.

        Returns:
            Original path and versioned JSON path, with the immutable document.

        Raises:
            DocumentError: Input, storage I/O, or existing-content conflict.
        """
        source = Path(source)
        try:
            with source.open("rb") as stream:
                data = stream.read(self.reader.options.max_file_bytes + 1)
        except OSError as error:
            raise DocumentError(
                "file_unreadable", "Не удалось прочитать исходный файл."
            ) from error
        document = self.reader.read(data, filename=source.name)
        original = self.root / "originals" / f"{document.document_id}.docx"
        parsed = (
            self.root
            / "parsed"
            / document.document_id
            / f"{document.text_version}.json"
        )
        payload = {
            "schema_version": 1,
            **document.model_dump(),
            "text_sha256": document.text_sha256,
        }
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        try:
            self._publish(original, data)
            self._publish(parsed, encoded)
        except OSError as error:
            raise DocumentError(
                "storage_error", "Не удалось сохранить документ."
            ) from error
        return StoredDocument(
            document=document, original_path=original, parsed_path=parsed
        )

    @staticmethod
    def _publish(path: Path, data: bytes) -> None:
        """Atomically publish a complete file without overwriting an existing one."""
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or path.read_bytes() != data
                    ):
                        raise DocumentError(
                            "storage_conflict",
                            "Сохранённый файл отличается; перезапись запрещена.",
                        ) from None
            finally:
                temporary.unlink(missing_ok=True)
