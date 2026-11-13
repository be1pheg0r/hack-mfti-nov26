"""Local DOCX body extraction with whole-block provenance, without OCR.

Tables are traversed row/cell/XML order, nested tables included; merged cells
are visited once per physical XML cell. Lists retain identity and level, not
Word's rendered labels. Headers, footers and non-body stories are outside this
reader's scope. Word pagination and visual multi-column order are not inferred.
"""

import hashlib
import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from pydantic import Field

from .contracts import (
    DocumentError,
    ImmutableContract,
    ListInfo,
    ParsedDocument,
    SourceLocation,
    TextBlock,
)

PARSER_VERSION = "docx-body-v1"


class ReaderOptions(ImmutableContract):
    """Version-affecting normalisation policy; no truncation is permitted."""

    unicode_form: Literal["none", "NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    whitespace: Literal["preserve", "collapse"] = "preserve"
    max_file_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_uncompressed_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_archive_members: int = Field(default=2000, gt=0)

    @property
    def version(self) -> str:
        """Return an explicit normalisation algorithm version."""
        return f"unicode-{self.unicode_form}:whitespace-{self.whitespace}:v1"

    def normalize(self, text: str) -> str:
        """Normalise without pretending resulting offsets address the original."""
        if self.unicode_form != "none":
            text = unicodedata.normalize(self.unicode_form, text)
        return (
            re.sub(r"\s+", " ", text).strip() if self.whitespace == "collapse" else text
        )


class DocxReader:
    """Read DOCX main-story paragraphs in logical XML reading order."""

    supported_suffixes = (".docx",)

    def __init__(self, options: ReaderOptions | None = None) -> None:
        """Use an explicit immutable normalisation policy."""
        self.options = options or ReaderOptions()

    def read(self, data: bytes, *, filename: str) -> ParsedDocument:
        """Extract supported bytes or raise an actionable DocumentError.

        Args:
            data: Original DOCX bytes, retained unchanged by the caller/store.
            filename: Used for format declaration only, not document identity.

        Raises:
            DocumentError: Empty, corrupt, oversized or unsupported content.
        """
        if Path(filename).suffix.lower() not in self.supported_suffixes:
            raise DocumentError(
                "unsupported_format",
                "Поддерживается только DOCX; PDF/OCR не реализованы.",
            )
        if not data:
            raise DocumentError("empty_file", "Файл пуст.")
        if len(data) > self.options.max_file_bytes:
            raise DocumentError(
                "file_too_large", "Размер файла превышает настроенный лимит."
            )
        try:
            warnings = self._validate_archive(data)
            document = Document(BytesIO(data))
            if document.element.body is None:
                raise ValueError("Missing document body")
            self._validate_body(document.element.body)
            if (
                next(document.element.body.iter(qn("w:drawing"), qn("w:pict")), None)
                is not None
            ):
                warnings = tuple(sorted(set(warnings) | {"images_not_extracted"}))
            return self._extract(document, data, warnings)
        except DocumentError:
            raise
        except (
            BadZipFile,
            XMLSyntaxError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            OSError,
            RuntimeError,
            EOFError,
            NotImplementedError,
            ZlibError,
        ) as error:
            raise DocumentError(
                "corrupt_file", "DOCX повреждён или не является документом Word."
            ) from error

    def _extract(
        self, document, data: bytes, warnings: tuple[str, ...]
    ) -> ParsedDocument:
        """Build immutable blocks and a content-addressed extraction version."""
        document_id = hashlib.sha256(data).hexdigest()
        drafts = []
        cursor = 0
        for element in self._paragraphs(document.element.body):
            paragraph = Paragraph(element, document)
            original = self._paragraph_text(element)
            text = self.options.normalize(original)
            info = self._list_info(paragraph)
            drafts.append(
                {
                    "block_id": f"b{len(drafts):06d}",
                    "kind": "list_item" if info is not None else "paragraph",
                    "text": text,
                    "original_text": original,
                    "start": cursor,
                    "end": cursor + len(text),
                    "source": SourceLocation(
                        part="word/document.xml",
                        xml_path=element.getroottree().getpath(element),
                    ),
                    "list_info": info,
                }
            )
            cursor += len(text) + 1
        version_payload = json.dumps(
            {
                "parser": PARSER_VERSION,
                "normalization": self.options.version,
                "blocks": [
                    {
                        **item,
                        "source": item["source"].model_dump(),
                        "list_info": item["list_info"].model_dump()
                        if item["list_info"]
                        else None,
                    }
                    for item in drafts
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        text_version = hashlib.sha256(version_payload.encode("utf-8")).hexdigest()
        blocks = tuple(
            TextBlock(document_id=document_id, text_version=text_version, **item)
            for item in drafts
        )
        if not any(block.text.strip() for block in blocks):
            if "images_not_extracted" in warnings:
                raise DocumentError(
                    "ocr_required",
                    "Нет текста, но есть изображения. OCR не реализован.",
                )
            raise DocumentError(
                "empty_document", "В основном содержимом DOCX нет текста."
            )
        return ParsedDocument(
            document_id=document_id,
            text_version=text_version,
            parser_version=PARSER_VERSION,
            normalization=self.options.version,
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
            warnings=warnings,
        )

    def _validate_archive(self, data: bytes) -> tuple[str, ...]:
        """Bound decompression, verify CRCs and disclose non-body stories."""
        warnings = set()
        with ZipFile(BytesIO(data)) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if len(members) != len(names):
                raise ValueError("Ambiguous duplicate ZIP entries")
            if (
                len(members) > self.options.max_archive_members
                or sum(member.file_size for member in members)
                > self.options.max_uncompressed_bytes
            ):
                raise DocumentError(
                    "archive_too_large",
                    "Распакованный DOCX превышает настроенный лимит.",
                )
            if not {"[Content_Types].xml", "_rels/.rels", "word/document.xml"} <= names:
                raise ValueError("Required DOCX parts are missing")
            for member in members:
                content = archive.read(member)
                if member.filename.endswith((".xml", ".rels")):
                    # DTD/entity declarations are not needed in OPC XML.
                    xml = content.replace(b"\x00", b"")
                    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
                        raise ValueError("Unsafe XML declarations")
                if member.filename.startswith("word/media/"):
                    warnings.add("images_not_extracted")
                if re.fullmatch(
                    r"word/(?:header\d+|footer\d+|footnotes|endnotes|comments)\.xml",
                    member.filename,
                ):
                    warnings.add("other_stories_not_extracted")
        return tuple(sorted(warnings))

    @staticmethod
    def _validate_body(body) -> None:
        """Fail explicitly for constructs whose visible text would be dropped."""
        unsupported = {
            qn(f"w:{tag}")
            for tag in (
                "sdt",
                "ins",
                "del",
                "moveFrom",
                "moveTo",
                "altChunk",
                "txbxContent",
                "fldSimple",
                "fldChar",
                "object",
                "subDoc",
                "contentPart",
                "sym",
                "customXml",
                "smartTag",
                "ruby",
            )
        }
        for element in body.iter():
            if element.tag in unsupported or element.tag.endswith("}AlternateContent"):
                raise DocumentError(
                    "unsupported_content",
                    "DOCX содержит неподдерживаемые поля, правки "
                    "или текстовые контейнеры.",
                )
            if element.tag == qn("w:p"):
                parent = element.getparent()
                while parent is not body:
                    if parent is None or parent.tag not in {
                        qn("w:tbl"),
                        qn("w:tr"),
                        qn("w:tc"),
                    }:
                        raise DocumentError(
                            "unsupported_content",
                            "Неподдерживаемый порядок текстовых контейнеров DOCX.",
                        )
                    parent = parent.getparent()

    @staticmethod
    def _paragraph_text(element) -> str:
        """Extract logical characters, including Word's non-breaking/soft hyphens."""
        tokens = {
            qn("w:tab"): "\t",
            qn("w:cr"): "\n",
            qn("w:noBreakHyphen"): "\u2011",
            qn("w:softHyphen"): "\u00ad",
        }
        text = []
        for child in element.iter():
            if child.tag == qn("w:t"):
                text.append(child.text or "")
            elif child.tag in tokens:
                text.append(tokens[child.tag])
            elif (
                child.tag == qn("w:br")
                and child.get(qn("w:type"), "textWrapping") == "textWrapping"
            ):
                text.append("\n")
        return "".join(text)

    @classmethod
    def _paragraphs(cls, parent):
        """Walk physical cells once, retaining interleaved nested table order."""
        for child in parent:
            if child.tag == qn("w:p"):
                yield child
            elif child.tag in {qn("w:tbl"), qn("w:tr"), qn("w:tc")}:
                yield from cls._paragraphs(child)

    @staticmethod
    def _list_info(paragraph: Paragraph) -> ListInfo | None:
        """Resolve direct or inherited numbering, including explicit cancellation."""
        properties = [paragraph._p.pPr]
        style = paragraph.style
        seen = set()
        while style is not None and style.style_id not in seen:
            seen.add(style.style_id)
            properties.append(style.element.pPr)
            style = getattr(style, "base_style", None)
        numbering_id = level = None
        for prop in properties:
            numbering = prop.numPr if prop is not None else None
            if numbering is None:
                continue
            if numbering_id is None and numbering.numId is not None:
                numbering_id = numbering.numId.val
            if level is None and numbering.ilvl is not None:
                level = numbering.ilvl.val
        if numbering_id in (None, 0):
            return None
        return ListInfo(numbering_id=numbering_id, level=level or 0)
