"""Document ingestion contracts exercised against real Word packages."""

import hashlib
import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from docx.oxml import OxmlElement
from pydantic import ValidationError


def test_contracts_have_json_schemas_and_validate_roundtrip(docx_bytes, tmp_path):
    """Expose typed JSON schemas and retain document consistency on validation."""
    from pydantic import BaseModel

    from vrscore.documents import DocxReader, ReaderOptions
    from vrscore.documents.contracts import (
        Evidence,
        ListInfo,
        ParsedDocument,
        SourceLocation,
        TextBlock,
    )
    from vrscore.documents.storage import StoredDocument

    for contract in (
        Evidence,
        ListInfo,
        ParsedDocument,
        SourceLocation,
        TextBlock,
        ReaderOptions,
        StoredDocument,
    ):
        assert issubclass(contract, BaseModel)
        assert contract.model_json_schema()["type"] == "object"

    document = DocxReader().read(
        docx_bytes(lambda doc: doc.add_paragraph("Python")), filename="resume.docx"
    )
    assert ParsedDocument.model_validate_json(document.model_dump_json()) == document
    payload = document.model_dump(mode="json")
    payload["blocks"][0]["end"] += 1
    with pytest.raises(ValidationError, match="offsets"):
        ParsedDocument.model_validate(payload)
    with pytest.raises(ValidationError):
        ReaderOptions(max_file_bytes=0)
    evidence = document.quote(document.blocks[0].block_id, 0, 6, "Python")
    assert Evidence.model_validate(evidence.model_dump()) == evidence
    with pytest.raises(ValidationError, match="unexpected"):
        Evidence.model_validate({**evidence.model_dump(), "unexpected": "field"})
    stored = StoredDocument(
        document=document,
        original_path=tmp_path / "original.docx",
        parsed_path=tmp_path / "parsed.json",
    )
    assert StoredDocument.model_validate_json(stored.model_dump_json()) == stored


def test_docx_reading_order_and_evidence(docx_bytes):
    """Keep table/list order, exact source text and validated evidence spans."""
    from vrscore.documents import DocxReader

    def populate(document):
        document.add_paragraph("  Cafe\u0301\u00a0 Python  ")
        document.add_paragraph("Vue.js", style="List Bullet")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Опыт"
        table.cell(0, 1).text = "3 года"
        table.cell(0, 1).add_table(rows=1, cols=1).cell(0, 0).text = "SQL"
        document.add_paragraph("Конец\tстроки\nNext")
        run = document.add_paragraph("non").add_run()
        run._r.append(OxmlElement("w:noBreakHyphen"))
        run.add_text("breaking")

    data = docx_bytes(populate)
    result = DocxReader().read(data, filename="resume.DOCX")
    assert [b.original_text for b in result.blocks if b.text] == [
        "  Cafe\u0301\u00a0 Python  ",
        "Vue.js",
        "Опыт",
        "3 года",
        "SQL",
        "Конец\tстроки\nNext",
        "non\u2011breaking",
    ]
    assert result.blocks[1].kind == "list_item"
    assert result.blocks[1].list_info.level == 0
    assert "w:tbl/w:tr/w:tc[2]/w:p[1]" in result.blocks[3].source.xml_path
    assert len({b.block_id for b in result.blocks}) == len(result.blocks)
    assert result == DocxReader().read(data, filename="other.docx")
    for block in result.blocks:
        assert result.text[block.start : block.end] == block.text
        assert block.document_id == result.document_id
        assert block.text_version == result.text_version
    assert result.blocks[0].text != result.blocks[0].original_text
    with pytest.raises(ValidationError):
        result.text = "changed"
    block = result.blocks[1]
    quote = result.quote(block.block_id, block.start, block.end, "Vue.js")
    assert result.resolve(quote) == "Vue.js"
    with pytest.raises(ValueError, match="literal"):
        result.quote(
            result.blocks[0].block_id, 0, result.blocks[0].end, result.blocks[0].text
        )
    for start, end, text in [
        (block.start, block.end, "Vue…"),
        (-1, block.end, "Vue.js"),
        (block.start, block.end + 1, "Vue.js\n"),
    ]:
        with pytest.raises(ValueError):
            result.quote(block.block_id, start, end, text)


def test_ingestion_failures_and_scope(docx_bytes):
    """Reject invalid input and expose unsupported content instead of guessing."""
    from vrscore.documents import DocumentError, DocxReader, ReaderOptions

    valid = docx_bytes(lambda doc: doc.add_paragraph("Python"))
    broken = BytesIO()
    with ZipFile(BytesIO(valid)) as source, ZipFile(broken, "w", ZIP_DEFLATED) as out:
        for item in source.infolist():
            out.writestr(
                item,
                b"<broken"
                if item.filename == "word/document.xml"
                else source.read(item),
            )
    cases = [
        (b"", "empty.docx", "empty_file"),
        (b"not a zip", "bad.docx", "corrupt_file"),
        (valid[:100], "truncated.docx", "corrupt_file"),
        (broken.getvalue(), "broken.docx", "corrupt_file"),
        (valid, "resume.pdf", "unsupported_format"),
        (docx_bytes(), "blank.docx", "empty_document"),
    ]
    for data, name, code in cases:
        with pytest.raises(DocumentError) as caught:
            DocxReader().read(data, filename=name)
        assert caught.value.code == code
        assert str(caught.value)
    with pytest.raises(DocumentError) as caught:
        DocxReader(ReaderOptions(max_file_bytes=10)).read(valid, filename="big.docx")
    assert caught.value.code == "file_too_large"
    with pytest.raises(DocumentError) as caught:
        DocxReader(ReaderOptions(max_uncompressed_bytes=10)).read(
            valid, filename="big.docx"
        )
    assert caught.value.code == "archive_too_large"

    def unsupported(document):
        document.add_paragraph("Python")
        document.element.body.insert(0, OxmlElement("w:sdt"))

    with pytest.raises(DocumentError) as caught:
        DocxReader().read(docx_bytes(unsupported), filename="content-control.docx")
    assert caught.value.code == "unsupported_content"

    def illustrated(document):
        document.add_paragraph("Python").add_run()._r.append(OxmlElement("w:drawing"))
        document.sections[0].header.paragraphs[0].text = "Header"

    result = DocxReader().read(docx_bytes(illustrated), filename="image.docx")
    assert "images_not_extracted" in result.warnings
    assert "other_stories_not_extracted" in result.warnings
    assert result.text == "Python"
    assert DocxReader.supported_suffixes == (".docx",)


def test_dataset_contract_bridge(docx_bytes):
    """Preserve IDs/text/offsets; overlapping sections are not physical blocks."""
    from vrscore.documents import DocxReader, from_dataset_record, to_dataset_record
    from vrscore.vacancy_parser.parser import parse_record, requirement_items

    text = (
        "Требования:\nОпыт Python от 3 лет.\nPython; SQL\n3.0\n"
        + "Описание работы. " * 8
    )
    parsed = DocxReader().read(
        docx_bytes(lambda doc: doc.add_paragraph(text)), filename="vacancy.docx"
    )
    metadata = {
        "id": "vacancy_001",
        "kind": "vacancy",
        "sampling_group": "it",
        "title": "Разработчик",
        "source": {
            "publisher": "local",
            "acquisition": "local_file",
            "original_url": "",
            "source_id": "original.docx",
            "source_id_kind": "local_filename",
        },
    }
    row = to_dataset_record(parsed, **metadata)
    assert row["text"] == parsed.text
    assert row["text_sha256"] == parsed.text_sha256
    assert row["source"]["document"]["text_version"] == parsed.text_version
    assert "document" not in metadata["source"]
    assert set(row) == {
        "id",
        "kind",
        "sampling_group",
        "title",
        "text",
        "attributes",
        "sections",
        "source",
        "text_sha256",
    }
    structured, _ = parse_record(row)
    assert requirement_items(structured)[0]["text"] == "Опыт Python от 3 лет."
    structured["attributes"].update(hardSkills=["Python", "SQL"], experience=3.0)
    for field, rendered in (("hardSkills", "Python; SQL"), ("experience", "3.0")):
        start = text.index(rendered)
        structured["sections"].append(
            {
                "source_field": field,
                "label": field,
                "start": start,
                "end": start + len(rendered),
            }
        )
    view = from_dataset_record(structured)
    assert view.document_id == row["id"]
    assert view.text == structured["text"]
    assert view.text_sha256 == structured["text_sha256"]
    # Nested sections share offsets, not independent document blocks.
    assert len(view.blocks) == 1
    block = view.blocks[0]
    assert view.quote(block.block_id, 0, 10, text[:10]).text == text[:10]
    with pytest.raises(ValueError, match="version"):
        view.resolve(
            view.quote(block.block_id, 0, 10, text[:10]).model_copy(
                update={"text_version": "stale"}
            )
        )
    with pytest.raises(ValueError):
        from_dataset_record({**structured, "text_sha256": "0" * 64})
    assert hashlib.sha256(text.encode()).hexdigest() == view.text_sha256


def test_immutable_storage_and_versions(docx_bytes, tmp_path):
    """Keep original bytes separately, deduplicate versions and never overwrite."""
    from vrscore.documents import DocumentError, DocumentStore, ReaderOptions

    data = docx_bytes(lambda doc: doc.add_paragraph("  Cafe\u0301\tPython  "))
    source = tmp_path / "resume.docx"
    source.write_bytes(data)
    store = DocumentStore(tmp_path / "store")
    first = store.ingest(source)
    assert first.original_path.read_bytes() == data
    payload = json.loads(first.parsed_path.read_text())
    assert payload["text"] == first.document.text
    assert payload["text_sha256"] == first.document.text_sha256
    assert payload["blocks"][0]["original_text"] == "  Cafe\u0301\tPython  "
    assert store.ingest(source) == first
    collapsed = DocumentStore(store.root, ReaderOptions(whitespace="collapse")).ingest(
        source
    )
    assert collapsed.original_path == first.original_path
    assert collapsed.parsed_path != first.parsed_path
    assert collapsed.document.text == "Café Python"
    assert (
        first.parsed_path.read_text()
        == json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    first.parsed_path.write_text("corrupted artifact")
    with pytest.raises(DocumentError) as caught:
        store.ingest(source)
    assert caught.value.code == "storage_conflict"
    assert first.parsed_path.read_text() == "corrupted artifact"
    with pytest.raises(DocumentError) as caught:
        store.ingest(tmp_path / "missing.docx")
    assert caught.value.code == "file_unreadable"
    invalid_store = DocumentStore(tmp_path / "invalid-store")
    source.write_bytes(b"broken")
    with pytest.raises(DocumentError):
        invalid_store.ingest(source)
    assert not invalid_store.root.exists()
