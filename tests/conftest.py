"""Shared in-memory document fixtures."""

from io import BytesIO

import pytest
from docx import Document


@pytest.fixture
def docx_bytes():
    """Build a real DOCX, optionally customising its content."""

    def build(populate=None):
        document = Document()
        if populate is not None:
            populate(document)
        stream = BytesIO()
        document.save(stream)
        return stream.getvalue()

    return build
