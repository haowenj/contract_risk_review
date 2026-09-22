from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evidence_review.rule_import import (
    RuleDocumentExtractor,
    validate_rule_upload,
)


def test_pdf_extension_requires_pdf_magic():
    with pytest.raises(ValueError, match="PDF 文件头"):
        validate_rule_upload("rules.pdf", b"not-a-pdf", max_bytes=1024)


def test_empty_oversized_and_unsupported_uploads_are_rejected():
    with pytest.raises(ValueError, match="不能为空"):
        validate_rule_upload("rules.md", b"", max_bytes=10)
    with pytest.raises(ValueError, match="超过"):
        validate_rule_upload("rules.md", b"x" * 11, max_bytes=10)
    with pytest.raises(ValueError, match="仅支持"):
        validate_rule_upload("rules.docx", b"x", max_bytes=10)


def test_utf8_bom_text_is_accepted_and_filename_is_sanitized():
    upload = validate_rule_upload(
        "../../rules.txt",
        "\ufeff付款周期过长".encode(),
        max_bytes=1024,
    )

    assert upload.filename == "rules.txt"
    assert upload.text == "付款周期过长"
    assert len(upload.sha256) == 64


def test_text_document_is_stored_in_managed_directory(tmp_path: Path):
    upload = validate_rule_upload(
        "rules.md", "# 审核规则\n付款周期过长".encode(), max_bytes=1024,
    )
    extractor = RuleDocumentExtractor(
        svr_url="http://mineru", backend="hybrid-engine", server_url=None,
    )

    document = extractor.extract(upload, tmp_path / "managed")

    assert document.page_count == 1
    assert document.blocks == [
        {"page_number": 1, "text": "# 审核规则\n付款周期过长"}
    ]
    assert (tmp_path / "managed" / "source.md").is_file()


def test_pdf_uses_mineru_and_maps_pages_to_one_based(tmp_path: Path):
    calls = []

    def fake_run_parse(source_path, output_path, **kwargs):
        calls.append((source_path, kwargs))
        output_path.write_text(
            json.dumps([
                {"page_idx": 0, "type": "text", "text": "第一项"},
                {"page_idx": 1, "type": "text", "content": "第二项"},
                {"page_idx": 1, "type": "image"},
            ]),
            encoding="utf-8",
        )

    upload = validate_rule_upload("rules.pdf", b"%PDF-test", max_bytes=1024)
    extractor = RuleDocumentExtractor(
        svr_url="http://mineru", backend="hybrid-engine", server_url=None,
        parser=fake_run_parse,
    )

    document = extractor.extract(upload, tmp_path / "managed")

    assert calls[0][0] == tmp_path / "managed" / "source.pdf"
    assert document.page_count == 2
    assert document.blocks == [
        {"page_number": 1, "text": "第一项"},
        {"page_number": 2, "text": "第二项"},
    ]


def test_pdf_parser_failure_is_propagated(tmp_path: Path):
    def failing_parser(source_path, output_path, **kwargs):
        raise RuntimeError("mineru unavailable")

    upload = validate_rule_upload("rules.pdf", b"%PDF-test", max_bytes=1024)
    extractor = RuleDocumentExtractor(
        svr_url="http://mineru", backend="hybrid-engine", server_url=None,
        parser=failing_parser,
    )

    with pytest.raises(RuntimeError, match="mineru unavailable"):
        extractor.extract(upload, tmp_path / "managed")
