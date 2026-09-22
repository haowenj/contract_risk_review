from __future__ import annotations

import json
from pathlib import Path

import pytest
import httpx

from app.evidence_review.rule_import import (
    RuleDocumentExtractor,
    run_rule_parse,
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


def test_rule_parser_falls_back_to_mineru_v1_structured_content(tmp_path: Path):
    calls = []
    poll_count = 0

    def legacy_parser(*args, **kwargs):
        raise RuntimeError('提交 MinerU 任务失败：HTTP 404 {"detail":"Not Found"}')

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/uploads":
            return httpx.Response(
                200,
                json={
                    "id": "upload-1",
                    "upload_url": "http://mineru.test/v1/uploads/upload-1/content",
                    "upload_headers": {"content-type": "application/octet-stream"},
                },
            )
        if request.method == "PUT" and request.url.path.endswith("/content"):
            return httpx.Response(200, json={})
        if request.method == "POST" and request.url.path.endswith("/complete"):
            return httpx.Response(200, json={"file": {"id": "file-input"}})
        if request.method == "POST" and request.url.path == "/v1/parse/jobs":
            body = json.loads(request.content)
            assert body["output_formats"] == ["structured_content"]
            assert body["tier"] == "standard"
            return httpx.Response(202, json={"job_id": "job-1"})
        if request.method == "GET" and request.url.path == "/v1/parse/jobs/job-1":
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(200, json={"status": "running"})
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "files": [
                        {
                            "status": "completed",
                            "output_files": {
                                "structured_content": {
                                    "file_id": "file-output",
                                    "bytes": 10,
                                }
                            },
                        }
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/v1/files/file-output/content":
            return httpx.Response(
                200,
                json={
                    "pages": [
                        {
                            "page_idx": 0,
                            "blocks": [
                                {"type": "paragraph_title", "content": "一、阶段"},
                                {"type": "text", "content": "1. 检查项"},
                            ],
                        },
                        {
                            "page_idx": 1,
                            "blocks": [
                                {"type": "text", "content": "2. 第二项"}
                            ],
                        },
                    ]
                },
            )
        if request.method == "DELETE" and request.url.path.startswith("/v1/files/"):
            return httpx.Response(200, json={})
        return httpx.Response(404)

    source = tmp_path / "rules.pdf"
    output = tmp_path / "raw.json"
    source.write_bytes(b"%PDF-test")
    client = httpx.Client(transport=httpx.MockTransport(handler))

    run_rule_parse(
        source,
        output,
        svr_url="http://mineru.test",
        backend="hybrid-engine",
        server_url=None,
        legacy_parser=legacy_parser,
        client=client,
        poll_interval=0,
    )
    client.close()

    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"page_idx": 0, "type": "text", "text": "一、阶段"},
        {"page_idx": 0, "type": "text", "text": "1. 检查项"},
        {"page_idx": 1, "type": "text", "text": "2. 第二项"},
    ]
    assert ("POST", "/v1/uploads") in calls
