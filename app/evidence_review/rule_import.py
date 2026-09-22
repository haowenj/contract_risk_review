from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mineru_raw_parse import run_parse


@dataclass(frozen=True)
class ValidatedRuleUpload:
    filename: str
    suffix: str
    content: bytes
    text: str | None
    sha256: str


@dataclass(frozen=True)
class ExtractedRuleDocument:
    title_hint: str
    blocks: list[dict[str, object]]
    page_count: int


def validate_rule_upload(
    filename: str,
    content: bytes,
    *,
    max_bytes: int,
) -> ValidatedRuleUpload:
    safe_name = Path(filename or "").name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise ValueError("规则文件仅支持 .pdf、.txt 或 .md。")
    if not content:
        raise ValueError("规则文件不能为空。")
    if len(content) > max_bytes:
        raise ValueError("规则文件大小超过允许上限。")

    text: str | None = None
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("PDF 文件头无效。")
    else:
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("规则文本文件必须使用 UTF-8 编码。") from exc
        if not text:
            raise ValueError("规则文件不能为空。")

    return ValidatedRuleUpload(
        filename=safe_name,
        suffix=suffix,
        content=content,
        text=text,
        sha256=hashlib.sha256(content).hexdigest(),
    )


Parser = Callable[..., None]


class RuleDocumentExtractor:
    def __init__(
        self,
        *,
        svr_url: str,
        backend: str,
        server_url: str | None,
        parser: Parser = run_parse,
    ):
        self.svr_url = svr_url
        self.backend = backend
        self.server_url = server_url
        self.parser = parser

    def extract(
        self,
        upload: ValidatedRuleUpload,
        storage_dir: Path,
    ) -> ExtractedRuleDocument:
        storage_dir.mkdir(parents=True, exist_ok=True)
        source_path = storage_dir / f"source{upload.suffix}"
        source_path.write_bytes(upload.content)
        title_hint = Path(upload.filename).stem

        if upload.text is not None:
            return ExtractedRuleDocument(
                title_hint=title_hint,
                blocks=[{"page_number": 1, "text": upload.text}],
                page_count=1,
            )

        raw_path = storage_dir / "raw_content_list.json"
        self.parser(
            source_path,
            raw_path,
            svr_url=self.svr_url,
            backend=self.backend,
            server_url=self.server_url,
        )
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("MinerU 规则解析结果必须是数组。")

        blocks: list[dict[str, object]] = []
        for value in payload:
            if not isinstance(value, dict):
                continue
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                text = value.get("content")
            if not isinstance(text, str) or not text.strip():
                continue
            page_idx = value.get("page_idx")
            page_number = page_idx + 1 if isinstance(page_idx, int) else 1
            blocks.append({"page_number": page_number, "text": text.strip()})
        if not blocks:
            raise ValueError("PDF 规则文件没有可用文本。")
        return ExtractedRuleDocument(
            title_hint=title_hint,
            blocks=blocks,
            page_count=max(int(value["page_number"]) for value in blocks),
        )
