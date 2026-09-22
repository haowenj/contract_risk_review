from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_openai import ChatOpenAI
from mineru_raw_parse import run_parse

from app.evidence_review.repository import (
    EvidenceReviewRepository,
    RuleSetRecord,
)
from app.evidence_review.schemas import RuleParseResult


LOGGER = logging.getLogger(__name__)
RULE_PARSE_TIMEOUT_SECONDS = 120.0
SAFE_PARSE_ERROR = "规则文件解析失败，请检查文件内容后重试。"


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


def build_rule_parse_llm() -> Any:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "contract_risk_rule_parse",
            "strict": True,
            "schema": RuleParseResult.model_json_schema(),
        },
    }
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0,
        timeout=RULE_PARSE_TIMEOUT_SECONDS,
        max_retries=0,
        reasoning_effort="none",
    ).bind(response_format=response_format)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)
    raise ValueError("规则解析模型没有返回 JSON 文本。")


class RuleSetImportService:
    def __init__(
        self,
        *,
        repository: EvidenceReviewRepository,
        extractor: RuleDocumentExtractor,
        parse_llm: Any,
        rule_sets_dir: Path,
        max_upload_bytes: int,
    ):
        self.repository = repository
        self.extractor = extractor
        self.parse_llm = parse_llm
        self.rule_sets_dir = Path(rule_sets_dir)
        self.max_upload_bytes = max_upload_bytes

    def import_rule_set(self, rule_set_id: str) -> RuleSetRecord:
        record = self.repository.get_rule_set(rule_set_id)
        if record is None:
            raise KeyError(rule_set_id)
        self.repository.mark_rule_set_processing(rule_set_id)

        try:
            content = Path(record.source_path).read_bytes()
            upload = validate_rule_upload(
                record.source_filename,
                content,
                max_bytes=self.max_upload_bytes,
            )
            document = self.extractor.extract(
                upload,
                self.rule_sets_dir / rule_set_id,
            )

            # Import locally to avoid a module cycle: prompts needs the extracted
            # document type defined in this module.
            from app.evidence_review.prompts import build_rule_parse_prompt

            response = self.parse_llm.invoke(build_rule_parse_prompt(document))
            parsed = RuleParseResult.model_validate_json(_response_text(response))
            parsed_json = parsed.model_dump(mode="json")
            parsed_json["schema_version"] = "1.0"
            parsed_json["source"] = {
                "filename": upload.filename,
                "sha256": upload.sha256,
                "page_count": document.page_count,
            }
            parsed_json["summary"] = self._build_summary(parsed)
            return self.repository.mark_rule_set_draft(
                rule_set_id,
                parsed_json,
            )
        except Exception:
            LOGGER.exception("Rule-set import failed for %s", rule_set_id)
            return self.repository.mark_rule_set_failed(
                rule_set_id,
                SAFE_PARSE_ERROR,
            )

    @staticmethod
    def _build_summary(parsed: RuleParseResult) -> dict[str, int]:
        process_count = sum(
            item.item_kind == "process_control"
            for item in parsed.review_items
        )
        return {
            "section_count": len(parsed.sections),
            "review_item_count": len(parsed.review_items),
            "review_check_count": len(parsed.review_items) - process_count,
            "process_control_count": process_count,
        }
