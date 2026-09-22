from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from langchain_openai import ChatOpenAI
from mineru_raw_parse import run_parse

from app.evidence_review.repository import (
    EvidenceReviewRepository,
    RuleSetRecord,
)
from app.evidence_review.schemas import RuleParseResult


LOGGER = logging.getLogger(__name__)
RULE_PARSE_TIMEOUT_SECONDS = float(
    os.getenv("RULE_PARSE_TIMEOUT_SECONDS", "300")
)
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


def _v1_response_json(response: httpx.Response, label: str) -> dict[str, Any]:
    if response.status_code not in {200, 202}:
        raise RuntimeError(
            f"MinerU v1 {label}失败：HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"MinerU v1 {label}响应不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"MinerU v1 {label}响应必须是 JSON 对象")
    return payload


def _run_v1_rule_parse(
    source_path: Path,
    output_path: Path,
    *,
    svr_url: str,
    backend: str,
    client: httpx.Client | None = None,
    poll_interval: float = 2.0,
) -> None:
    """Parse a rule PDF through MinerU 4.x and flatten page text blocks."""
    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        follow_redirects=True,
    )
    base_url = svr_url.rstrip("/")
    input_file_id: str | None = None
    output_file_id: str | None = None
    remove_input_file = False
    try:
        content = source_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        upload = _v1_response_json(
            http_client.post(
                f"{base_url}/v1/uploads",
                json={
                    "filename": source_path.name,
                    "bytes": len(content),
                    "mime_type": "application/pdf",
                    "purpose": "parse",
                    "sha256sum": digest,
                },
            ),
            "创建上传",
        )
        upload_id = upload.get("id")
        if not isinstance(upload_id, str) or not upload_id:
            raise RuntimeError("MinerU v1 返回了无效 upload id")
        file_payload = upload.get("file")
        if upload.get("status") != "completed" or not isinstance(
            file_payload, dict
        ):
            upload_url = upload.get("upload_url")
            if not isinstance(upload_url, str) or not upload_url:
                upload_url = f"{base_url}/v1/uploads/{upload_id}/content"
            elif upload_url.startswith("/"):
                upload_url = f"{base_url}{upload_url}"
            headers = upload.get("upload_headers")
            if not isinstance(headers, dict):
                headers = {"content-type": "application/octet-stream"}
            upload_response = http_client.put(
                upload_url,
                headers={
                    str(key): str(value) for key, value in headers.items()
                },
                content=content,
            )
            if upload_response.status_code != 200:
                raise RuntimeError(
                    "MinerU v1 上传内容失败："
                    f"HTTP {upload_response.status_code}"
                )
            completed = _v1_response_json(
                http_client.post(
                    f"{base_url}/v1/uploads/{upload_id}/complete",
                    json={"sha256sum": digest},
                ),
                "完成上传",
            )
            file_payload = completed.get("file")
            remove_input_file = True
        input_file_id = (
            file_payload.get("id")
            if isinstance(file_payload, dict)
            else None
        )
        if not isinstance(input_file_id, str) or not input_file_id:
            raise RuntimeError("MinerU v1 返回了无效 input file id")

        tier = "basic" if backend == "pipeline" else "standard"
        job = _v1_response_json(
            http_client.post(
                f"{base_url}/v1/parse/jobs",
                json={
                    "files": [
                        {
                            "source": {
                                "type": "file_id",
                                "file_id": input_file_id,
                            }
                        }
                    ],
                    "tier": tier,
                    "ocr_mode": "auto",
                    "output_formats": ["structured_content"],
                },
            ),
            "创建解析任务",
        )
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("MinerU v1 返回了无效 job id")

        deadline = time.monotonic() + 30 * 60
        while time.monotonic() < deadline:
            job = _v1_response_json(
                http_client.get(f"{base_url}/v1/parse/jobs/{job_id}"),
                "查询解析任务",
            )
            status = job.get("status")
            if status in {"queued", "running"}:
                time.sleep(poll_interval)
                continue
            if status in {"completed", "partial"}:
                break
            raise RuntimeError(f"MinerU v1 解析任务未完成：{status}")
        else:
            raise RuntimeError("等待 MinerU v1 解析任务超时")

        files = job.get("files")
        first_file = files[0] if isinstance(files, list) and files else None
        output_files = (
            first_file.get("output_files")
            if isinstance(first_file, dict)
            else None
        )
        structured_ref = (
            output_files.get("structured_content")
            if isinstance(output_files, dict)
            else None
        )
        output_file_id = (
            structured_ref.get("file_id")
            if isinstance(structured_ref, dict)
            else None
        )
        if not isinstance(output_file_id, str) or not output_file_id:
            raise RuntimeError("MinerU v1 结果缺少 structured_content")
        structured = _v1_response_json(
            http_client.get(
                f"{base_url}/v1/files/{output_file_id}/content"
            ),
            "下载结构化结果",
        )
        pages = structured.get("pages")
        if not isinstance(pages, list):
            raise RuntimeError("MinerU v1 structured_content 缺少 pages")
        flattened: list[dict[str, Any]] = []
        for fallback_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_idx = page.get("page_idx")
            if not isinstance(page_idx, int):
                page_idx = fallback_idx
            blocks = page.get("blocks")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_content = block.get("content")
                if not isinstance(block_content, str) or not block_content.strip():
                    continue
                flattened.append(
                    {
                        "page_idx": page_idx,
                        "type": "text",
                        "text": block_content.strip(),
                    }
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(flattened, ensure_ascii=False),
            encoding="utf-8",
        )
    finally:
        cleanup_file_ids = [output_file_id]
        if remove_input_file:
            cleanup_file_ids.append(input_file_id)
        for file_id in cleanup_file_ids:
            if file_id:
                try:
                    http_client.delete(f"{base_url}/v1/files/{file_id}")
                except httpx.HTTPError:
                    LOGGER.warning("Could not remove temporary MinerU file")
        if owns_client:
            http_client.close()


def run_rule_parse(
    source_path: Path,
    output_path: Path,
    *,
    svr_url: str,
    backend: str,
    server_url: str | None,
    legacy_parser: Parser = run_parse,
    client: httpx.Client | None = None,
    poll_interval: float = 2.0,
) -> None:
    try:
        legacy_parser(
            source_path,
            output_path,
            svr_url=svr_url,
            backend=backend,
            server_url=server_url,
            **(
                {"client": client, "poll_interval": poll_interval}
                if client is not None
                else {}
            ),
        )
        return
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        LOGGER.info("Legacy MinerU API unavailable; falling back to v1")
    _run_v1_rule_parse(
        source_path,
        output_path,
        svr_url=svr_url,
        backend=backend,
        client=client,
        poll_interval=poll_interval,
    )


class RuleDocumentExtractor:
    def __init__(
        self,
        *,
        svr_url: str,
        backend: str,
        server_url: str | None,
        parser: Parser = run_rule_parse,
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


def _number_component(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    component = value.strip().strip("、，,。.．:：;；-—")
    return component or None


def _normalize_rule_parse_payload(payload: object) -> object:
    """Repair known schema-weak response shapes before strict validation.

    Some compatible chat endpoints ignore JSON Schema nesting or return leaf
    requirements as string shorthand. This adapter only expands those values
    and derives structural metadata from the same response. All rule content
    is still checked by RuleParseResult.
    """
    if not isinstance(payload, dict):
        return payload
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list):
        return payload

    sections: list[dict[str, Any]] = []
    nested_items_by_section: dict[str, list[dict[str, Any]]] = {}
    used_compat_shape = False
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            sections.append(raw_section)
            continue
        section = dict(raw_section)
        nested_items = section.pop("review_items", None)
        if isinstance(nested_items, list):
            used_compat_shape = True
            section_id = section.get("section_id")
            if isinstance(section_id, str):
                nested_items_by_section[section_id] = [
                    dict(item) for item in nested_items if isinstance(item, dict)
                ]
        if "title" not in section and isinstance(
            section.get("section_title"), str
        ):
            used_compat_shape = True
            section["title"] = section.pop("section_title")
        sections.append(section)

    top_level_items = payload.get("review_items")
    top_level_uses_shorthand = isinstance(top_level_items, list) and any(
        isinstance(item, dict)
        and (
            "risk_type" in item
            or "section_path" not in item
            or "name" not in item
            or "decision_mode" not in item
            or any(
                isinstance(value, str)
                for value in item.get("fact_requirements", [])
            )
            or any(
                isinstance(value, str)
                for value in item.get("research_requirements", [])
            )
        )
        for item in top_level_items
    )
    if not used_compat_shape and not top_level_uses_shorthand:
        return payload

    by_id = {
        section.get("section_id"): section
        for section in sections
        if isinstance(section, dict)
        and isinstance(section.get("section_id"), str)
    }

    def section_level(section: dict[str, Any], seen: set[str]) -> int:
        existing = section.get("level")
        if isinstance(existing, int):
            return existing
        section_id = section.get("section_id")
        if not isinstance(section_id, str) or section_id in seen:
            return 1
        parent_id = section.get("parent_section_id")
        parent = by_id.get(parent_id)
        if not isinstance(parent, dict):
            return 1
        return section_level(parent, seen | {section_id}) + 1

    def number_path(section: dict[str, Any], seen: set[str]) -> list[str]:
        section_id = section.get("section_id")
        if not isinstance(section_id, str) or section_id in seen:
            return []
        parent = by_id.get(section.get("parent_section_id"))
        prefix = (
            number_path(parent, seen | {section_id})
            if isinstance(parent, dict)
            else []
        )
        component = _number_component(section.get("source_number"))
        return prefix + ([component] if component else [])

    def title_path(section: dict[str, Any], seen: set[str]) -> list[str]:
        section_id = section.get("section_id")
        if not isinstance(section_id, str) or section_id in seen:
            return []
        parent = by_id.get(section.get("parent_section_id"))
        prefix = (
            title_path(parent, seen | {section_id})
            if isinstance(parent, dict)
            else []
        )
        title = section.get("title")
        return prefix + ([title] if isinstance(title, str) and title else [])

    def normalize_compat_item(
        item: dict[str, Any], section: dict[str, Any] | None
    ) -> dict[str, Any]:
        normalized_item = dict(item)
        if "section_path" not in normalized_item:
            normalized_item["section_path"] = (
                title_path(section, set())
                if isinstance(section, dict)
                else []
            )
        rule_text = normalized_item.get("rule_text")
        if "name" not in normalized_item and isinstance(rule_text, str):
            normalized_item["name"] = rule_text.strip().splitlines()[0][:80]

        risk_type = normalized_item.pop("risk_type", None)
        if "item_kind" not in normalized_item and isinstance(risk_type, str):
            normalized_item["item_kind"] = risk_type

        raw_facts = normalized_item.get("fact_requirements")
        if isinstance(raw_facts, list) and all(
            isinstance(value, str) for value in raw_facts
        ):
            normalized_item["fact_requirements"] = [
                {
                    "fact_key": (
                        f"fact_{index}_"
                        f"{re.sub(r'\W+', '_', label).strip('_')[:40]}"
                    ).rstrip("_"),
                    "label": label,
                    "value_type": "string",
                    "required": True,
                }
                for index, label in enumerate(raw_facts, start=1)
                if label.strip()
            ]

        fact_keys = [
            value.get("fact_key")
            for value in normalized_item.get("fact_requirements", [])
            if isinstance(value, dict)
            and isinstance(value.get("fact_key"), str)
        ]
        raw_research = normalized_item.get("research_requirements")
        if isinstance(raw_research, list) and all(
            isinstance(value, str) for value in raw_research
        ):
            scope = normalized_item.get("evidence_scope")
            source_type = {
                "internal_material": "internal_material",
                "external_query": "public_query",
                "hybrid": "public_query",
            }.get(scope, "expert_opinion")
            target_type = {
                "internal_material": "内部材料",
                "external_query": "外部查询对象",
                "hybrid": "外部查询对象",
            }.get(scope, "专业复核对象")
            normalized_item["research_requirements"] = [
                {
                    "source_type": source_type,
                    "target_type": target_type,
                    "required_fact_keys": fact_keys,
                    "query_topics": [requirement],
                    "comparison_points": [
                        rule_text
                        if isinstance(rule_text, str) and rule_text.strip()
                        else requirement
                    ],
                }
                for requirement in raw_research
                if requirement.strip()
            ]

        if "decision_mode" not in normalized_item:
            if normalized_item.get("evidence_scope") in {
                "external_query",
                "hybrid",
            } or normalized_item.get("research_requirements"):
                normalized_item["decision_mode"] = "query_and_compare"
            else:
                normalized_item["decision_mode"] = "expert_review"
        return normalized_item

    def matching_section(item: dict[str, Any]) -> dict[str, Any] | None:
        item_number = _number_component(item.get("source_number"))
        if item_number:
            numbered = [
                section
                for section in sections
                if isinstance(section, dict)
                and (prefix := _number_component(section.get("source_number")))
                and (
                    item_number == prefix
                    or item_number.startswith(f"{prefix}-")
                )
            ]
            if len(numbered) == 1:
                return numbered[0]

        item_pages = item.get("source_pages")
        if isinstance(item_pages, list):
            page_set = {
                page
                for page in item_pages
                if isinstance(page, int) and page > 0
            }
            paged = [
                section
                for section in sections
                if isinstance(section, dict)
                and isinstance(section.get("source_pages"), list)
                and page_set
                and page_set.issubset(set(section["source_pages"]))
            ]
            if len(paged) == 1:
                return paged[0]

        dict_sections = [
            section for section in sections if isinstance(section, dict)
        ]
        return dict_sections[0] if len(dict_sections) == 1 else None

    def prefix_item_number(
        item: dict[str, Any], section: dict[str, Any] | None
    ) -> None:
        prefix = number_path(section, set()) if isinstance(section, dict) else []
        item_number = _number_component(item.get("source_number"))
        if prefix and item_number:
            current_parts = item_number.split("-")
            if current_parts[: len(prefix)] != prefix:
                item["source_number"] = "-".join(prefix + [item_number])

    flattened: list[dict[str, Any]] = []
    if isinstance(top_level_items, list):
        for raw_item in top_level_items:
            if not isinstance(raw_item, dict):
                continue
            section = matching_section(raw_item)
            item = (
                normalize_compat_item(raw_item, section)
                if top_level_uses_shorthand
                else dict(raw_item)
            )
            prefix_item_number(item, section)
            flattened.append(item)

    direct_pages: dict[str, set[int]] = {}
    for section_id, nested_items in nested_items_by_section.items():
        section = by_id.get(section_id)
        for raw_item in nested_items:
            item = (
                normalize_compat_item(raw_item, section)
                if isinstance(section, dict)
                else raw_item
            )
            prefix_item_number(item, section)
            flattened.append(item)
            pages = item.get("source_pages")
            if isinstance(pages, list):
                direct_pages.setdefault(section_id, set()).update(
                    page for page in pages if isinstance(page, int) and page > 0
                )

    children: dict[str, list[str]] = {}
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_id = section.get("section_id")
        parent_id = section.get("parent_section_id")
        if isinstance(section_id, str) and isinstance(parent_id, str):
            children.setdefault(parent_id, []).append(section_id)

    def section_pages(section_id: str, seen: set[str]) -> set[int]:
        if section_id in seen:
            return set()
        pages = set(direct_pages.get(section_id, set()))
        for child_id in children.get(section_id, []):
            pages.update(section_pages(child_id, seen | {section_id}))
        return pages

    for section in sections:
        if not isinstance(section, dict):
            continue
        section["level"] = section_level(section, set())
        if "source_pages" not in section:
            section_id = section.get("section_id")
            pages = (
                section_pages(section_id, set())
                if isinstance(section_id, str)
                else set()
            )
            section["source_pages"] = sorted(pages)

    normalized = dict(payload)
    normalized["sections"] = sections
    normalized["review_items"] = flattened
    return normalized


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
            raw_payload = json.loads(_response_text(response))
            normalized_payload = _normalize_rule_parse_payload(raw_payload)
            parsed = RuleParseResult.model_validate(normalized_payload)
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
        except Exception as exc:
            LOGGER.error(
                "Rule-set import failed for %s (error_type=%s)",
                rule_set_id,
                type(exc).__name__,
            )
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
