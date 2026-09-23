from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from app.evidence_review.repository import (
    EvidenceReviewRepository,
    RuleSetRecord,
)
from app.evidence_review.rule_batching import RuleParseBatch, split_rule_document
from app.evidence_review.schemas import RuleParseChunkResult, RuleParseResult
from app.llm_gateway import MAX_CONCURRENT_LLM_CALLS, invoke_llm
from mineru_raw_parse import run_parse

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


run_rule_parse = run_parse


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
            if value.get("type") not in {None, "text", "paragraph_title"}:
                continue
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                text = value.get("content")
            if not isinstance(text, str) or not text.strip():
                continue
            page_idx = value.get("page_idx")
            page_number = page_idx + 1 if isinstance(page_idx, int) else 1
            block: dict[str, object] = {
                "page_number": page_number,
                "text": text.strip(),
            }
            if isinstance(value.get("text_level"), int):
                block["text_level"] = value["text_level"]
            blocks.append(block)
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
            "schema": RuleParseChunkResult.model_json_schema(),
        },
    }
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0,
        max_tokens=8192,
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


def _merge_rule_batches(
    chunks: list[RuleParseChunkResult],
) -> RuleParseResult:
    if len(chunks) == 1:
        return RuleParseResult.model_validate(chunks[0].model_dump(mode="json"))

    sections: list[dict[str, Any]] = []
    sections_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    items_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for chunk in chunks:
        local_sections = {section.section_id: section for section in chunk.sections}

        def section_key(
            section_id: str,
            seen: set[str],
            local_sections: dict[str, Any] = local_sections,
        ) -> tuple[Any, ...]:
            if section_id in seen:
                raise ValueError("规则章节存在循环引用")
            section = local_sections[section_id]
            parent_key = (
                section_key(section.parent_section_id, seen | {section_id})
                if section.parent_section_id else ()
            )
            return parent_key + ((section.source_number, section.title),)

        for section in sorted(
            chunk.sections,
            key=lambda value: len(section_key(value.section_id, set())),
        ):
            key = section_key(section.section_id, set())
            existing = sections_by_key.get(key)
            if existing is None:
                parent_key = key[:-1]
                parent = sections_by_key.get(parent_key) if parent_key else None
                if parent_key and parent is None:
                    raise ValueError("规则章节缺少父章节")
                existing = section.model_dump(mode="json")
                existing["section_id"] = f"section-{len(sections) + 1}"
                existing["parent_section_id"] = (
                    parent["section_id"] if parent else None
                )
                existing["level"] = len(key)
                sections_by_key[key] = existing
                sections.append(existing)
            else:
                existing["source_pages"] = sorted(
                    set(existing["source_pages"]) | set(section.source_pages)
                )

        for item in chunk.review_items:
            key = (
                tuple(item.section_path),
                item.source_number,
                " ".join(item.rule_text.split()),
            )
            existing = next(
                (
                    candidate
                    for candidate in items_by_key.get(key, [])
                    if any(
                        abs(existing_page - incoming_page) <= 1
                        for existing_page in candidate["source_pages"]
                        for incoming_page in item.source_pages
                    )
                ),
                None,
            )
            if existing is None:
                existing = item.model_dump(mode="json")
                existing["item_id"] = f"item-{len(items) + 1}"
                items_by_key.setdefault(key, []).append(existing)
                items.append(existing)
            else:
                incoming = item.model_dump(mode="json")
                for field in ("item_kind", "evidence_scope", "decision_mode"):
                    if existing[field] != incoming[field]:
                        raise ValueError("跨批规则分类不一致")
                existing["source_pages"] = sorted(
                    set(existing["source_pages"]) | set(item.source_pages)
                )
                existing["retrieval_queries"] = list(dict.fromkeys(
                    existing["retrieval_queries"] + incoming["retrieval_queries"]
                ))
                known_facts = {
                    fact["fact_key"]: fact
                    for fact in existing["fact_requirements"]
                }
                for fact in incoming["fact_requirements"]:
                    previous = known_facts.get(fact["fact_key"])
                    if previous is None:
                        existing["fact_requirements"].append(fact)
                        known_facts[fact["fact_key"]] = fact
                    elif previous != fact:
                        raise ValueError("跨批事实字段定义不一致")
                for requirement in incoming["research_requirements"]:
                    if requirement not in existing["research_requirements"]:
                        existing["research_requirements"].append(requirement)

    return RuleParseResult.model_validate({
        "document_title": chunks[0].document_title,
        "sections": sections,
        "review_items": items,
    })


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

            batches = split_rule_document(document)

            def parse_batch(batch: RuleParseBatch) -> RuleParseChunkResult:
                prompt = build_rule_parse_prompt(
                    batch.document,
                    context_before=batch.context_before,
                    context_after=batch.context_after,
                )
                response = invoke_llm(self.parse_llm, prompt)
                metadata = getattr(response, "response_metadata", {}) or {}
                if metadata.get("finish_reason") == "length":
                    raise ValueError("规则解析模型输出达到 Token 上限")
                raw_payload = json.loads(_response_text(response))
                normalized_payload = _normalize_rule_parse_payload(raw_payload)
                return RuleParseChunkResult.model_validate(normalized_payload)

            with ThreadPoolExecutor(
                max_workers=min(MAX_CONCURRENT_LLM_CALLS, len(batches))
            ) as executor:
                chunks = list(executor.map(parse_batch, batches))
            parsed = _merge_rule_batches(chunks)
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
