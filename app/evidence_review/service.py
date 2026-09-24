from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import model_validator

from app.evidence_review.prompts import build_fact_extraction_prompt
from app.evidence_review.schemas import (
    Evidence,
    EvidencePackage,
    ExtractedFact,
    QueryTarget,
    ResearchPackage,
    RuleItem,
    StrictModel,
)
from app.llm_gateway import invoke_llm, llm_backend_kwargs

LOGGER = logging.getLogger(__name__)
FACT_EXTRACTION_TIMEOUT_SECONDS = 120.0
SAFE_ITEM_ERROR = "该审查项取证失败，请人工处理。"
ProgressCallback = Callable[[str, dict[str, Any]], None]


class FactExtractionResult(StrictModel):
    extracted_facts: list[ExtractedFact]
    missing_sources: list[str]

    @model_validator(mode="after")
    def merge_repeated_facts(self) -> FactExtractionResult:
        merged: list[ExtractedFact] = []
        positions: dict[tuple[str, str, str | None], int] = {}
        for fact in self.extracted_facts:
            identity = (fact.fact_key, fact.value, fact.unit)
            if identity not in positions:
                positions[identity] = len(merged)
                merged.append(fact)
                continue
            index = positions[identity]
            previous = merged[index]
            merged[index] = previous.model_copy(update={
                "evidence_indices": list(dict.fromkeys(
                    [*previous.evidence_indices, *fact.evidence_indices]
                )),
            })
        self.extracted_facts = merged
        return self


def build_fact_extraction_llm() -> Any:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "contract_evidence_fact_extraction",
            "strict": True,
            "schema": FactExtractionResult.model_json_schema(),
        },
    }
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0,
        timeout=FACT_EXTRACTION_TIMEOUT_SECONDS,
        max_retries=0,
        reasoning_effort="none",
        **llm_backend_kwargs(),
    ).bind(response_format=response_format)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            value.get("text", "")
            for value in content
            if isinstance(value, dict) and isinstance(value.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    raise ValueError("事实提取模型没有返回 JSON 文本。")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _is_sensitive_fact(fact_key: str, label: str) -> bool:
    key = fact_key.casefold()
    text = f"{key} {label}".casefold()
    if any(
        marker in text
        for marker in (
            "bank_account",
            "account_number",
            "银行卡号",
            "银行账号",
            "收款账号",
            "phone",
            "mobile",
            "telephone",
            "手机号",
            "电话号码",
            "联系电话",
            "identity_number",
            "identity_card",
            "id_card",
            "id_number",
            "身份证",
            "证件号码",
            "personal_address",
            "home_address",
            "个人住址",
            "家庭住址",
            "家庭地址",
            "联系地址",
        )
    ):
        return True
    if "address" in key and key not in {
        "registered_address",
        "company_address",
        "project_address",
        "site_address",
    }:
        return True
    return False


class EvidenceReviewService:
    def __init__(self, *, contract_service: Any, fact_llm: Any):
        self.contract_service = contract_service
        self.fact_llm = fact_llm

    def extract_item(self, contract_id: str, item: RuleItem) -> EvidencePackage:
        try:
            return self._extract_item(contract_id, item)
        except Exception as exc:
            LOGGER.error(
                "Evidence extraction failed for contract=%s item=%s "
                "(error_type=%s)",
                contract_id,
                item.item_id,
                type(exc).__name__,
            )
            return EvidencePackage(
                rule_item_id=item.item_id,
                evidence_status="extraction_failed",
                evidence=[],
                extracted_facts=[],
                missing_sources=[],
                research_package=None,
                item_error=SAFE_ITEM_ERROR,
            )

    def _extract_item(
        self,
        contract_id: str,
        item: RuleItem,
    ) -> EvidencePackage:
        if item.item_kind == "process_control":
            missing_sources = [
                f"流程控制项需要人工确认：{item.rule_text}"
            ]
            return EvidencePackage(
                rule_item_id=item.item_id,
                evidence_status="source_missing",
                evidence=[],
                extracted_facts=[],
                missing_sources=missing_sources,
                research_package=self._build_research_package(item, []),
                item_error=None,
            )

        evidence = self._retrieve_evidence(contract_id, item)
        facts: list[ExtractedFact] = []
        missing_sources: list[str] = []
        if evidence and item.fact_requirements:
            extracted = FactExtractionResult.model_validate_json(
                _response_text(
                    invoke_llm(
                        self.fact_llm,
                        build_fact_extraction_prompt(item, evidence),
                    )
                )
            )
            expected = {
                requirement.fact_key: requirement
                for requirement in item.fact_requirements
            }
            for fact in extracted.extracted_facts:
                requirement = expected.get(fact.fact_key)
                if requirement is None:
                    raise ValueError("model returned an unrequested fact")
                facts.append(
                    fact.model_copy(update={"label": requirement.label})
                )
            missing_sources = _unique(extracted.missing_sources)

        if item.evidence_scope == "internal_material" and not evidence:
            missing_sources.append("规则要求的内部材料尚未提供")
        missing_sources = _unique(missing_sources)

        if missing_sources:
            evidence_status = "source_missing"
        elif evidence:
            evidence_status = "found"
        else:
            evidence_status = "not_found"

        # Constructing the package performs the final evidence-index check.
        return EvidencePackage(
            rule_item_id=item.item_id,
            evidence_status=evidence_status,
            evidence=evidence,
            extracted_facts=facts,
            missing_sources=missing_sources,
            research_package=self._build_research_package(item, facts),
            item_error=None,
        )

    def _retrieve_evidence(
        self,
        contract_id: str,
        item: RuleItem,
    ) -> list[Evidence]:
        if item.evidence_scope not in {"contract", "hybrid"}:
            return []

        evidence: list[Evidence] = []
        seen_indices: set[int] = set()
        for query in item.retrieval_queries:
            for raw in self.contract_service.search_contract(contract_id, query):
                if isinstance(raw, Evidence):
                    value = raw
                else:
                    evidence_text = raw.get("evidence_text") or raw.get("text")
                    value = Evidence.model_validate(
                        {
                            "source_object_index": raw.get(
                                "source_object_index"
                            ),
                            "page_idx": raw.get("page_idx"),
                            "node_type": raw.get("node_type"),
                            "evidence_text": evidence_text,
                        }
                    )
                if value.source_object_index in seen_indices:
                    continue
                evidence.append(value)
                seen_indices.add(value.source_object_index)
        return evidence

    def _build_research_package(
        self,
        item: RuleItem,
        facts: list[ExtractedFact],
    ) -> ResearchPackage | None:
        required = item.evidence_scope in {
            "internal_material",
            "external_query",
            "hybrid",
        } or bool(item.research_requirements)
        if not required:
            return None

        fact_values: dict[str, list[str]] = {}
        for fact in facts:
            value = f"{fact.value} {fact.unit}" if fact.unit else fact.value
            values = fact_values.setdefault(fact.fact_key, [])
            if value not in values:
                values.append(value)
        fact_labels = {
            value.fact_key: value.label for value in item.fact_requirements
        }
        targets: list[QueryTarget] = []
        missing_identifiers: list[str] = []
        source_types: list[str] = []
        query_topics: list[str] = []
        comparison_points: list[str] = []

        for requirement in item.research_requirements:
            source_types.append(requirement.source_type)
            query_topics.extend(requirement.query_topics)
            comparison_points.extend(requirement.comparison_points)
            fields: dict[str, str] = {}
            for key in requirement.required_fact_keys:
                if _is_sensitive_fact(key, fact_labels.get(key, "")):
                    continue
                if key in fact_values:
                    fields[key] = "；".join(fact_values[key])
                else:
                    missing_identifiers.append(key)
            targets.append(
                QueryTarget(
                    target_type=requirement.target_type,
                    fields=fields,
                )
            )

        if not source_types:
            fallback_type = {
                "internal_material": "internal_material",
                "external_query": "public_query",
                "hybrid": "public_query",
            }.get(item.evidence_scope, "expert_opinion")
            source_types.append(fallback_type)

        return ResearchPackage(
            required=True,
            research_status="pending",
            source_types=_unique(source_types),
            reason="该审查项需要补充合同外资料并由人工查询比对。",
            query_targets=targets,
            query_topics=_unique(query_topics),
            comparison_points=_unique(comparison_points),
            missing_identifiers=_unique(missing_identifiers),
        )

    def run(
        self,
        contract_id: str,
        items: list[RuleItem],
        progress_callback: ProgressCallback | None = None,
    ) -> list[EvidencePackage]:
        packages: list[EvidencePackage] = []
        total = len(items)
        for index, item in enumerate(items):
            if progress_callback is not None:
                progress_callback(
                    "evidence_item_started",
                    {
                        "completed": index,
                        "total": total,
                        "rule_item_id": item.item_id,
                    },
                )
            packages.append(self.extract_item(contract_id, item))
            if progress_callback is not None:
                progress_callback(
                    "evidence_item_completed",
                    {
                        "completed": index + 1,
                        "total": total,
                        "rule_item_id": item.item_id,
                    },
                )
        return packages
