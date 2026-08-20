from __future__ import annotations

import json
import unicodedata
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RiskStatus = Literal["risk", "no_obvious_risk", "needs_review"]
RiskLevel = Literal["high", "medium", "low"]
EvidenceStatus = Literal["found", "insufficient", "absence_verified"]
NodeType = Literal["text", "table", "image"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReviewItem(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    rule_basis: str = Field(min_length=1)
    review_goal: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)


class ReviewItemList(StrictModel):
    review_items: list[ReviewItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ReviewItemList:
        item_ids = [item.id for item in self.review_items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("review item ids must be unique")
        return self


def _normalize_keyword_key(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


_SURROUNDING_KEYWORD_PUNCTUATION = frozenset(
    "\"'“”‘’()（）[]【】{}《》〈〉「」『』。，、；;：:！？!?…"
)


def _strip_surrounding_punctuation(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and value[start] in _SURROUNDING_KEYWORD_PUNCTUATION:
        start += 1
    while (
        end > start
        and value[end - 1] in _SURROUNDING_KEYWORD_PUNCTUATION
    ):
        end -= 1
    return value[start:end].strip()


_GENERIC_SCAN_KEYWORDS = frozenset(
    {
        "第三方",
        "转让",
        "同意",
        "批准",
        "许可",
        "授权",
        "责任",
        "合同",
        "书面同意",
        "书面批准",
        "书面许可",
        "书面授权",
    }
)

_APPROVAL_TERMS = (
    "书面同意书",
    "书面批准书",
    "书面许可书",
    "书面授权书",
    "同意书",
    "批准书",
    "许可书",
    "授权书",
    "同意",
    "批准",
    "许可",
    "授权",
    "认可",
    "确认",
)
_GENERIC_APPROVAL_PARTS = tuple(
    sorted(
        {
            *_APPROVAL_TERMS,
            "甲方",
            "乙方",
            "双方",
            "各方",
            "一方",
            "对方",
            "相关方",
            "当事人",
            "事先",
            "预先",
            "提前",
            "书面",
            "须经",
            "需经",
            "应经",
            "未经",
            "获得",
            "取得",
            "之后",
            "以后",
            "后方可",
            "才允许",
            "才可以",
            "方可",
            "才可",
            "后",
            "可",
            "经",
            "的",
        },
        key=len,
        reverse=True,
    )
)


def _is_generic_scan_keyword(normalized_key: str) -> bool:
    classification_key = "".join(
        character
        for character in normalized_key
        if not unicodedata.category(character).startswith("P")
    )
    if classification_key in _GENERIC_SCAN_KEYWORDS:
        return True
    if not any(term in classification_key for term in _APPROVAL_TERMS):
        return False

    residual = classification_key
    for part in _GENERIC_APPROVAL_PARTS:
        residual = residual.replace(part, "")
    return not residual


def _clean_keywords(
    value: Any,
    *,
    filter_generic: bool,
    require_nonempty: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("keywords must be a list")

    cleaned: list[str] = []
    seen: set[str] = set()
    for keyword in value:
        if not isinstance(keyword, str):
            raise ValueError("keywords must contain only strings")
        display_value = _strip_surrounding_punctuation(keyword.strip())
        normalized_key = _normalize_keyword_key(display_value)
        if (
            not normalized_key
            or (filter_generic and _is_generic_scan_keyword(normalized_key))
            or normalized_key in seen
        ):
            continue
        seen.add(normalized_key)
        cleaned.append(display_value)

    if require_nonempty and not cleaned:
        raise ValueError("keywords must contain at least one non-empty value")
    return cleaned


class RetrievalQueryRewrite(StrictModel):
    retrieval_query: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    primary_keywords: list[str] = Field(min_length=1)
    secondary_keywords: list[str]

    @field_validator("primary_keywords", mode="before")
    @classmethod
    def normalize_primary_keywords(cls, value: Any) -> list[str]:
        return _clean_keywords(
            value,
            filter_generic=True,
            require_nonempty=True,
        )

    @field_validator("secondary_keywords", mode="before")
    @classmethod
    def normalize_secondary_keywords(cls, value: Any) -> list[str]:
        return _clean_keywords(
            value,
            filter_generic=False,
            require_nonempty=False,
        )

    @model_validator(mode="after")
    def remove_cross_tier_duplicates(self) -> RetrievalQueryRewrite:
        primary_keys = {
            _normalize_keyword_key(value) for value in self.primary_keywords
        }
        self.secondary_keywords = [
            value
            for value in self.secondary_keywords
            if _normalize_keyword_key(value) not in primary_keys
        ]
        return self


class AbsenceCheckMetadata(StrictModel):
    primary_keywords: list[str] = Field(min_length=1)
    secondary_keywords: list[str]
    candidate_count: int = Field(ge=0)

    @field_validator("primary_keywords", mode="before")
    @classmethod
    def normalize_primary_keywords(cls, value: Any) -> list[str]:
        return _clean_keywords(
            value,
            filter_generic=True,
            require_nonempty=True,
        )

    @field_validator("secondary_keywords", mode="before")
    @classmethod
    def normalize_secondary_keywords(cls, value: Any) -> list[str]:
        return _clean_keywords(
            value,
            filter_generic=False,
            require_nonempty=False,
        )

    @model_validator(mode="after")
    def remove_cross_tier_duplicates(self) -> AbsenceCheckMetadata:
        primary_keys = {
            _normalize_keyword_key(value) for value in self.primary_keywords
        }
        self.secondary_keywords = [
            value
            for value in self.secondary_keywords
            if _normalize_keyword_key(value) not in primary_keys
        ]
        return self


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    source_object_index: int
    page_idx: int | None = None
    node_type: NodeType
    evidence_text: str = Field(min_length=1)
    text: str | None = None
    node_id: str | None = None
    score: float | None = None


def _validate_risk_fields(
    *,
    risk_status: RiskStatus,
    risk_level: RiskLevel | None,
    evidence_status: EvidenceStatus,
) -> None:
    if evidence_status == "insufficient" and risk_status != "needs_review":
        raise ValueError(
            "insufficient evidence requires risk_status=needs_review"
        )
    if risk_status != "risk" and risk_level is not None:
        raise ValueError("risk_level must be null when risk_status is not risk")


class RiskDecision(StrictModel):
    risk_status: RiskStatus
    risk_level: RiskLevel | None
    evidence_status: EvidenceStatus
    finding: str = Field(min_length=1)
    risk_description: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status_combination(self) -> RiskDecision:
        _validate_risk_fields(
            risk_status=self.risk_status,
            risk_level=self.risk_level,
            evidence_status=self.evidence_status,
        )
        return self


class ReviewResult(RiskDecision):
    item_id: str = Field(min_length=1)
    item_name: str = Field(min_length=1)
    evidence: list[Evidence]
    absence_check: AbsenceCheckMetadata | None = None

    @model_validator(mode="after")
    def validate_absence_verified_audit(self) -> ReviewResult:
        if self.evidence_status != "absence_verified":
            return self
        if self.evidence:
            raise ValueError("absence_verified requires an empty evidence list")
        if self.absence_check is None:
            raise ValueError("absence_verified requires absence_check metadata")
        if self.absence_check.candidate_count != 0:
            raise ValueError("absence_verified requires candidate_count=0")
        return self


class ReviewSummary(StrictModel):
    total_items: int = Field(ge=0)
    risk_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    medium_risk_count: int = Field(ge=0)
    low_risk_count: int = Field(ge=0)
    no_obvious_risk_count: int = Field(ge=0)
    needs_review_count: int = Field(ge=0)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _response_text(response: Any) -> str:
    content = response if isinstance(response, str) else getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    raise ValueError("LLM response does not contain text content")


def parse_llm_response(response: Any, model_type: type[ModelT]) -> ModelT:
    if isinstance(response, dict):
        payload: Any = response
    else:
        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        payload = additional_kwargs.get("parsed")
        if payload is None:
            payload = json.loads(_response_text(response))
    if isinstance(payload, str):
        payload = json.loads(payload)
    return model_type.model_validate(payload)
