from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


RiskStatus = Literal["risk", "no_obvious_risk", "needs_review"]
RiskLevel = Literal["high", "medium", "low"]
EvidenceStatus = Literal["found", "insufficient"]
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
    if risk_status == "risk" and risk_level is None:
        raise ValueError("risk_status=risk requires a risk_level")
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
