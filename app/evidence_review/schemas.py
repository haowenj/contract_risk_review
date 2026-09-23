from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contract_review.schemas import RiskDecision

EvidenceScope = Literal[
    "contract",
    "internal_material",
    "external_query",
    "hybrid",
]
DecisionMode = Literal[
    "automatic_structure_check",
    "threshold_required",
    "expert_review",
    "query_and_compare",
]
ItemKind = Literal["review_check", "process_control"]
EvidenceStatus = Literal[
    "found",
    "not_found",
    "source_missing",
    "extraction_failed",
]
ResearchStatus = Literal["not_required", "pending", "completed"]
HumanReviewStatus = Literal["pending", "completed"]
HumanDecisionValue = Literal[
    "risk",
    "no_obvious_risk",
    "cannot_determine",
]
RiskLevel = Literal["high", "medium", "low"]
SourceType = Literal[
    "public_query",
    "internal_material",
    "professional_database",
    "expert_opinion",
]
FactValueType = Literal[
    "string",
    "integer",
    "decimal",
    "date",
    "percentage",
    "currency",
    "boolean",
]
NodeType = Literal["text", "table", "image"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RuleSection(StrictModel):
    section_id: str = Field(min_length=1)
    source_number: str | None = None
    title: str = Field(min_length=1)
    parent_section_id: str | None = None
    level: int = Field(ge=1)
    source_pages: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pages(self) -> RuleSection:
        if any(page < 1 for page in self.source_pages):
            raise ValueError("source_pages must use positive one-based numbers")
        return self


class FactRequirement(StrictModel):
    fact_key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value_type: FactValueType
    required: bool


class ResearchRequirement(StrictModel):
    source_type: SourceType
    target_type: str = Field(min_length=1)
    required_fact_keys: list[str]
    query_topics: list[str] = Field(min_length=1)
    comparison_points: list[str] = Field(min_length=1)


class RuleItem(StrictModel):
    item_id: str = Field(min_length=1)
    source_number: str | None = None
    section_path: list[str]
    name: str = Field(min_length=1)
    rule_text: str = Field(min_length=1)
    source_pages: list[int] = Field(min_length=1)
    item_kind: ItemKind
    evidence_scope: EvidenceScope
    decision_mode: DecisionMode
    retrieval_queries: list[str] = Field(min_length=1)
    fact_requirements: list[FactRequirement]
    research_requirements: list[ResearchRequirement]

    @model_validator(mode="after")
    def validate_lists(self) -> RuleItem:
        if any(page < 1 for page in self.source_pages):
            raise ValueError("source_pages must use positive one-based numbers")
        if any(not query.strip() for query in self.retrieval_queries):
            raise ValueError("retrieval_queries must not contain empty values")
        fact_keys = [value.fact_key for value in self.fact_requirements]
        if len(fact_keys) != len(set(fact_keys)):
            raise ValueError("fact requirement keys must be unique")
        return self


class RuleParseResult(StrictModel):
    document_title: str = Field(min_length=1)
    sections: list[RuleSection]
    review_items: list[RuleItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> RuleParseResult:
        section_ids = [value.section_id for value in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("section ids must be unique")
        item_ids = [value.item_id for value in self.review_items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("review item ids must be unique")
        known_sections = set(section_ids)
        if any(
            value.parent_section_id is not None
            and value.parent_section_id not in known_sections
            for value in self.sections
        ):
            raise ValueError("parent section ids must reference existing sections")
        return self


class RuleParseChunkResult(RuleParseResult):
    """A batch may contain only headings; the merged document may not."""

    review_items: list[RuleItem]


class Evidence(StrictModel):
    source_object_index: int = Field(ge=0)
    page_idx: int | None = Field(default=None, ge=0)
    node_type: NodeType
    evidence_text: str = Field(min_length=1)


class ExtractedFact(StrictModel):
    fact_key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    unit: str | None = None
    evidence_indices: list[int] = Field(min_length=1)


class QueryTarget(StrictModel):
    target_type: str = Field(min_length=1)
    fields: dict[str, str]


class ResearchPackage(StrictModel):
    required: bool
    research_status: ResearchStatus
    source_types: list[SourceType]
    reason: str = Field(min_length=1)
    query_targets: list[QueryTarget]
    query_topics: list[str]
    comparison_points: list[str]
    missing_identifiers: list[str]


class EvidencePackage(StrictModel):
    rule_item_id: str = Field(min_length=1)
    evidence_status: EvidenceStatus
    evidence: list[Evidence]
    extracted_facts: list[ExtractedFact]
    missing_sources: list[str]
    research_package: ResearchPackage | None
    item_error: str | None
    # Older saved runs may contain this field; current reviews do not generate or show it.
    system_suggestion: RiskDecision | None = None

    @model_validator(mode="after")
    def validate_fact_references(self) -> EvidencePackage:
        evidence_count = len(self.evidence)
        for fact in self.extracted_facts:
            if any(
                index < 0 or index >= evidence_count
                for index in fact.evidence_indices
            ):
                raise ValueError(
                    "evidence_indices must reference evidence in this package"
                )
        return self


class HumanDecision(StrictModel):
    human_review_status: HumanReviewStatus
    decision: HumanDecisionValue | None = None
    risk_level: RiskLevel | None = None
    opinion: str = ""
    research_status: ResearchStatus
    research_notes: str = ""
    updated_at: datetime

    @model_validator(mode="after")
    def validate_decision(self) -> HumanDecision:
        if self.human_review_status == "completed":
            if self.decision is None:
                raise ValueError("completed human review requires a decision")
            if not self.opinion:
                raise ValueError("completed human review requires an opinion")
        if self.research_status == "pending" and self.decision in {
            "risk",
            "no_obvious_risk",
        }:
            raise ValueError("pending research disallows conclusive decisions")
        if self.decision != "risk" and self.risk_level is not None:
            raise ValueError("risk_level must be null for non-risk decisions")
        return self
