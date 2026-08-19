from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.contract_review.schemas import (
    Evidence,
    ReviewItem,
    ReviewResult,
    ReviewSummary,
    RiskDecision,
)


class ContractReviewState(TypedDict):
    contract_id: str
    review_rule_text: str
    review_items: list[ReviewItem]
    current_item_index: int
    review_results: Annotated[list[ReviewResult], operator.add]
    summary: ReviewSummary | None
    retrieval_attempt: int
    current_retrieval_query: str
    retrieved_evidence: list[Evidence]
    absence_keywords: list[str]
    absence_candidates: list[Evidence]
    absence_candidate_count: int | None
    current_decision: RiskDecision | None
