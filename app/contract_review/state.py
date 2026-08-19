from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.contract_review.schemas import ReviewItem, ReviewResult, ReviewSummary


class ContractReviewState(TypedDict):
    contract_id: str
    review_rule_text: str
    review_items: list[ReviewItem]
    current_item_index: int
    review_results: Annotated[list[ReviewResult], operator.add]
    summary: ReviewSummary | None
