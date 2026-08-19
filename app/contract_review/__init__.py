"""Contract risk review workflow."""

from app.contract_review.service import (
    ContractReviewService,
    build_default_contract_review_service,
)


__all__ = [
    "ContractReviewService",
    "build_default_contract_review_service",
]
