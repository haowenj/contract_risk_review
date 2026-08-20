from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractReviewRun:
    run_id: str
    contract_id: str
    status: str
    review_rule_text: str
    result: dict[str, Any]
    progress: dict[str, Any]
    created_at: str
    started_at: str | None
    completed_at: str | None
    error_message: str | None
