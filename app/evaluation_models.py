from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    case_id: int
    contract_id: str
    index_version: str
    question: str
    expected_source_object_indices: list[int]
    sort_order: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    contract_id: str
    scope: str
    status: str
    index_version: str
    pipeline_version: str
    config_snapshot: dict[str, Any]
    created_at: str
    started_at: str | None
    completed_at: str | None
    error_message: str | None


@dataclass(frozen=True)
class EvaluationRunItem:
    run_id: str
    case_id: int
    question_snapshot: str
    expected_source_object_indices_snapshot: list[int]
    result: dict[str, Any]
