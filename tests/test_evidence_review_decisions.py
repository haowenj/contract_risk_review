from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.evidence_review.repository import (
    DecisionConflictError,
    EvidenceReviewRepository,
)
from app.evidence_review.schemas import HumanDecision
from app.evidence_review.web_service import EvidenceReviewWebService


def item(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "source_number": item_id,
        "section_path": ["测试"],
        "name": item_id,
        "rule_text": f"{item_id} rule",
        "source_pages": [1],
        "item_kind": "review_check",
        "evidence_scope": "external_query",
        "decision_mode": "expert_review",
        "retrieval_queries": [item_id],
        "fact_requirements": [],
        "research_requirements": [],
    }


def package(item_id: str, *, research_status="pending") -> dict:
    return {
        "rule_item_id": item_id,
        "evidence_status": "not_found",
        "evidence": [],
        "extracted_facts": [],
        "missing_sources": [],
        "research_package": {
            "required": True,
            "research_status": research_status,
            "source_types": ["public_query"],
            "reason": "需要人工查询",
            "query_targets": [],
            "query_topics": ["企业状态"],
            "comparison_points": ["是否存续"],
            "missing_identifiers": [],
        },
        "item_error": None,
    }


def ready_run(repository: EvidenceReviewRepository, item_ids=("a", "b")):
    snapshot = {
        "document_title": "规则",
        "sections": [],
        "review_items": [item(value) for value in item_ids],
    }
    run = repository.create_evidence_run(
        contract_id="c1",
        rule_set_id="rules",
        rule_snapshot=snapshot,
    )
    repository.mark_evidence_run_processing(run.run_id)
    return repository.complete_evidence_run(
        run.run_id,
        [
            {
                "rule_item_id": value,
                "rule_snapshot": item(value),
                "evidence_package": package(value),
            }
            for value in item_ids
        ],
        progress={"stage": "completed", "completed": len(item_ids), "total": len(item_ids)},
    )


def web_service(repository: EvidenceReviewRepository):
    return EvidenceReviewWebService(
        contract_service=SimpleNamespace(),
        repository=repository,
    )


@pytest.mark.parametrize("decision", ["risk", "no_obvious_risk"])
def test_pending_research_blocks_conclusive_decisions(tmp_path: Path, decision: str):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    run = ready_run(repository, ("a",))
    current = repository.get_evidence_decision(run.run_id, "a")

    with pytest.raises(ValidationError, match="pending research"):
        web_service(repository).save_human_decision(
            run.run_id,
            "a",
            {
                "human_review_status": "completed",
                "decision": decision,
                "risk_level": None,
                "opinion": "人工意见",
                "research_status": "pending",
                "research_notes": "尚未查询",
            },
            current.updated_at,
        )


def test_completed_decision_requires_opinion(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    run = ready_run(repository, ("a",))
    current = repository.get_evidence_decision(run.run_id, "a")

    with pytest.raises(ValidationError, match="opinion"):
        web_service(repository).save_human_decision(
            run.run_id,
            "a",
            {
                "human_review_status": "completed",
                "decision": "cannot_determine",
                "risk_level": None,
                "opinion": "",
                "research_status": "pending",
                "research_notes": "等待材料",
            },
            current.updated_at,
        )


def test_non_risk_level_is_rejected_but_risk_level_is_optional(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    run = ready_run(repository, ("a",))
    current = repository.get_evidence_decision(run.run_id, "a")
    service = web_service(repository)

    with pytest.raises(ValidationError, match="risk_level"):
        service.save_human_decision(
            run.run_id,
            "a",
            {
                "human_review_status": "completed",
                "decision": "no_obvious_risk",
                "risk_level": "low",
                "opinion": "人工核验无明显风险",
                "research_status": "completed",
                "research_notes": "已查询",
            },
            current.updated_at,
        )

    saved = service.save_human_decision(
        run.run_id,
        "a",
        {
            "human_review_status": "completed",
            "decision": "risk",
            "risk_level": None,
            "opinion": "人工确认存在风险，原规则未规定等级",
            "research_status": "completed",
            "research_notes": "已查询",
        },
        current.updated_at,
    )
    assert saved.decision.decision == "risk"
    assert saved.decision.risk_level is None


def test_stale_timestamp_cannot_overwrite_newer_decision(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    run = ready_run(repository, ("a",))
    current = repository.get_evidence_decision(run.run_id, "a")
    service = web_service(repository)
    data = {
        "human_review_status": "completed",
        "decision": "cannot_determine",
        "risk_level": None,
        "opinion": "资料不足，暂无法判断",
        "research_status": "pending",
        "research_notes": "等待查询",
    }

    service.save_human_decision(
        run.run_id, "a", data, current.updated_at
    )
    with pytest.raises(DecisionConflictError):
        service.save_human_decision(
            run.run_id, "a", data, current.updated_at
        )


def test_saving_decisions_recalculates_run_human_status_and_summary(
    tmp_path: Path,
):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    run = ready_run(repository)
    service = web_service(repository)

    first = repository.get_evidence_decision(run.run_id, "a")
    service.save_human_decision(
        run.run_id,
        "a",
        {
            "human_review_status": "completed",
            "decision": "cannot_determine",
            "risk_level": None,
            "opinion": "资料不足",
            "research_status": "pending",
            "research_notes": "等待查询",
        },
        first.updated_at,
    )
    assert repository.get_evidence_run(run.run_id).human_status == "in_progress"

    second = repository.get_evidence_decision(run.run_id, "b")
    service.save_human_decision(
        run.run_id,
        "b",
        {
            "human_review_status": "completed",
            "decision": "no_obvious_risk",
            "risk_level": None,
            "opinion": "人工查询后未发现明显风险",
            "research_status": "completed",
            "research_notes": "已查询企业状态",
        },
        second.updated_at,
    )

    payload = service.get_run_payload("c1", run.run_id)
    assert repository.get_evidence_run(run.run_id).human_status == "completed"
    assert payload["human_summary"] == {
        "pending_count": 0,
        "completed_count": 2,
        "risk_count": 0,
        "no_obvious_risk_count": 1,
        "cannot_determine_count": 1,
    }
    assert all(
        isinstance(value["human_decision"], dict)
        for value in payload["items"]
    )
