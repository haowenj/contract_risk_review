from __future__ import annotations

from pathlib import Path

import pytest

from app.evidence_review.repository import EvidenceReviewRepository


def rule_snapshot() -> dict:
    return {
        "document_title": "合同审核规则",
        "sections": [],
        "review_items": [
            {
                "item_id": "item-1",
                "name": "付款条件",
                "rule_text": "核验付款条件",
            }
        ],
    }


def package(item_id: str, status: str = "found") -> dict:
    return {
        "rule_item_id": item_id,
        "evidence_status": status,
        "evidence": [],
        "extracted_facts": [],
        "missing_sources": [],
        "research_package": None,
        "item_error": None,
    }


def test_machine_ready_keeps_human_review_pending(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    run = repository.create_evidence_run(
        contract_id="c1",
        rule_set_id="rules-1",
        rule_snapshot=rule_snapshot(),
    )
    repository.mark_evidence_run_processing(run.run_id)

    ready = repository.complete_evidence_run(
        run.run_id,
        [
            {
                "rule_item_id": "item-1",
                "rule_snapshot": rule_snapshot()["review_items"][0],
                "evidence_package": package("item-1"),
            }
        ],
        progress={"stage": "completed", "completed": 1, "total": 1},
    )

    assert ready.status == "ready"
    assert ready.human_status == "pending"
    assert repository.list_evidence_items(run.run_id)[0].rule_item_id == "item-1"


def test_run_rule_snapshot_is_an_immutable_copy(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    snapshot = rule_snapshot()
    run = repository.create_evidence_run(
        contract_id="c1",
        rule_set_id="rules-1",
        rule_snapshot=snapshot,
    )

    snapshot["review_items"][0]["rule_text"] = "后来被修改"

    stored = repository.get_evidence_run(run.run_id)
    assert stored.rule_snapshot["review_items"][0]["rule_text"] == "核验付款条件"


def test_duplicate_item_keys_are_rejected_without_partial_completion(
    tmp_path: Path,
):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    run = repository.create_evidence_run(
        contract_id="c1",
        rule_set_id="rules-1",
        rule_snapshot=rule_snapshot(),
    )
    repository.mark_evidence_run_processing(run.run_id)
    duplicate = {
        "rule_item_id": "item-1",
        "rule_snapshot": rule_snapshot()["review_items"][0],
        "evidence_package": package("item-1"),
    }

    with pytest.raises(ValueError, match="unique"):
        repository.complete_evidence_run(
            run.run_id,
            [duplicate, duplicate],
            progress={"stage": "completed", "completed": 2, "total": 2},
        )

    assert repository.get_evidence_run(run.run_id).status == "processing"
    assert repository.list_evidence_items(run.run_id) == []


def test_interrupted_runs_recover_to_safe_failed_state(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    queued = repository.create_evidence_run(
        contract_id="c1",
        rule_set_id="rules-1",
        rule_snapshot=rule_snapshot(),
    )
    processing = repository.create_evidence_run(
        contract_id="c2",
        rule_set_id="rules-1",
        rule_snapshot=rule_snapshot(),
    )
    repository.mark_evidence_run_processing(processing.run_id)

    count = repository.recover_incomplete_evidence_runs(
        "服务重启导致人工取证任务中断，请重新创建任务。"
    )

    assert count == 2
    for run_id in [queued.run_id, processing.run_id]:
        recovered = repository.get_evidence_run(run_id)
        assert recovered.status == "failed"
        assert recovered.human_status == "pending"
        assert recovered.error_message == "服务重启导致人工取证任务中断，请重新创建任务。"
