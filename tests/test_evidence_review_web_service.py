from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evidence_review.repository import EvidenceReviewRepository
from app.evidence_review.schemas import EvidencePackage, ResearchPackage
from app.evidence_review.web_service import (
    EvidenceReviewWebService,
    RuleSetNotActiveError,
)
from app.service import ContractNotFoundError, ContractNotReadyError


def parsed_rules() -> dict:
    items = []
    for item_id, scope in [
        ("found", "contract"),
        ("failed", "contract"),
        ("research", "external_query"),
    ]:
        items.append(
            {
                "item_id": item_id,
                "source_number": item_id,
                "section_path": ["测试"],
                "name": item_id,
                "rule_text": f"{item_id} rule",
                "source_pages": [1],
                "item_kind": "review_check",
                "evidence_scope": scope,
                "decision_mode": "expert_review",
                "retrieval_queries": [f"query {item_id}"],
                "fact_requirements": [],
                "research_requirements": [],
            }
        )
    return {
        "schema_version": "1.0",
        "document_title": "三项规则",
        "sections": [],
        "review_items": items,
        "summary": {
            "section_count": 0,
            "review_item_count": 3,
            "review_check_count": 3,
            "process_control_count": 0,
        },
    }


class ContractServiceStub:
    def __init__(self, contract):
        self.contract = contract

    def get_contract(self, contract_id: str):
        if self.contract is None or self.contract.contract_id != contract_id:
            return None
        return self.contract


class FakeEvidenceService:
    def run(self, contract_id, items, progress_callback=None):
        assert contract_id == "c1"
        outputs = [
            EvidencePackage(
                rule_item_id="found",
                evidence_status="found",
                evidence=[
                    {
                        "source_object_index": 3,
                        "page_idx": 0,
                        "node_type": "text",
                        "evidence_text": "付款期限为30日",
                    }
                ],
                extracted_facts=[],
                missing_sources=[],
                research_package=None,
                item_error=None,
            ),
            EvidencePackage(
                rule_item_id="failed",
                evidence_status="extraction_failed",
                evidence=[],
                extracted_facts=[],
                missing_sources=[],
                research_package=None,
                item_error="该审查项取证失败，请人工处理。",
            ),
            EvidencePackage(
                rule_item_id="research",
                evidence_status="not_found",
                evidence=[],
                extracted_facts=[],
                missing_sources=[],
                research_package=ResearchPackage(
                    required=True,
                    research_status="pending",
                    source_types=["public_query"],
                    reason="需要查询企业状态",
                    query_targets=[],
                    query_topics=["企业登记状态"],
                    comparison_points=["主体是否存续"],
                    missing_identifiers=["counterparty_name"],
                ),
                item_error=None,
            ),
        ]
        for index, item in enumerate(items, start=1):
            if progress_callback:
                progress_callback(
                    "evidence_item_completed",
                    {
                        "completed": index,
                        "total": len(items),
                        "rule_item_id": item.item_id,
                    },
                )
        return outputs


def active_rule_set(repository: EvidenceReviewRepository, tmp_path: Path):
    source = tmp_path / "rules.md"
    source.write_text("rules", encoding="utf-8")
    record = repository.create_rule_set(
        name="规则",
        source_filename="rules.md",
        source_path=source,
        source_sha256="abc",
    )
    repository.mark_rule_set_processing(record.rule_set_id)
    repository.mark_rule_set_draft(record.rule_set_id, parsed_rules())
    return repository.activate_rule_set(record.rule_set_id)


def build_service(tmp_path: Path, *, contract_status="ready"):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    rule_set = active_rule_set(repository, tmp_path)
    contract = SimpleNamespace(contract_id="c1", status=contract_status)
    service = EvidenceReviewWebService(
        contract_service=ContractServiceStub(contract),
        repository=repository,
        evidence_service_factory=lambda **kwargs: FakeEvidenceService(),
    )
    return service, repository, rule_set


def test_run_persists_found_failed_and_research_items_and_reaches_ready(
    tmp_path: Path,
):
    service, repository, rule_set = build_service(tmp_path)
    run = service.create_run("c1", rule_set.rule_set_id)

    completed = service.execute_run(run.run_id)
    payload = service.get_run_payload("c1", run.run_id)

    assert completed.status == "ready"
    assert completed.human_status == "pending"
    assert completed.progress == {
        "stage": "completed",
        "message": "已完成 3 / 3",
        "completed": 3,
        "total": 3,
    }
    assert [
        item["evidence_package"]["evidence_status"]
        for item in payload["items"]
    ] == ["found", "extraction_failed", "not_found"]
    assert payload["items"][2]["evidence_package"]["research_package"][
        "research_status"
    ] == "pending"
    assert len(repository.list_evidence_items(run.run_id)) == 3


def test_create_run_validates_contract_and_active_rule_set(tmp_path: Path):
    service, repository, rule_set = build_service(tmp_path)

    service.contract_service = ContractServiceStub(None)
    with pytest.raises(ContractNotFoundError):
        service.create_run("missing", rule_set.rule_set_id)

    service.contract_service = ContractServiceStub(
        SimpleNamespace(contract_id="c1", status="processing")
    )
    with pytest.raises(ContractNotReadyError):
        service.create_run("c1", rule_set.rule_set_id)

    other = repository.create_rule_set(
        name="草稿",
        source_filename="draft.md",
        source_path=tmp_path / "draft.md",
        source_sha256="draft",
    )
    service.contract_service = ContractServiceStub(
        SimpleNamespace(contract_id="c1", status="ready")
    )
    with pytest.raises(RuleSetNotActiveError):
        service.create_run("c1", other.rule_set_id)


def test_run_uses_rule_snapshot_even_if_caller_mutates_returned_payload(
    tmp_path: Path,
):
    service, repository, rule_set = build_service(tmp_path)
    run = service.create_run("c1", rule_set.rule_set_id)
    external = run.rule_snapshot
    external["review_items"][0]["rule_text"] = "tampered"

    completed = service.execute_run(run.run_id)

    assert completed.status == "ready"
    stored_items = repository.list_evidence_items(run.run_id)
    assert stored_items[0].rule_snapshot["rule_text"] == "found rule"


def test_recover_interrupted_runs_uses_public_message(tmp_path: Path):
    service, repository, rule_set = build_service(tmp_path)
    run = service.create_run("c1", rule_set.rule_set_id)

    assert service.recover_interrupted_runs() == 1
    recovered = repository.get_evidence_run(run.run_id)
    assert recovered.status == "failed"
    assert recovered.error_message == "服务重启导致人工取证任务中断，请重新创建任务。"
