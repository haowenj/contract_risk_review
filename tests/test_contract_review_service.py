import json
from types import SimpleNamespace

import pytest

from app.contract_review.service import ContractReviewService
from app.service import ContractNotFoundError, ContractNotReadyError


ITEMS = {
    "review_items": [
        {
            "id": "item_1",
            "name": "付款期限",
            "rule_basis": "付款期限不得超过90日",
            "review_goal": "判断合同付款期限是否超过90日",
            "retrieval_query": "合同约定的付款期限是多久",
        }
    ]
}

DECISION = {
    "risk_status": "risk",
    "risk_level": "high",
    "evidence_status": "found",
    "finding": "合同约定付款期限为180日。",
    "risk_description": "付款期限超过90日。",
    "suggestion": "将付款期限调整到90日以内。",
}

EVIDENCE = {
    "source_object_index": 27,
    "page_idx": 4,
    "node_type": "text",
    "text": "付款期限为180日",
    "evidence_text": "付款期限为180日",
}


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        if isinstance(self.payload, str):
            content = self.payload
        else:
            content = json.dumps(self.payload, ensure_ascii=False)
        return SimpleNamespace(content=content, additional_kwargs={})


class FakeContractService:
    def __init__(self, contract=None, evidence=None):
        self.contract = contract
        self.evidence = EVIDENCE if evidence is None else evidence
        self.get_calls = []
        self.searches = []

    def get_contract(self, contract_id):
        self.get_calls.append(contract_id)
        return self.contract

    def search_contract(self, contract_id, query):
        self.searches.append((contract_id, query))
        return [self.evidence] if self.evidence else []


def build_service(contract_service, parse_payload=ITEMS, decision_payload=DECISION):
    return ContractReviewService(
        contract_service=contract_service,
        parse_llm=FakeLLM(parse_payload),
        review_llm=FakeLLM(decision_payload),
    )


def ready_contract():
    return SimpleNamespace(contract_id="contract-1", status="ready")


def test_service_rejects_blank_inputs_before_contract_lookup():
    contract_service = FakeContractService(ready_contract())
    service = build_service(contract_service)

    with pytest.raises(ValueError, match="contract_id must not be empty"):
        service.run("  ", "规范")
    with pytest.raises(ValueError, match="review_rule_text must not be empty"):
        service.run("contract-1", "  ")

    assert contract_service.get_calls == []
    assert contract_service.searches == []


def test_service_preflight_requires_existing_ready_contract_without_rag():
    missing_service = FakeContractService(None)
    with pytest.raises(ContractNotFoundError):
        build_service(missing_service).run("missing", "规范")
    assert missing_service.searches == []

    queued = SimpleNamespace(contract_id="contract-1", status="queued")
    queued_service = FakeContractService(queued)
    with pytest.raises(ContractNotReadyError):
        build_service(queued_service).run("contract-1", "规范")
    assert queued_service.searches == []


def test_service_parse_failure_proves_preflight_does_not_trigger_rag():
    contract_service = FakeContractService(ready_contract())
    service = build_service(contract_service, parse_payload="not-json")

    with pytest.raises(RuntimeError, match="parse_review_rules failed"):
        service.run("contract-1", "付款期限不得超过90日")

    assert contract_service.get_calls == ["contract-1"]
    assert contract_service.searches == []


def test_service_runs_graph_and_returns_json_serializable_result():
    contract_service = FakeContractService(ready_contract())
    service = build_service(contract_service)

    result = service.run("contract-1", "付款期限不得超过90日")

    assert result["contract_id"] == "contract-1"
    assert result["review_items"][0]["id"] == "item_1"
    assert result["review_results"][0]["risk_status"] == "risk"
    assert result["review_results"][0]["evidence"][0]["source_object_index"] == 27
    assert result["summary"] == {
        "total_items": 1,
        "risk_count": 1,
        "high_risk_count": 1,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "no_obvious_risk_count": 0,
        "needs_review_count": 0,
    }
    json.dumps(result, ensure_ascii=False)
    assert contract_service.get_calls == ["contract-1"]
    assert contract_service.searches == [
        ("contract-1", "合同约定的付款期限是多久")
    ]
