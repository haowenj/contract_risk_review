import json
from types import SimpleNamespace

import pytest

from app.contract_review.nodes import ContractReviewNodes
from app.contract_review.schemas import ReviewItem, ReviewResult


ITEMS_PAYLOAD = {
    "review_items": [
        {
            "id": "item_1",
            "name": "付款期限",
            "rule_basis": "付款期限不得超过90日",
            "review_goal": "判断合同付款期限是否超过90日",
            "retrieval_query": "合同约定的付款期限是多久",
        },
        {
            "id": "item_2",
            "name": "违约责任",
            "rule_basis": "延期履约应约定违约责任",
            "review_goal": "判断延期履约责任是否明确",
            "retrieval_query": "延期履约需要承担什么违约责任",
        },
    ]
}

RISK_DECISION = {
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
    "node_type": "table",
    "text": "付款期限：180日",
    "evidence_text": "付款期限：180日",
    "table_body": "<table><tr><td>180日</td></tr></table>",
}


class FakeLLM:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            return SimpleNamespace(content=payload, additional_kwargs={})
        return SimpleNamespace(
            content=json.dumps(payload, ensure_ascii=False),
            additional_kwargs={},
        )


class FakeContractService:
    def __init__(self, *evidence_sets):
        self.evidence_sets = list(evidence_sets)
        self.searches = []

    def search_contract(self, contract_id, query):
        self.searches.append((contract_id, query))
        evidence = self.evidence_sets.pop(0)
        if isinstance(evidence, Exception):
            raise evidence
        return evidence


def initial_state(**updates):
    state = {
        "contract_id": "contract-1",
        "review_rule_text": "付款期限不得超过90日。",
        "review_items": [],
        "current_item_index": 0,
        "review_results": [],
        "summary": None,
    }
    state.update(updates)
    return state


def result_for(status, level=None):
    return ReviewResult.model_validate(
        {
            "item_id": f"item_{status}_{level}",
            "item_name": "审查项",
            "risk_status": status,
            "risk_level": level,
            "evidence_status": "insufficient" if status == "needs_review" else "found",
            "finding": "审查发现",
            "risk_description": "风险说明",
            "suggestion": "修改建议",
            "evidence": [] if status == "needs_review" else [EVIDENCE],
        }
    )


def test_parse_review_rules_node_validates_items_and_emits_serializable_progress():
    events = []
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(ITEMS_PAYLOAD),
        review_llm=FakeLLM(),
        contract_service=FakeContractService(),
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    update = nodes.parse_review_rules(initial_state())

    assert [item.id for item in update["review_items"]] == ["item_1", "item_2"]
    assert update["current_item_index"] == 0
    assert [event for event, _ in events] == ["review_items_parsed"]
    assert events[0][1]["review_items"][0]["id"] == "item_1"
    json.dumps(events[0][1], ensure_ascii=False)


def test_review_item_node_uses_current_query_and_preserves_rag_citations():
    events = []
    contract_service = FakeContractService([EVIDENCE])
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=FakeLLM(RISK_DECISION),
        contract_service=contract_service,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )
    item = ReviewItem.model_validate(ITEMS_PAYLOAD["review_items"][0])

    update = nodes.review_item(initial_state(review_items=[item]))

    assert contract_service.searches == [
        ("contract-1", "合同约定的付款期限是多久")
    ]
    assert update["current_item_index"] == 1
    assert len(update["review_results"]) == 1
    result = update["review_results"][0]
    assert result.item_id == "item_1"
    assert result.evidence[0].source_object_index == 27
    assert result.evidence[0].page_idx == 4
    assert result.evidence[0].node_type == "table"
    assert [event for event, _ in events] == [
        "review_item_started",
        "evidence_retrieved",
        "review_item_completed",
    ]
    for _, payload in events:
        json.dumps(payload, ensure_ascii=False)


def test_review_item_node_forces_needs_review_when_rag_evidence_is_empty():
    optimistic_decision = {
        **RISK_DECISION,
        "risk_status": "no_obvious_risk",
        "risk_level": None,
    }
    review_llm = FakeLLM(optimistic_decision)
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=review_llm,
        contract_service=FakeContractService([]),
    )
    item = ReviewItem.model_validate(ITEMS_PAYLOAD["review_items"][0])

    update = nodes.review_item(initial_state(review_items=[item]))

    result = update["review_results"][0]
    assert len(review_llm.prompts) == 1
    assert result.risk_status == "needs_review"
    assert result.risk_level is None
    assert result.evidence_status == "insufficient"
    assert result.evidence == []
    assert "合同没有约定" not in result.finding


def test_aggregate_results_counts_all_statuses_without_llm():
    events = []
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=FakeLLM(),
        contract_service=FakeContractService(),
        progress_callback=lambda event, payload: events.append((event, payload)),
    )
    results = [
        result_for("risk", "high"),
        result_for("risk", "low"),
        result_for("no_obvious_risk"),
        result_for("needs_review"),
    ]

    update = nodes.aggregate_results(initial_state(review_results=results))

    assert update["summary"].model_dump() == {
        "total_items": 4,
        "risk_count": 2,
        "high_risk_count": 1,
        "medium_risk_count": 0,
        "low_risk_count": 1,
        "no_obvious_risk_count": 1,
        "needs_review_count": 1,
    }
    assert events == [("review_summary", {"summary": update["summary"].model_dump()})]


def test_node_failures_include_stage_context_and_preserve_cause():
    parse_nodes = ContractReviewNodes(
        parse_llm=FakeLLM("not-json"),
        review_llm=FakeLLM(),
        contract_service=FakeContractService(),
    )
    with pytest.raises(RuntimeError, match="parse_review_rules failed") as parse_error:
        parse_nodes.parse_review_rules(initial_state())
    assert parse_error.value.__cause__ is not None

    item = ReviewItem.model_validate(ITEMS_PAYLOAD["review_items"][0])
    review_nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=FakeLLM(RISK_DECISION),
        contract_service=FakeContractService(RuntimeError("rerank unavailable")),
    )
    with pytest.raises(RuntimeError, match="review_item item_1 failed") as review_error:
        review_nodes.review_item(initial_state(review_items=[item]))
    assert str(review_error.value.__cause__) == "rerank unavailable"
