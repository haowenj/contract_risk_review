import json
from types import SimpleNamespace

import pytest

from app.contract_review.graph import build_contract_review_graph
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

SECOND_EVIDENCE = {
    "source_object_index": 99,
    "page_idx": 12,
    "node_type": "text",
    "text": "乙方不得将合同义务转委托给第三方。",
    "evidence_text": "乙方不得将合同义务转委托给第三方。",
}

QUERY_REWRITE = {
    "retrieval_query": "乙方权利义务、转委托、第三方履约及委托其他单位实施的约定",
    "reason": "扩展分包和转包的近义表达与相关章节名称",
    "keywords": ["分包", "转包", "转委托", "委托第三方"],
}

RERANK_TOP3_DEBUG = [
    {"source_object_index": 51, "text": "（二）乙方权利与义务"},
    {"source_object_index": 47, "text": "（一）甲方权利与义务"},
    {"source_object_index": 46, "text": "第五条 双方权利与义务"},
]

INSUFFICIENT_DECISION = {
    "risk_status": "needs_review",
    "risk_level": None,
    "evidence_status": "insufficient",
    "finding": "现有证据只有章节标题。",
    "risk_description": "证据不足以判断是否允许分包。",
    "suggestion": "继续检索乙方权利义务及第三方履约约定。",
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

    def search_contract(self, contract_id, query, *, debug_callback=None):
        self.searches.append((contract_id, query))
        evidence = self.evidence_sets.pop(0)
        if isinstance(evidence, Exception):
            raise evidence
        if not evidence and debug_callback is not None:
            debug_callback(RERANK_TOP3_DEBUG)
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
    query_rewrite_llm = FakeLLM()
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=FakeLLM(RISK_DECISION),
        query_rewrite_llm=query_rewrite_llm,
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
    assert query_rewrite_llm.prompts == []
    assert [event for event, _ in events] == [
        "review_item_started",
        "evidence_retrieved",
        "review_item_completed",
    ]
    for _, payload in events:
        json.dumps(payload, ensure_ascii=False)


def test_review_item_node_rewrites_query_after_empty_evidence_without_first_review_call():
    events = []
    review_llm = FakeLLM(RISK_DECISION)
    query_rewrite_llm = FakeLLM(QUERY_REWRITE)
    contract_service = FakeContractService([], [SECOND_EVIDENCE])
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=review_llm,
        query_rewrite_llm=query_rewrite_llm,
        contract_service=contract_service,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )
    item = ReviewItem.model_validate(ITEMS_PAYLOAD["review_items"][0])

    update = nodes.review_item(initial_state(review_items=[item]))

    result = update["review_results"][0]
    assert contract_service.searches == [
        ("contract-1", "合同约定的付款期限是多久"),
        ("contract-1", QUERY_REWRITE["retrieval_query"]),
    ]
    assert len(query_rewrite_llm.prompts) == 1
    assert len(review_llm.prompts) == 1
    assert result.risk_status == "risk"
    assert [value.source_object_index for value in result.evidence] == [99]
    assert [event for event, _ in events] == [
        "review_item_started",
        "evidence_retrieved",
        "empty_evidence_rerank_debug",
        "retrieval_query_rewritten",
        "evidence_retrieved",
        "review_item_completed",
    ]
    assert [
        payload["attempt"]
        for event, payload in events
        if event == "evidence_retrieved"
    ] == [1, 2]
    assert events[2][1] == {
        "item_id": "item_1",
        "attempt": 1,
        "retrieval_query": "合同约定的付款期限是多久",
        "rerank_top3": RERANK_TOP3_DEBUG,
    }
    assert events[3][1]["retrieval_query"] == QUERY_REWRITE["retrieval_query"]


def test_review_item_node_retries_insufficient_decision_and_merges_unique_evidence():
    review_llm = FakeLLM(INSUFFICIENT_DECISION, RISK_DECISION)
    query_rewrite_llm = FakeLLM(QUERY_REWRITE)
    contract_service = FakeContractService(
        [EVIDENCE],
        [EVIDENCE, SECOND_EVIDENCE],
    )
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=review_llm,
        query_rewrite_llm=query_rewrite_llm,
        contract_service=contract_service,
    )
    item = ReviewItem.model_validate(ITEMS_PAYLOAD["review_items"][0])

    update = nodes.review_item(initial_state(review_items=[item]))

    result = update["review_results"][0]
    assert len(review_llm.prompts) == 2
    assert len(query_rewrite_llm.prompts) == 1
    assert len(contract_service.searches) == 2
    assert [value.source_object_index for value in result.evidence] == [27, 99]


def test_review_item_node_stops_after_second_empty_retrieval_without_review_llm():
    review_llm = FakeLLM()
    query_rewrite_llm = FakeLLM(QUERY_REWRITE)
    contract_service = FakeContractService([], [])
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(),
        review_llm=review_llm,
        query_rewrite_llm=query_rewrite_llm,
        contract_service=contract_service,
    )
    item = ReviewItem.model_validate(ITEMS_PAYLOAD["review_items"][0])

    update = nodes.review_item(initial_state(review_items=[item]))

    result = update["review_results"][0]
    assert len(contract_service.searches) == 2
    assert len(query_rewrite_llm.prompts) == 1
    assert review_llm.prompts == []
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


def test_real_graph_loops_sequentially_and_accumulates_results():
    second_decision = {
        **RISK_DECISION,
        "risk_status": "no_obvious_risk",
        "risk_level": None,
        "finding": "延期履约责任约定明确。",
        "risk_description": "未发现明显风险。",
        "suggestion": "保持现有条款。",
    }
    second_evidence = {
        **EVIDENCE,
        "source_object_index": 99,
        "page_idx": 12,
        "node_type": "text",
        "text": "延期履约按日承担违约金。",
        "evidence_text": "延期履约按日承担违约金。",
    }
    contract_service = FakeContractService([EVIDENCE], [second_evidence])
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(ITEMS_PAYLOAD),
        review_llm=FakeLLM(RISK_DECISION, second_decision),
        contract_service=contract_service,
    )
    graph = build_contract_review_graph(nodes)

    final_state = graph.invoke(initial_state())

    assert [result.item_id for result in final_state["review_results"]] == [
        "item_1",
        "item_2",
    ]
    assert final_state["current_item_index"] == 2
    assert final_state["summary"].total_items == 2
    assert contract_service.searches == [
        ("contract-1", "合同约定的付款期限是多久"),
        ("contract-1", "延期履约需要承担什么违约责任"),
    ]
