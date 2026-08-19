from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.contract_review.nodes import ContractReviewNodes
from app.contract_review.state import ContractReviewState


def route_after_retrieve(state: ContractReviewState) -> str:
    if state["retrieved_evidence"]:
        return "risk_decision"
    if state["retrieval_attempt"] < 2:
        return "rewrite_query"
    return "absence_check"


def route_after_absence_check(state: ContractReviewState) -> str:
    if state["absence_candidates"]:
        return "risk_decision"
    return "insufficient_result"


def route_after_risk_decision(state: ContractReviewState) -> str:
    decision = state["current_decision"]
    if (
        decision is not None
        and decision.evidence_status == "insufficient"
        and state["retrieval_attempt"] < 2
    ):
        return "rewrite_query"
    return "finalize_review_item"


def route_after_finalize(state: ContractReviewState) -> str:
    if state["current_item_index"] < len(state["review_items"]):
        return "prepare_review_item"
    return "aggregate_results"


def build_contract_review_graph(nodes: ContractReviewNodes) -> Any:
    builder = StateGraph(ContractReviewState)
    builder.add_node("parse_review_rules", nodes.parse_review_rules)
    builder.add_node("prepare_review_item", nodes.prepare_review_item)
    builder.add_node("retrieve_evidence", nodes.retrieve_evidence)
    builder.add_node("rewrite_query", nodes.rewrite_query)
    builder.add_node("risk_decision", nodes.risk_decision)
    builder.add_node("absence_check", nodes.absence_check)
    builder.add_node("insufficient_result", nodes.insufficient_result)
    builder.add_node("finalize_review_item", nodes.finalize_review_item)
    builder.add_node("aggregate_results", nodes.aggregate_results)
    builder.add_edge(START, "parse_review_rules")
    builder.add_edge("parse_review_rules", "prepare_review_item")
    builder.add_edge("prepare_review_item", "retrieve_evidence")
    builder.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieve,
        {
            "risk_decision": "risk_decision",
            "rewrite_query": "rewrite_query",
            "absence_check": "absence_check",
        },
    )
    builder.add_edge("rewrite_query", "retrieve_evidence")
    builder.add_conditional_edges(
        "absence_check",
        route_after_absence_check,
        {
            "risk_decision": "risk_decision",
            "insufficient_result": "insufficient_result",
        },
    )
    builder.add_conditional_edges(
        "risk_decision",
        route_after_risk_decision,
        {
            "rewrite_query": "rewrite_query",
            "finalize_review_item": "finalize_review_item",
        },
    )
    builder.add_edge("insufficient_result", "finalize_review_item")
    builder.add_conditional_edges(
        "finalize_review_item",
        route_after_finalize,
        {
            "prepare_review_item": "prepare_review_item",
            "aggregate_results": "aggregate_results",
        },
    )
    builder.add_edge("aggregate_results", END)
    return builder.compile()
