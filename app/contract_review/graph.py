from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.contract_review.nodes import ContractReviewNodes
from app.contract_review.state import ContractReviewState


def route_after_review(state: ContractReviewState) -> str:
    if state["current_item_index"] < len(state["review_items"]):
        return "review_item"
    return "aggregate_results"


def build_contract_review_graph(nodes: ContractReviewNodes) -> Any:
    builder = StateGraph(ContractReviewState)
    builder.add_node("parse_review_rules", nodes.parse_review_rules)
    builder.add_node("review_item", nodes.review_item)
    builder.add_node("aggregate_results", nodes.aggregate_results)
    builder.add_edge(START, "parse_review_rules")
    builder.add_edge("parse_review_rules", "review_item")
    builder.add_conditional_edges(
        "review_item",
        route_after_review,
        {
            "review_item": "review_item",
            "aggregate_results": "aggregate_results",
        },
    )
    builder.add_edge("aggregate_results", END)
    return builder.compile()
