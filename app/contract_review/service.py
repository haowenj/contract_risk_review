from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.contract_review.graph import build_contract_review_graph
from app.contract_review.nodes import ContractReviewNodes, ProgressCallback
from app.contract_review.schemas import (
    ReviewItemList,
    RiskDecision,
)
from app.llm_gateway import llm_backend_kwargs
from app.service import ContractNotFoundError, ContractNotReadyError

REVIEW_LLM_TIMEOUT_SECONDS = 120.0


def _build_structured_llm(
    model_type: type[BaseModel],
    *,
    schema_name: str,
) -> Any:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": model_type.model_json_schema(),
        },
    }
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0,
        timeout=REVIEW_LLM_TIMEOUT_SECONDS,
        max_retries=0,
        reasoning_effort="none",
        **llm_backend_kwargs(),
    ).bind(response_format=response_format)


class ContractReviewService:
    def __init__(
        self,
        *,
        contract_service: Any,
        parse_llm: Any,
        review_llm: Any,
        progress_callback: ProgressCallback | None = None,
    ):
        self.contract_service = contract_service
        self.nodes = ContractReviewNodes(
            parse_llm=parse_llm,
            review_llm=review_llm,
            contract_service=contract_service,
            progress_callback=progress_callback,
        )
        self.graph = build_contract_review_graph(self.nodes)
        self.preparsed_graph = build_contract_review_graph(
            self.nodes, skip_rule_parse=True
        )

    def parse_rules(self, review_rule_text: str) -> list[dict[str, Any]]:
        text = review_rule_text.strip()
        if not text:
            raise ValueError("review_rule_text must not be empty")
        parsed = self.nodes.parse_review_rules({"review_rule_text": text})
        return [item.model_dump(mode="json") for item in parsed["review_items"]]

    def run_preparsed(
        self,
        contract_id: str,
        review_rule_text: str,
        review_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        items = ReviewItemList.model_validate(
            {"review_items": review_items}
        ).review_items
        return self._run(contract_id, review_rule_text, items=items)

    def run(
        self,
        contract_id: str,
        review_rule_text: str,
    ) -> dict[str, Any]:
        return self._run(contract_id, review_rule_text)

    def _run(
        self,
        contract_id: str,
        review_rule_text: str,
        *,
        items: list[Any] | None = None,
    ) -> dict[str, Any]:
        contract_id = contract_id.strip()
        review_rule_text = review_rule_text.strip()
        if not contract_id:
            raise ValueError("contract_id must not be empty")
        if not review_rule_text:
            raise ValueError("review_rule_text must not be empty")

        contract = self.contract_service.get_contract(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status != "ready":
            raise ContractNotReadyError(contract)

        final_state = (
            self.preparsed_graph if items is not None else self.graph
        ).invoke(
            {
                "contract_id": contract_id,
                "review_rule_text": review_rule_text,
                "review_items": items if items is not None else [],
                "current_item_index": 0,
                "review_results": [],
                "summary": None,
                "retrieval_attempt": 0,
                "current_retrieval_query": "",
                "retrieved_evidence": [],
                "absence_primary_keywords": [],
                "absence_secondary_keywords": [],
                "absence_candidates": [],
                "absence_candidate_count": None,
                "current_decision": None,
            }
        )
        summary = final_state["summary"]
        return {
            "contract_id": final_state["contract_id"],
            "review_rule_text": final_state["review_rule_text"],
            "review_items": [
                item.model_dump(mode="json")
                for item in final_state["review_items"]
            ],
            "current_item_index": final_state["current_item_index"],
            "review_results": [
                result.model_dump(mode="json")
                for result in final_state["review_results"]
            ],
            "summary": summary.model_dump(mode="json") if summary else None,
        }


def build_default_contract_review_service(
    *,
    progress_callback: ProgressCallback | None = None,
) -> ContractReviewService:
    from app.api import build_default_service
    from app.config import load_settings

    contract_service = build_default_service(load_settings())
    return build_contract_review_service(
        contract_service=contract_service,
        progress_callback=progress_callback,
    )


def build_contract_review_service(
    *,
    contract_service: Any,
    progress_callback: ProgressCallback | None = None,
) -> ContractReviewService:
    return ContractReviewService(
        contract_service=contract_service,
        parse_llm=_build_structured_llm(
            ReviewItemList,
            schema_name="contract_review_items",
        ),
        review_llm=_build_structured_llm(
            RiskDecision,
            schema_name="contract_review_result",
        ),
        progress_callback=progress_callback,
    )
