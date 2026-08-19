from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.contract_review.graph import build_contract_review_graph
from app.contract_review.nodes import ContractReviewNodes, ProgressCallback
from app.contract_review.schemas import ReviewItemList, RiskDecision
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
        model=os.getenv("LLM_MODEL", "qwen3.7-plus"),
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0,
        timeout=REVIEW_LLM_TIMEOUT_SECONDS,
        max_retries=0,
        extra_body={"enable_thinking": False},
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

    def run(
        self,
        contract_id: str,
        review_rule_text: str,
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

        final_state = self.graph.invoke(
            {
                "contract_id": contract_id,
                "review_rule_text": review_rule_text,
                "review_items": [],
                "current_item_index": 0,
                "review_results": [],
                "summary": None,
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
