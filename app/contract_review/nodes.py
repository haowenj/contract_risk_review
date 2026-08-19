from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.contract_review.prompts import (
    build_parse_review_rules_prompt,
    build_review_item_prompt,
)
from app.contract_review.schemas import (
    Evidence,
    ReviewItemList,
    ReviewResult,
    ReviewSummary,
    RiskDecision,
    parse_llm_response,
)
from app.contract_review.state import ContractReviewState


type ProgressCallback = Callable[[str, dict[str, Any]], None]


class ContractReviewNodes:
    def __init__(
        self,
        *,
        parse_llm: Any,
        review_llm: Any,
        contract_service: Any,
        progress_callback: ProgressCallback | None = None,
    ):
        self.parse_llm = parse_llm
        self.review_llm = review_llm
        self.contract_service = contract_service
        self.progress_callback = progress_callback

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event, payload)

    def parse_review_rules(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        try:
            response = self.parse_llm.invoke(
                build_parse_review_rules_prompt(state["review_rule_text"])
            )
            parsed = parse_llm_response(response, ReviewItemList)
        except Exception as exc:
            raise RuntimeError("parse_review_rules failed") from exc

        items_payload = [
            item.model_dump(mode="json") for item in parsed.review_items
        ]
        self._emit(
            "review_items_parsed",
            {"review_items": items_payload},
        )
        return {
            "review_items": parsed.review_items,
            "current_item_index": 0,
        }

    def review_item(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            self._emit(
                "review_item_started",
                {
                    "current_item_index": state["current_item_index"],
                    "item": item.model_dump(mode="json"),
                },
            )
            raw_evidence = self.contract_service.search_contract(
                state["contract_id"],
                item.retrieval_query,
            )
            evidence = [Evidence.model_validate(value) for value in raw_evidence]
            self._emit(
                "evidence_retrieved",
                {
                    "item_id": item.id,
                    "retrieval_query": item.retrieval_query,
                    "evidence": [
                        value.model_dump(mode="json") for value in evidence
                    ],
                },
            )
            response = self.review_llm.invoke(
                build_review_item_prompt(item, evidence)
            )
            decision = parse_llm_response(response, RiskDecision)
            if not evidence:
                decision = RiskDecision(
                    risk_status="needs_review",
                    risk_level=None,
                    evidence_status="insufficient",
                    finding="当前检索未获得足以支持判断的合同证据。",
                    risk_description="证据不足，无法可靠判断该审查项是否存在风险。",
                    suggestion="请人工核对合同全文及相关附件后再作判断。",
                )
            result = ReviewResult(
                item_id=item.id,
                item_name=item.name,
                evidence=evidence,
                **decision.model_dump(),
            )
            self._emit(
                "review_item_completed",
                {"result": result.model_dump(mode="json")},
            )
        except Exception as exc:
            raise RuntimeError(f"review_item {item.id} failed") from exc

        return {
            "review_results": [result],
            "current_item_index": state["current_item_index"] + 1,
        }

    def aggregate_results(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        results = state["review_results"]
        summary = ReviewSummary(
            total_items=len(results),
            risk_count=sum(result.risk_status == "risk" for result in results),
            high_risk_count=sum(
                result.risk_status == "risk" and result.risk_level == "high"
                for result in results
            ),
            medium_risk_count=sum(
                result.risk_status == "risk" and result.risk_level == "medium"
                for result in results
            ),
            low_risk_count=sum(
                result.risk_status == "risk" and result.risk_level == "low"
                for result in results
            ),
            no_obvious_risk_count=sum(
                result.risk_status == "no_obvious_risk" for result in results
            ),
            needs_review_count=sum(
                result.risk_status == "needs_review" for result in results
            ),
        )
        self._emit(
            "review_summary",
            {"summary": summary.model_dump(mode="json")},
        )
        return {"summary": summary}
