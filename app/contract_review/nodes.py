from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.contract_review.absence import scan_source_objects
from app.contract_review.prompts import (
    build_absence_result_prompt,
    build_parse_review_rules_prompt,
    build_retrieval_query_rewrite_prompt,
    build_review_item_prompt,
)
from app.contract_review.schemas import (
    AbsenceCheckMetadata,
    Evidence,
    ReviewItemList,
    ReviewResult,
    ReviewSummary,
    RetrievalQueryRewrite,
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
        query_rewrite_llm: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.parse_llm = parse_llm
        self.review_llm = review_llm
        self.query_rewrite_llm = query_rewrite_llm
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

    def prepare_review_item(
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
        except Exception as exc:
            raise RuntimeError(f"prepare_review_item {item.id} failed") from exc

        return {
            "retrieval_attempt": 1,
            "current_retrieval_query": item.retrieval_query,
            "retrieved_evidence": [],
            "absence_keywords": [],
            "absence_candidates": [],
            "absence_candidate_count": None,
            "current_decision": None,
        }

    def retrieve_evidence(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            rerank_top3_debug: list[dict[str, Any]] = []
            raw_evidence = self.contract_service.search_contract(
                state["contract_id"],
                state["current_retrieval_query"],
                debug_callback=rerank_top3_debug.extend,
            )
            current_evidence = [
                Evidence.model_validate(value) for value in raw_evidence
            ]
            merged_evidence = list(state["retrieved_evidence"])
            existing_indices = {
                value.source_object_index for value in merged_evidence
            }
            for value in current_evidence:
                if value.source_object_index not in existing_indices:
                    merged_evidence.append(value)
                    existing_indices.add(value.source_object_index)

            self._emit(
                "evidence_retrieved",
                {
                    "item_id": item.id,
                    "attempt": state["retrieval_attempt"],
                    "retrieval_query": state["current_retrieval_query"],
                    "evidence": [
                        value.model_dump(mode="json")
                        for value in current_evidence
                    ],
                },
            )
            if not current_evidence:
                self._emit(
                    "empty_evidence_rerank_debug",
                    {
                        "item_id": item.id,
                        "attempt": state["retrieval_attempt"],
                        "retrieval_query": state["current_retrieval_query"],
                        "rerank_top3": rerank_top3_debug,
                    },
                )
        except Exception as exc:
            raise RuntimeError(f"retrieve_evidence {item.id} failed") from exc
        return {"retrieved_evidence": merged_evidence}

    def rewrite_query(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            if self.query_rewrite_llm is None:
                raise ValueError("query_rewrite_llm is required for retry")
            previous_query = state["current_retrieval_query"]
            response = self.query_rewrite_llm.invoke(
                build_retrieval_query_rewrite_prompt(
                    item,
                    attempted_queries=[previous_query],
                    evidence=state["retrieved_evidence"],
                    decision=state["current_decision"],
                )
            )
            rewrite = parse_llm_response(response, RetrievalQueryRewrite)
            if rewrite.retrieval_query == previous_query:
                raise ValueError(
                    "rewritten retrieval_query must differ from attempted query"
                )
            self._emit(
                "retrieval_query_rewritten",
                {
                    "item_id": item.id,
                    "next_attempt": 2,
                    "previous_query": previous_query,
                    "retrieval_query": rewrite.retrieval_query,
                    "reason": rewrite.reason,
                },
            )
        except Exception as exc:
            raise RuntimeError(f"rewrite_query {item.id} failed") from exc
        return {
            "retrieval_attempt": 2,
            "current_retrieval_query": rewrite.retrieval_query,
            "absence_keywords": rewrite.keywords,
        }

    def risk_decision(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            evidence = state["absence_candidates"] or state["retrieved_evidence"]
            response = self.review_llm.invoke(
                build_review_item_prompt(item, evidence)
            )
            decision = parse_llm_response(response, RiskDecision)
        except Exception as exc:
            raise RuntimeError(f"risk_decision {item.id} failed") from exc
        return {"current_decision": decision}

    def insufficient_result(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            decision = RiskDecision(
                risk_status="needs_review",
                risk_level=None,
                evidence_status="insufficient",
                finding="两次检索均未获得足以支持判断的合同证据。",
                risk_description="证据不足，无法可靠判断该审查项是否存在风险。",
                suggestion="请人工核对合同全文及相关附件后再作判断。",
            )
        except Exception as exc:
            raise RuntimeError(f"insufficient_result {item.id} failed") from exc
        return {"current_decision": decision}

    def absence_check(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            self._emit(
                "absence_check_started",
                {
                    "item_id": item.id,
                    "retrieval_attempt": state["retrieval_attempt"],
                },
            )
            self._emit(
                "absence_keywords_generated",
                {
                    "item_id": item.id,
                    "keywords": state["absence_keywords"],
                },
            )
            source_objects = self.contract_service.load_contract_content_objects(
                state["contract_id"]
            )
            scan = scan_source_objects(
                source_objects,
                state["absence_keywords"],
            )
            candidates = [
                Evidence.model_validate(value) for value in scan.candidates
            ]
            self._emit(
                "absence_candidates_found",
                {
                    "item_id": item.id,
                    "candidate_count": scan.candidate_count,
                    "candidates": [
                        value.model_dump(mode="json") for value in candidates
                    ],
                },
            )
        except Exception as exc:
            raise RuntimeError(f"absence_check {item.id} failed") from exc
        return {
            "absence_candidates": candidates,
            "absence_candidate_count": scan.candidate_count,
        }

    def absence_result(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            if state["absence_candidate_count"] != 0:
                raise ValueError("absence_result requires zero scan candidates")
            response = self.review_llm.invoke(
                build_absence_result_prompt(
                    item,
                    keywords=state["absence_keywords"],
                )
            )
            decision = parse_llm_response(response, RiskDecision)
            if decision.evidence_status != "absence_verified":
                raise ValueError("absence_result must return absence_verified")
            forbidden = "合同肯定没有"
            if any(
                forbidden in value
                for value in (
                    decision.finding,
                    decision.risk_description,
                    decision.suggestion,
                )
            ):
                raise ValueError("absence_result used an absolute absence claim")
            if "基于当前合同全文解析结果" not in decision.finding:
                raise ValueError(
                    "absence_result finding lacks parsed-content scope"
                )
            self._emit(
                "absence_confirmed",
                {
                    "item_id": item.id,
                    "keywords": state["absence_keywords"],
                    "candidate_count": 0,
                    "decision": decision.model_dump(mode="json"),
                },
            )
        except Exception as exc:
            raise RuntimeError(f"absence_result {item.id} failed") from exc
        return {"current_decision": decision}

    def finalize_review_item(
        self,
        state: ContractReviewState,
    ) -> dict[str, Any]:
        item = state["review_items"][state["current_item_index"]]
        try:
            decision = state["current_decision"]
            if decision is None:
                raise ValueError("current_decision is required")
            evidence = state["absence_candidates"] or state["retrieved_evidence"]
            absence_check = None
            if state["absence_candidate_count"] is not None:
                absence_check = AbsenceCheckMetadata(
                    keywords=state["absence_keywords"],
                    candidate_count=state["absence_candidate_count"],
                )
            result = ReviewResult(
                item_id=item.id,
                item_name=item.name,
                evidence=evidence,
                absence_check=absence_check,
                **decision.model_dump(),
            )
            self._emit(
                "review_item_completed",
                {"result": result.model_dump(mode="json")},
            )
        except Exception as exc:
            raise RuntimeError(f"finalize_review_item {item.id} failed") from exc
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
