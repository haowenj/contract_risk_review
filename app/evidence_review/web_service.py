from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.evidence_review.repository import (
    EvidenceReviewRepository,
    EvidenceReviewDecisionRecord,
    EvidenceReviewRunRecord,
    EvidenceRunTransitionError,
)
from app.evidence_review.schemas import EvidencePackage, HumanDecision, RuleItem
from app.evidence_review.service import (
    EvidenceReviewService,
    build_fact_extraction_llm,
)
from app.service import ContractNotFoundError, ContractNotReadyError


LOGGER = logging.getLogger(__name__)
INTERRUPTED_EVIDENCE_RUN_MESSAGE = (
    "服务重启导致人工取证任务中断，请重新创建任务。"
)
PUBLIC_EVIDENCE_RUN_FAILED_MESSAGE = "人工取证任务执行失败，请稍后重试。"
EvidenceServiceFactory = Callable[..., EvidenceReviewService | Any]


class RuleSetNotFoundError(KeyError):
    pass


class RuleSetNotActiveError(RuntimeError):
    pass


def build_evidence_review_service(*, contract_service: Any) -> EvidenceReviewService:
    return EvidenceReviewService(
        contract_service=contract_service,
        fact_llm=build_fact_extraction_llm(),
    )


class EvidenceReviewWebService:
    def __init__(
        self,
        *,
        contract_service: Any,
        repository: EvidenceReviewRepository,
        evidence_service_factory: EvidenceServiceFactory = (
            build_evidence_review_service
        ),
    ):
        self.contract_service = contract_service
        self.repository = repository
        self.evidence_service_factory = evidence_service_factory

    def create_run(
        self,
        contract_id: str,
        rule_set_id: str,
    ) -> EvidenceReviewRunRecord:
        contract = self.contract_service.get_contract(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status != "ready":
            raise ContractNotReadyError(contract)

        rule_set = self.repository.get_rule_set(rule_set_id)
        if rule_set is None:
            raise RuleSetNotFoundError(rule_set_id)
        if rule_set.status != "active":
            raise RuleSetNotActiveError(rule_set_id)
        raw_items = rule_set.parsed_rules.get("review_items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("active rule set has no review items")
        for raw_item in raw_items:
            RuleItem.model_validate(raw_item)

        return self.repository.create_evidence_run(
            contract_id=contract_id,
            rule_set_id=rule_set_id,
            rule_snapshot=rule_set.parsed_rules,
        )

    def execute_run(self, run_id: str) -> EvidenceReviewRunRecord:
        run = self.repository.get_evidence_run(run_id)
        if run is None:
            raise KeyError(run_id)
        try:
            self.repository.mark_evidence_run_processing(run_id)
        except EvidenceRunTransitionError:
            current = self.repository.get_evidence_run(run_id)
            if current is None:
                raise KeyError(run_id)
            return current

        try:
            items = [
                RuleItem.model_validate(value)
                for value in run.rule_snapshot["review_items"]
            ]
            total = len(items)

            def progress_callback(event: str, payload: dict[str, Any]) -> None:
                if event != "evidence_item_completed":
                    return
                completed = int(payload.get("completed", 0))
                item_id = str(payload.get("rule_item_id", ""))
                self.repository.update_evidence_run_progress(
                    run_id,
                    {
                        "stage": "processing",
                        "message": f"已完成 {completed} / {total}",
                        "completed": completed,
                        "total": total,
                        "rule_item_id": item_id,
                    },
                )

            service = self.evidence_service_factory(
                contract_service=self.contract_service
            )
            packages = service.run(
                run.contract_id,
                items,
                progress_callback=progress_callback,
            )
            package_by_id = {
                value.rule_item_id: value for value in packages
            }
            if len(package_by_id) != len(items) or set(package_by_id) != {
                value.item_id for value in items
            }:
                raise ValueError("evidence service returned mismatched items")

            records = [
                {
                    "rule_item_id": item.item_id,
                    "rule_snapshot": item.model_dump(mode="json"),
                    "evidence_package": package_by_id[
                        item.item_id
                    ].model_dump(mode="json"),
                }
                for item in items
            ]
            return self.repository.complete_evidence_run(
                run_id,
                records,
                progress={
                    "stage": "completed",
                    "message": f"已完成 {total} / {total}",
                    "completed": total,
                    "total": total,
                },
            )
        except Exception as exc:
            LOGGER.error(
                "Evidence review run failed: %s (error_type=%s)",
                run_id,
                type(exc).__name__,
            )
            try:
                return self.repository.mark_evidence_run_failed(
                    run_id,
                    PUBLIC_EVIDENCE_RUN_FAILED_MESSAGE,
                )
            except EvidenceRunTransitionError:
                current = self.repository.get_evidence_run(run_id)
                if current is None:
                    raise KeyError(run_id) from exc
                return current

    def get_run_payload(
        self,
        contract_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        run = self.repository.get_evidence_run(run_id)
        if run is None or run.contract_id != contract_id:
            raise KeyError(run_id)
        items = self.repository.list_evidence_items(run_id)
        decisions = {
            value.rule_item_id: value
            for value in self.repository.list_evidence_decisions(run_id)
        }
        completed_decisions = [
            value.decision
            for value in decisions.values()
            if value.decision.human_review_status == "completed"
        ]

        def decision_payload(item_id: str) -> dict[str, Any]:
            record = decisions[item_id]
            payload = record.decision.model_dump(mode="json")
            # Keep the exact compare-and-swap token stored by SQLite. Pydantic
            # may otherwise render UTC as ``Z`` instead of ``+00:00``.
            payload["updated_at"] = record.updated_at
            return payload

        return {
            "run_id": run.run_id,
            "contract_id": run.contract_id,
            "rule_set_id": run.rule_set_id,
            "status": run.status,
            "human_status": run.human_status,
            "progress": run.progress,
            "items": [
                {
                    "rule_item_id": item.rule_item_id,
                    "rule_snapshot": RuleItem.model_validate(
                        item.rule_snapshot
                    ).model_dump(mode="json"),
                    "evidence_package": EvidencePackage.model_validate(
                        item.evidence_package
                    ).model_dump(mode="json"),
                    "human_decision": decision_payload(item.rule_item_id),
                }
                for item in items
            ],
            "human_summary": {
                "pending_count": len(items) - len(completed_decisions),
                "completed_count": len(completed_decisions),
                "risk_count": sum(
                    value.decision == "risk"
                    for value in completed_decisions
                ),
                "no_obvious_risk_count": sum(
                    value.decision == "no_obvious_risk"
                    for value in completed_decisions
                ),
                "cannot_determine_count": sum(
                    value.decision == "cannot_determine"
                    for value in completed_decisions
                ),
            },
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "error_message": run.error_message,
        }

    def save_human_decision(
        self,
        run_id: str,
        item_id: str,
        decision: HumanDecision | dict[str, Any],
        expected_updated_at: str,
    ) -> EvidenceReviewDecisionRecord:
        raw = (
            decision.model_dump(mode="json")
            if isinstance(decision, HumanDecision)
            else dict(decision)
        )
        raw["updated_at"] = datetime.now(UTC)
        validated = HumanDecision.model_validate(raw)
        return self.repository.save_evidence_decision(
            run_id,
            item_id,
            validated,
            expected_updated_at,
        )

    def recover_interrupted_runs(self) -> int:
        return self.repository.recover_incomplete_evidence_runs(
            INTERRUPTED_EVIDENCE_RUN_MESSAGE
        )
