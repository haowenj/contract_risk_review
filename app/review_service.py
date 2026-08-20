from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.contract_review.service import build_contract_review_service
from app.review_db import ContractReviewRepository, ReviewRunTransitionError
from app.review_logging import ReviewRunJournal
from app.review_models import ContractReviewRun
from app.service import ContractNotFoundError, ContractNotReadyError

logger = logging.getLogger(__name__)
INTERRUPTED_REVIEW_RUN_MESSAGE = "服务重启导致风险审查任务中断"
PUBLIC_REVIEW_RUN_FAILED_MESSAGE = "风险审查执行失败，请查看服务日志。"

ReviewServiceFactory = Callable[..., Any]


def _business_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    if not result:
        return {}

    summary = result.get("summary")
    safe_summary = None
    if isinstance(summary, dict):
        summary_fields = {
            "total_items",
            "risk_count",
            "high_risk_count",
            "medium_risk_count",
            "low_risk_count",
            "no_obvious_risk_count",
            "needs_review_count",
        }
        safe_summary = {
            key: value for key, value in summary.items() if key in summary_fields
        }

    safe_results = []
    result_fields = {
        "item_id",
        "item_name",
        "risk_status",
        "risk_level",
        "evidence_status",
        "finding",
        "risk_description",
        "suggestion",
    }
    evidence_fields = {
        "page_idx",
        "source_object_index",
        "node_type",
        "evidence_text",
    }
    absence_fields = {
        "primary_keywords",
        "secondary_keywords",
        "candidate_count",
    }
    raw_results = result.get("review_results", [])
    if isinstance(raw_results, list):
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            safe_result = {
                key: value
                for key, value in raw_result.items()
                if key in result_fields
            }
            raw_evidence = raw_result.get("evidence", [])
            safe_result["evidence"] = [
                {
                    key: value
                    for key, value in evidence.items()
                    if key in evidence_fields
                }
                for evidence in raw_evidence
                if isinstance(evidence, dict)
            ] if isinstance(raw_evidence, list) else []
            raw_absence = raw_result.get("absence_check")
            safe_result["absence_check"] = (
                {
                    key: value
                    for key, value in raw_absence.items()
                    if key in absence_fields
                }
                if isinstance(raw_absence, dict)
                else None
            )
            safe_results.append(safe_result)

    return {
        "summary": safe_summary,
        "review_results": safe_results,
    }


class _BusinessProgressTracker:
    def __init__(
        self,
        repository: ContractReviewRepository,
        run_id: str,
    ):
        self.repository = repository
        self.run_id = run_id
        self.total = 0
        self.current = 0
        self.completed = 0
        self.item_name = ""

    def _save(self, stage: str, message: str) -> None:
        progress: dict[str, Any] = {"stage": stage, "message": message}
        if self.total:
            progress["total"] = self.total
        if self.current:
            progress["current"] = self.current
        if self.item_name:
            progress["item_name"] = self.item_name
        self.repository.update_progress(self.run_id, progress)

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if event == "review_items_parsed":
            review_items = payload.get("review_items", [])
            self.total = len(review_items) if isinstance(review_items, list) else 0
            self._save("rules_parsed", f"已解析 {self.total} 个审查项")
            return

        if event == "review_item_started":
            raw_index = payload.get("current_item_index", 0)
            self.current = int(raw_index) + 1
            item = payload.get("item", {})
            self.item_name = str(item.get("name", "")) if isinstance(item, dict) else ""
            self._save(
                "reviewing",
                f"正在审查 {self.current} / {self.total}：{self.item_name}",
            )
            return

        if event == "retrieval_query_rewritten":
            self._save("second_retrieval", "正在进行第二次证据检索")
            return

        if event == "absence_check_started":
            self._save("absence_check", "正在执行全文缺失核验")
            return

        if event == "review_item_completed":
            self.completed += 1
            self._save(
                "item_completed",
                f"已完成 {self.completed} / {self.total}",
            )


class ContractReviewWebService:
    def __init__(
        self,
        *,
        contract_service: Any,
        review_repository: ContractReviewRepository,
        review_service_factory: ReviewServiceFactory = build_contract_review_service,
        review_runs_dir: Path | None = None,
    ):
        self.contract_service = contract_service
        self.review_repository = review_repository
        self.review_service_factory = review_service_factory
        self.review_runs_dir = review_runs_dir

    def create_run(
        self,
        contract_id: str,
        review_rule_text: str,
    ) -> ContractReviewRun:
        normalized_text = review_rule_text.strip()
        if not normalized_text:
            raise ValueError("review rule text must not be empty")

        contract = self.contract_service.get_contract(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status != "ready":
            raise ContractNotReadyError(contract)
        run = self.review_repository.create_run(contract_id, normalized_text)
        if self.review_runs_dir is not None:
            ReviewRunJournal(self.review_runs_dir, run.run_id).created(
                contract_id=run.contract_id,
                review_rule_text=run.review_rule_text,
                created_at=run.created_at,
            )
        return run

    def execute_run(self, run_id: str) -> ContractReviewRun:
        run = self.review_repository.get_run(run_id)
        if run is None:
            raise KeyError(run_id)

        try:
            self.review_repository.mark_processing(
                run_id,
                {"stage": "parsing_rules", "message": "正在解析审查规范"},
            )
        except ReviewRunTransitionError:
            current = self.review_repository.get_run(run_id)
            if current is None:
                raise KeyError(run_id)
            return current

        journal: ReviewRunJournal | None = None
        try:
            tracker = _BusinessProgressTracker(self.review_repository, run_id)
            journal = (
                ReviewRunJournal(self.review_runs_dir, run_id)
                if self.review_runs_dir is not None
                else None
            )
            if journal is not None:
                journal.started(run.contract_id)

            def progress_callback(event: str, payload: dict[str, Any]) -> None:
                if journal is not None:
                    journal.record(event, payload)
                tracker(event, payload)

            review_service = self.review_service_factory(
                contract_service=self.contract_service,
                progress_callback=progress_callback,
            )
            result = review_service.run(run.contract_id, run.review_rule_text)
            summary = result.get("summary") or {}
            total = int(summary.get("total_items", tracker.total))
            completed_run = self.review_repository.mark_ready(
                run_id,
                result,
                {
                    "stage": "completed",
                    "message": f"已完成 {total} / {total}",
                    "current": total,
                    "total": total,
                },
            )
            if journal is not None:
                journal.completed(result)
            return completed_run
        except Exception as exc:
            if journal is not None:
                journal.failed(exc)
            logger.exception("contract review run failed: %s", run_id)
            try:
                return self.review_repository.mark_failed(run_id, str(exc))
            except ReviewRunTransitionError:
                current = self.review_repository.get_run(run_id)
                if current is None:
                    raise KeyError(run_id) from exc
                return current

    def get_run_payload(
        self,
        contract_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        run = self.review_repository.get_run(run_id)
        if run is None or run.contract_id != contract_id:
            raise KeyError(run_id)
        return {
            "run_id": run.run_id,
            "contract_id": run.contract_id,
            "status": run.status,
            "result": _business_result_payload(run.result),
            "progress": run.progress,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "error_message": (
                PUBLIC_REVIEW_RUN_FAILED_MESSAGE
                if run.status == "failed"
                else None
            ),
        }

    def recover_interrupted_runs(self) -> int:
        return self.review_repository.recover_incomplete_runs(
            INTERRUPTED_REVIEW_RUN_MESSAGE
        )
