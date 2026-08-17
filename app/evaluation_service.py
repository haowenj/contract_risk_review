from __future__ import annotations

import logging
from typing import Any

from app.db import ContractRepository
from app.evaluation_db import EvaluationRepository
from app.evaluation_metrics import (
    build_config_snapshot,
    build_evaluation_result,
    serialize_pipeline_result,
)
from app.evaluation_models import EvaluationCase, EvaluationRun
from app.index_manager import IndexManager
from app.models import ContractRecord
from app.rag_pipeline import RAGPipeline
import retrieval_evaluation


logger = logging.getLogger(__name__)
INTERRUPTED_RUN_MESSAGE = "服务重启导致评测任务中断"


class EvaluationContractNotFoundError(LookupError):
    pass


class EvaluationContractNotReadyError(RuntimeError):
    def __init__(self, record: ContractRecord):
        self.record = record
        super().__init__(f"contract {record.contract_id} is {record.status}")


class EvaluationCaseNotFoundError(LookupError):
    pass


class EvaluationStaleError(RuntimeError):
    pass


class EvaluationService:
    def __init__(
        self,
        contract_repository: ContractRepository,
        evaluation_repository: EvaluationRepository,
        index_manager: IndexManager,
        pipeline: RAGPipeline | Any | None = None,
    ):
        self.contract_repository = contract_repository
        self.evaluation_repository = evaluation_repository
        self.index_manager = index_manager
        self.pipeline = pipeline or RAGPipeline()

    def _contract(self, contract_id: str) -> ContractRecord:
        contract = self.contract_repository.get(contract_id)
        if contract is None:
            raise EvaluationContractNotFoundError(contract_id)
        return contract

    def _ready_contract(self, contract_id: str) -> ContractRecord:
        contract = self._contract(contract_id)
        if contract.status != "ready" or not contract.index_version:
            raise EvaluationContractNotReadyError(contract)
        return contract

    @staticmethod
    def default_cases() -> list[tuple[str, list[int]]]:
        return [
            (
                item["query"],
                list(item["expected_source_object_indices"]),
            )
            for item in retrieval_evaluation.EVALUATION_QUERIES
        ]

    def list_cases(self, contract_id: str) -> list[EvaluationCase]:
        return self.evaluation_repository.list_cases(contract_id)

    def save_cases(
        self,
        contract_id: str,
        entries: list[tuple[str, list[int]]],
    ) -> list[EvaluationCase]:
        contract = self._ready_contract(contract_id)
        return self.evaluation_repository.replace_cases(
            contract_id,
            contract.index_version,
            entries,
        )

    def _check_case_version(
        self,
        contract: ContractRecord,
        case: EvaluationCase,
    ) -> None:
        if case.index_version != contract.index_version:
            raise EvaluationStaleError(
                "评测集对应旧索引，请重新保存/重新标注"
            )

    def create_single_run(self, contract_id: str, case_id: int) -> EvaluationRun:
        contract = self._ready_contract(contract_id)
        case = self.evaluation_repository.get_case(contract_id, case_id)
        if case is None:
            raise EvaluationCaseNotFoundError(case_id)
        self._check_case_version(contract, case)
        return self.evaluation_repository.create_run(
            contract_id,
            "single",
            contract.index_version,
            build_config_snapshot(),
            [case],
        )

    def create_all_run(self, contract_id: str) -> EvaluationRun:
        contract = self._ready_contract(contract_id)
        cases = self.evaluation_repository.list_cases(contract_id)
        if not cases:
            raise ValueError("evaluation set is empty")
        for case in cases:
            self._check_case_version(contract, case)
        return self.evaluation_repository.create_run(
            contract_id,
            "all",
            contract.index_version,
            build_config_snapshot(),
            cases,
        )

    def execute_run(self, run_id: str) -> EvaluationRun:
        run = self.evaluation_repository.get_run(run_id)
        if run is None:
            raise KeyError(run_id)

        self.evaluation_repository.mark_processing(run_id)
        try:
            contract = self._ready_contract(run.contract_id)
            if contract.index_version != run.index_version:
                raise EvaluationStaleError(
                    "评测运行对应旧索引，请重新保存评测集后再试"
                )
            index = self.index_manager.get(contract)
            cases = self.evaluation_repository.list_run_items(run_id)
            for item in cases:
                case = EvaluationCase(
                    case_id=item.case_id,
                    contract_id=run.contract_id,
                    index_version=run.index_version,
                    question=item.question_snapshot,
                    expected_source_object_indices=(
                        item.expected_source_object_indices_snapshot
                    ),
                    sort_order=len(cases),
                    created_at=run.created_at,
                    updated_at=run.created_at,
                )
                pipeline_result = self.pipeline.run(index, case.question)
                evaluated = build_evaluation_result(
                    pipeline_result,
                    case.expected_source_object_indices,
                )
                self.evaluation_repository.save_item(
                    run_id,
                    case,
                    serialize_pipeline_result(evaluated),
                )
            return self.evaluation_repository.mark_ready(run_id)
        except Exception as exc:
            logger.exception("evaluation run failed: %s", run_id)
            return self.evaluation_repository.mark_failed(run_id, str(exc))

    def get_run_payload(self, run_id: str) -> dict[str, Any]:
        run = self.evaluation_repository.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return {
            "run_id": run.run_id,
            "contract_id": run.contract_id,
            "scope": run.scope,
            "status": run.status,
            "index_version": run.index_version,
            "pipeline_version": run.pipeline_version,
            "config_snapshot": run.config_snapshot,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "error_message": run.error_message,
            "items": [
                {
                    "case_id": item.case_id,
                    "question": item.question_snapshot,
                    "expected_source_object_indices": (
                        item.expected_source_object_indices_snapshot
                    ),
                    "result": item.result,
                }
                for item in self.evaluation_repository.list_run_items(run_id)
            ],
        }

    def latest_run_payload(self, contract_id: str) -> dict[str, Any] | None:
        run = self.evaluation_repository.latest_run(contract_id)
        return None if run is None else self.get_run_payload(run.run_id)

    def recover_interrupted_runs(self) -> int:
        return self.evaluation_repository.recover_incomplete_runs(
            INTERRUPTED_RUN_MESSAGE
        )
