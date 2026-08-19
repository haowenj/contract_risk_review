from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import Settings
from app.db import ContractRepository
from app.evaluation_db import EvaluationRepository
from app.evaluation_service import EvaluationService
from app.evidence_serialization import serialize_node_result
from app.index_manager import IndexManager
from app.models import ContractRecord
from app.qa import answer_question
from app.rag_pipeline import RAGPipeline


class ContractNotFoundError(LookupError):
    pass


class ContractNotReadyError(RuntimeError):
    def __init__(self, record: ContractRecord):
        self.record = record
        super().__init__(f"contract {record.contract_id} is {record.status}")


class ContractReprocessNotAllowedError(RuntimeError):
    def __init__(self, record: ContractRecord):
        self.record = record
        super().__init__(
            f"contract {record.contract_id} cannot reprocess from {record.status}"
        )


class ContractRawContentNotFoundError(FileNotFoundError):
    pass


class ContractService:
    def __init__(
        self,
        repository: ContractRepository,
        settings: Settings,
        processor: Any,
        index_manager: IndexManager,
        *,
        rag_pipeline: RAGPipeline | None = None,
        evaluation_service: EvaluationService | Any | None = None,
    ):
        self.repository = repository
        self.settings = settings
        self.processor = processor
        self.index_manager = index_manager
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.evaluation_service = evaluation_service or EvaluationService(
            repository,
            EvaluationRepository(settings.database_path),
            index_manager,
            self.rag_pipeline,
        )

    def create_upload(self, filename: str, content: bytes) -> ContractRecord:
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise ValueError("only PDF files are supported")

        contract_id = self._new_contract_id()
        storage_dir = self.settings.contracts_dir / contract_id
        storage_dir.mkdir(parents=True, exist_ok=False)
        (storage_dir / "source.pdf").write_bytes(content)
        return self.repository.create(filename, storage_dir, contract_id=contract_id)

    def _new_contract_id(self) -> str:
        import uuid

        return str(uuid.uuid4())

    def get_contract(self, contract_id: str) -> ContractRecord | None:
        return self.repository.get(contract_id)

    def list_contracts(self) -> list[ContractRecord]:
        return self.repository.list()

    def reprocess_contract(self, contract_id: str, mode: str) -> ContractRecord:
        if mode not in {"reuse_existing", "from_scratch"}:
            raise ValueError(f"unsupported reprocess mode: {mode}")

        contract = self.repository.get(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status not in {"ready", "failed"}:
            raise ContractReprocessNotAllowedError(contract)
        if mode == "reuse_existing":
            raw_path = Path(contract.storage_dir) / "raw_content_list.json"
            if not raw_path.is_file():
                raise ContractRawContentNotFoundError(str(raw_path))
        return self.repository.update_status(contract_id, "queued")

    def ask(self, contract_id: str, question: str, debug: bool = False) -> dict[str, Any]:
        contract = self.repository.get(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status != "ready":
            raise ContractNotReadyError(contract)

        index = self.index_manager.get(contract)
        result = answer_question(
            index,
            question,
            debug=debug,
            pipeline=self.rag_pipeline,
        )
        return {
            "contract_id": contract_id,
            "question": question,
            **result,
        }

    def search_contract(
        self,
        contract_id: str,
        query: str,
        *,
        debug_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")

        contract = self.repository.get(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status != "ready":
            raise ContractNotReadyError(contract)

        index = self.index_manager.get(contract)
        retrieval = self.rag_pipeline.retrieve_evidence(
            index,
            query,
            fallback_on_empty_selection=False,
        )
        selected_nodes = retrieval.get("selected_nodes", [])
        if not selected_nodes and debug_callback is not None:
            debug_callback(
                [
                    {
                        "source_object_index": serialized.get(
                            "source_object_index"
                        ),
                        "text": serialized.get("text", ""),
                    }
                    for result in retrieval.get("reranked_results", [])[:3]
                    for serialized in [serialize_node_result(result)]
                ]
            )
        return [
            serialize_node_result(result)
            for result in selected_nodes
        ]

    def load_contract_content_objects(
        self,
        contract_id: str,
    ) -> list[dict[str, Any]]:
        contract = self.repository.get(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status != "ready":
            raise ContractNotReadyError(contract)

        source_path = Path(contract.storage_dir) / "merged_content_list.json"
        with source_path.open("r", encoding="utf-8") as source_file:
            payload = json.load(source_file)
        if not isinstance(payload, list):
            raise ValueError("merged_content_list.json must contain a JSON array")
        if any(not isinstance(value, dict) for value in payload):
            raise ValueError("merged content objects must be JSON objects")
        return payload

    def recover_interrupted_evaluation_runs(self) -> int:
        return self.evaluation_service.recover_interrupted_runs()

    def list_evaluation_cases(self, contract_id: str):
        return self.evaluation_service.list_cases(contract_id)

    def default_evaluation_cases(self):
        return self.evaluation_service.default_cases()

    def list_source_object_entries(self, contract_id: str):
        return self.evaluation_service.list_source_object_entries(contract_id)

    def list_retrieval_context_entries(self, contract_id: str):
        return self.evaluation_service.list_retrieval_context_entries(contract_id)

    def save_evaluation_cases(self, contract_id: str, entries):
        return self.evaluation_service.save_cases(contract_id, entries)

    def create_single_evaluation_run(self, contract_id: str, case_id: int):
        return self.evaluation_service.create_single_run(contract_id, case_id)

    def create_all_evaluation_run(self, contract_id: str):
        return self.evaluation_service.create_all_run(contract_id)

    def execute_evaluation_run(self, run_id: str):
        return self.evaluation_service.execute_run(run_id)

    def get_evaluation_run_payload(self, run_id: str):
        return self.evaluation_service.get_run_payload(run_id)

    def latest_evaluation_run_payload(self, contract_id: str):
        return self.evaluation_service.latest_run_payload(contract_id)
