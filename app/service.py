from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.db import ContractRepository
from app.index_manager import IndexManager
from app.models import ContractRecord
from app.qa import answer_question


class ContractNotFoundError(LookupError):
    pass


class ContractNotReadyError(RuntimeError):
    def __init__(self, record: ContractRecord):
        self.record = record
        super().__init__(f"contract {record.contract_id} is {record.status}")


class ContractService:
    def __init__(
        self,
        repository: ContractRepository,
        settings: Settings,
        processor: Any,
        index_manager: IndexManager,
    ):
        self.repository = repository
        self.settings = settings
        self.processor = processor
        self.index_manager = index_manager

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

    def ask(self, contract_id: str, question: str, debug: bool = False) -> dict[str, Any]:
        contract = self.repository.get(contract_id)
        if contract is None:
            raise ContractNotFoundError(contract_id)
        if contract.status != "ready":
            raise ContractNotReadyError(contract)

        index = self.index_manager.get(contract)
        result = answer_question(index, question, debug=debug)
        return {
            "contract_id": contract_id,
            "question": question,
            **result,
        }
