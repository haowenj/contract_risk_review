from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, load_index_from_storage

from app.models import ContractRecord


class IndexManager:
    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, contract: ContractRecord) -> Any:
        with self._lock:
            cached = self._cache.get(contract.contract_id)
            if cached is not None:
                return cached

            index_dir = Path(contract.storage_dir) / "index"
            if not index_dir.is_dir():
                raise FileNotFoundError(
                    f"persisted index directory not found: {index_dir}"
                )

            storage_context = StorageContext.from_defaults(
                persist_dir=str(index_dir)
            )
            index = load_index_from_storage(
                storage_context,
                embed_model=self.embedding_model,
            )
            self._cache[contract.contract_id] = index
            return index

    def put(self, contract_id: str, index: Any) -> None:
        with self._lock:
            self._cache[contract_id] = index

    def clear(self, contract_id: str | None = None) -> None:
        with self._lock:
            if contract_id is None:
                self._cache.clear()
            else:
                self._cache.pop(contract_id, None)
