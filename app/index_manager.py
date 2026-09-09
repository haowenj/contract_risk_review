from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, load_index_from_storage

from app.bm25_index import BM25Index
from app.models import ContractRecord


class IndexManager:
    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model
        self._cache: dict[tuple[str, str | None], Any] = {}
        self._bm25_cache: dict[tuple[str, str | None], BM25Index] = {}
        self._lock = threading.RLock()

    def get(self, contract: ContractRecord) -> Any:
        with self._lock:
            cache_key = (
                contract.contract_id,
                getattr(contract, "index_version", None),
            )
            cached = self._cache.get(cache_key)
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
            self._cache[cache_key] = index
            return index

    def get_bm25(self, contract: ContractRecord) -> BM25Index:
        with self._lock:
            cache_key = (
                contract.contract_id,
                getattr(contract, "index_version", None),
            )
            cached = self._bm25_cache.get(cache_key)
            if cached is not None:
                return cached

            index_dir = Path(contract.storage_dir) / "bm25_index"
            if not index_dir.is_dir():
                raise FileNotFoundError(
                    f"persisted BM25 index directory not found: {index_dir}"
                )
            index = BM25Index.load(index_dir)
            self._bm25_cache[cache_key] = index
            return index

    def put(
        self,
        contract_id: str,
        index: Any,
        *,
        index_version: str | None = None,
    ) -> None:
        with self._lock:
            self._cache[(contract_id, index_version)] = index

    def put_bm25(
        self,
        contract_id: str,
        index: BM25Index,
        *,
        index_version: str | None = None,
    ) -> None:
        with self._lock:
            self._bm25_cache[(contract_id, index_version)] = index

    def clear(self, contract_id: str | None = None) -> None:
        with self._lock:
            if contract_id is None:
                self._cache.clear()
                self._bm25_cache.clear()
            else:
                for cache_key in [
                    key for key in self._cache if key[0] == contract_id
                ]:
                    self._cache.pop(cache_key, None)
                for cache_key in [
                    key for key in self._bm25_cache if key[0] == contract_id
                ]:
                    self._bm25_cache.pop(cache_key, None)
