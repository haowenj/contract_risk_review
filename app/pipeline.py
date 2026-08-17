from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex

from app.config import Settings
from app.db import ContractRepository
from app.index_manager import IndexManager
from app.models import ContractRecord
from clean_mineru_data import clean_content_list_file
from merge_cross_page_paragraphs import merge_content_list_file
from mineru_raw_parse import run_parse


logger = logging.getLogger(__name__)


def generate_contexts(objects: list[dict[str, Any]]) -> dict[int, str | None]:
    from retrieval_context_preprocess import generate_contexts as generate

    return generate(objects)


def save_retrieval_contexts(
    contexts: dict[int, str | None],
    path: Path,
) -> None:
    from retrieval_context_preprocess import save_retrieval_contexts as save

    save(contexts, path)


def build_nodes(
    objects: list[dict[str, Any]],
    *,
    retrieval_contexts: dict[int, str | None],
) -> list[Any]:
    from mineru_to_nodes import build_nodes as build

    return build(objects, retrieval_contexts=retrieval_contexts)


def get_embedding_model() -> Any:
    from mineru_to_nodes import embedding_model

    return embedding_model


class ContractProcessor:
    def __init__(
        self,
        repository: ContractRepository,
        settings: Settings,
        index_manager: IndexManager,
        *,
        embedding_model: Any | None = None,
    ):
        self.repository = repository
        self.settings = settings
        self.index_manager = index_manager
        self.embedding_model = embedding_model

    def process(self, contract_id: str) -> ContractRecord:
        contract = self.repository.get(contract_id)
        if contract is None:
            raise KeyError(contract_id)

        self.repository.update_status(contract_id, "processing")
        self.index_manager.clear(contract_id)
        try:
            paths = self._paths(contract)
            run_parse(
                paths["source"],
                paths["raw"],
                svr_url=self.settings.mineru_url,
                backend=self.settings.mineru_backend,
                server_url=self.settings.mineru_server_url,
            )
            clean_content_list_file(paths["raw"], paths["cleaned"])
            merge_content_list_file(
                paths["cleaned"],
                paths["merged"],
                paths["merge_log"],
            )

            objects = json.loads(paths["merged"].read_text(encoding="utf-8"))
            if not isinstance(objects, list):
                raise ValueError("merged content list must contain a JSON list")

            contexts = generate_contexts(objects)
            save_retrieval_contexts(contexts, paths["context"])
            nodes = build_nodes(objects, retrieval_contexts=contexts)
            model = self.embedding_model or get_embedding_model()
            index = VectorStoreIndex(nodes, embed_model=model)
            paths["index"].mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(paths["index"]))
            index_version = str(uuid.uuid4())
            self.index_manager.put(
                contract_id,
                index,
                index_version=index_version,
            )
            return self.repository.update_status(
                contract_id,
                "ready",
                index_version=index_version,
            )
        except Exception as exc:
            logger.exception("contract ingestion failed: %s", contract_id)
            return self.repository.update_status(contract_id, "failed", str(exc))

    @staticmethod
    def _paths(contract: ContractRecord) -> dict[str, Path]:
        storage_dir = Path(contract.storage_dir)
        return {
            "source": storage_dir / "source.pdf",
            "raw": storage_dir / "raw_content_list.json",
            "cleaned": storage_dir / "cleaned_content_list.json",
            "merged": storage_dir / "merged_content_list.json",
            "merge_log": storage_dir / "merge_log.json",
            "context": storage_dir / "retrieval_context.json",
            "index": storage_dir / "index",
        }
