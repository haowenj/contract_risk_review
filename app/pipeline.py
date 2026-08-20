from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from llama_index.core import VectorStoreIndex

from app.config import Settings
from app.db import ContractRepository
from app.index_manager import IndexManager
from app.image_ingestion import ContractImageIngestionService, write_json_atomic
from app.models import ContractRecord
from clean_mineru_data import clean_content_list_file
from merge_cross_page_paragraphs import merge_content_list_file
from mineru_raw_parse import run_parse


logger = logging.getLogger(__name__)
ProcessMode = Literal["reuse_existing", "from_scratch"]


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


def build_default_image_ingestion_service(
    settings: Settings,
) -> ContractImageIngestionService:
    """Construct the configured VL and MinerU OCR adapters on demand."""

    from app.image_ocr import MinerUImageOCRService
    from app.image_understanding import ImageUnderstandingService

    vision_service = ImageUnderstandingService(
        model_name=settings.image_vision_model or os.environ["LLM_MODEL"],
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        timeout_seconds=settings.image_vision_timeout_seconds,
    )
    ocr_service = MinerUImageOCRService(
        svr_url=settings.mineru_url,
        backend=settings.mineru_backend,
        server_url=settings.mineru_server_url,
    )
    return ContractImageIngestionService(
        vision_service=vision_service,
        ocr_service=ocr_service,
    )


class ContractProcessor:
    def __init__(
        self,
        repository: ContractRepository,
        settings: Settings,
        index_manager: IndexManager,
        *,
        embedding_model: Any | None = None,
        image_ingestion_service: ContractImageIngestionService | None = None,
    ):
        self.repository = repository
        self.settings = settings
        self.index_manager = index_manager
        self.embedding_model = embedding_model
        self.image_ingestion_service = image_ingestion_service

    def process(
        self,
        contract_id: str,
        *,
        mode: ProcessMode = "from_scratch",
    ) -> ContractRecord:
        contract = self.repository.get(contract_id)
        if contract is None:
            raise KeyError(contract_id)
        if mode not in {"reuse_existing", "from_scratch"}:
            raise ValueError(f"unsupported process mode: {mode}")

        self.repository.update_status(contract_id, "processing")
        self.index_manager.clear(contract_id)
        try:
            paths = self._paths(contract)
            if mode == "from_scratch":
                run_parse(
                    paths["source"],
                    paths["raw"],
                    svr_url=self.settings.mineru_url,
                    backend=self.settings.mineru_backend,
                    server_url=self.settings.mineru_server_url,
                )
            elif not paths["raw"].is_file():
                raise FileNotFoundError(
                    f"raw_content_list.json not found: {paths['raw']}"
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

            if any(
                isinstance(obj, dict)
                and (
                    obj.get("type") == "image"
                    or (
                        obj.get("type") == "table"
                        and isinstance(obj.get("img_path"), str)
                        and bool(obj["img_path"].strip())
                    )
                )
                for obj in objects
            ):
                image_service = (
                    self.image_ingestion_service
                    or build_default_image_ingestion_service(self.settings)
                )
                objects = image_service.enrich_images(
                    objects,
                    storage_dir=Path(contract.storage_dir),
                )
                write_json_atomic(paths["merged"], objects)

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
