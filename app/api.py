from __future__ import annotations

import logging
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, load_settings
from app.db import ContractRepository
from app.index_manager import IndexManager
from app.pipeline import ContractProcessor
from app.service import (
    ContractNotFoundError,
    ContractNotReadyError,
    ContractService,
)


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


def build_default_service(settings: Settings) -> ContractService:
    from mineru_to_nodes import embedding_model

    repository = ContractRepository(settings.database_path)
    index_manager = IndexManager(embedding_model)
    processor = ContractProcessor(
        repository,
        settings,
        index_manager,
        embedding_model=embedding_model,
    )
    return ContractService(repository, settings, processor, index_manager)


def _record_payload(record: Any) -> dict[str, Any]:
    return record.to_dict()


def create_app(
    *,
    settings: Settings | None = None,
    service: ContractService | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    active_service = service or build_default_service(settings)
    application = FastAPI(title="Contract Risk Review")

    @application.post("/api/contracts", status_code=202)
    async def upload_contract(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            record = active_service.create_upload(
                file.filename or "",
                await file.read(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(active_service.processor.process, record.contract_id)
        return _record_payload(record)

    @application.get("/api/contracts")
    def list_contracts() -> list[dict[str, Any]]:
        return [
            _record_payload(record)
            for record in active_service.list_contracts()
        ]

    @application.get("/api/contracts/{contract_id}")
    def get_contract(contract_id: str) -> dict[str, Any]:
        record = active_service.get_contract(contract_id)
        if record is None:
            raise HTTPException(status_code=404, detail="contract not found")
        return _record_payload(record)

    @application.post("/api/contracts/{contract_id}/chat")
    def chat(contract_id: str, request: ChatRequest) -> dict[str, Any]:
        try:
            return active_service.ask(contract_id, request.question, request.debug)
        except ContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail="contract not found") from exc
        except ContractNotReadyError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "contract is not ready",
                    "status": exc.record.status,
                },
            ) from exc
        except FileNotFoundError as exc:
            logger.exception("persisted index unavailable for %s", contract_id)
            raise HTTPException(status_code=500, detail="persisted index unavailable") from exc
        except Exception as exc:
            logger.exception("contract question failed for %s", contract_id)
            raise HTTPException(status_code=500, detail="question failed") from exc

    return application
