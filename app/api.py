from __future__ import annotations

import logging
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, load_settings
from app.db import ContractRepository
from app.evaluation_forms import format_expected_indices, parse_expected_indices
from app.evaluation_service import (
    EvaluationCaseNotFoundError,
    EvaluationContractNotFoundError,
    EvaluationContractNotReadyError,
    EvaluationMetadataInvalidError,
    EvaluationMetadataNotFoundError,
    EvaluationRetrievalContextInvalidError,
    EvaluationRetrievalContextNotFoundError,
    EvaluationStaleError,
)
from app.index_manager import IndexManager
from app.markdown import render_markdown
from app.pipeline import ContractProcessor
from app.review_db import ContractReviewRepository
from app.review_service import ContractReviewWebService
from app.service import (
    ContractNotFoundError,
    ContractNotReadyError,
    ContractRawContentNotFoundError,
    ContractReprocessNotAllowedError,
    ContractService,
)
from app.status import status_label

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


class ReprocessRequest(BaseModel):
    mode: Literal["reuse_existing", "from_scratch"]


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
    contract_review_web_service: ContractReviewWebService | Any | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    active_service = service or build_default_service(settings)
    active_service.evaluation_service.recover_interrupted_runs()
    active_contract_review_web_service = (
        contract_review_web_service
        or ContractReviewWebService(
            contract_service=active_service,
            review_repository=ContractReviewRepository(settings.database_path),
            review_runs_dir=settings.data_dir / "review_runs",
        )
    )
    active_contract_review_web_service.recover_interrupted_runs()
    application = FastAPI(title="Contract Risk Review")
    templates = Jinja2Templates(
        directory=str(Path(__file__).resolve().parent / "templates")
    )
    templates.env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
    templates.env.filters["markdown"] = render_markdown
    templates.env.filters["status_label"] = status_label

    def render_dashboard(
        request: Request,
        *,
        error: str | None = None,
    ) -> HTMLResponse:
        contracts = active_service.list_contracts()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "contracts": contracts,
                "error": error,
            },
        )

    def render_chat_page(
        request: Request,
        contract_id: str,
        *,
        answer_result: dict[str, Any] | None = None,
        question: str = "",
        debug: bool = False,
        error: str | None = None,
    ) -> HTMLResponse:
        selected_contract = active_service.get_contract(contract_id)
        if selected_contract is None:
            raise HTTPException(status_code=404, detail="contract not found")
        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={
                "selected_contract": selected_contract,
                "answer_result": answer_result,
                "question": question,
                "debug": debug,
                "error": error,
            },
        )

    def render_chat_result(
        request: Request,
        contract_id: str,
        question: str,
        debug: bool,
    ) -> HTMLResponse:
        question = question.strip()
        if not question:
            return render_chat_page(
                request,
                contract_id,
                question=question,
                debug=debug,
                error="请输入问题。",
            )
        try:
            result = active_service.ask(contract_id, question, debug)
            return render_chat_page(
                request,
                contract_id,
                answer_result=result,
                question=question,
                debug=debug,
            )
        except ContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail="contract not found") from exc
        except ContractNotReadyError as exc:
            return render_chat_page(
                request,
                contract_id,
                question=question,
                debug=debug,
                error=f"合同当前状态为 {status_label(exc.record.status)}，请等待入库完成。",
            )
        except FileNotFoundError as exc:
            logger.exception("persisted index unavailable for %s", contract_id)
            return render_chat_page(
                request,
                contract_id,
                question=question,
                debug=debug,
                error="持久化索引不存在，请重新入库后再试。",
            )
        except Exception:
            logger.exception("server-rendered question failed for %s", contract_id)
            return render_chat_page(
                request,
                contract_id,
                question=question,
                debug=debug,
                error="提问失败，请稍后重试。",
            )

    def render_review_page(
        request: Request,
        contract_id: str,
        *,
        run_id: str | None = None,
        review_rule_text: str = "",
        error: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        selected_contract = active_service.get_contract(contract_id)
        if selected_contract is None:
            raise HTTPException(status_code=404, detail="contract not found")

        review_run = None
        if run_id:
            try:
                review_run = active_contract_review_web_service.get_run_payload(
                    contract_id,
                    run_id,
                )
            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="review run not found",
                ) from exc

        if selected_contract.status != "ready" and error is None:
            error = (
                f"合同当前状态为 {status_label(selected_contract.status)}，"
                "完成入库后才能进行风险评估。"
            )

        return templates.TemplateResponse(
            request=request,
            name="review.html",
            status_code=status_code,
            context={
                "selected_contract": selected_contract,
                "review_run": review_run,
                "run_id": run_id,
                "review_rule_text": review_rule_text,
                "error": error,
            },
        )

    async def resolve_review_rule_text(
        review_rule_text: str,
        review_rule_file: UploadFile | None,
    ) -> str:
        normalized_text = review_rule_text.strip()
        has_file = bool(review_rule_file and review_rule_file.filename)
        if normalized_text and has_file:
            raise ValueError("请只选择一种审查规范输入方式。")
        if normalized_text:
            return normalized_text
        if not has_file or review_rule_file is None:
            raise ValueError("请输入或上传审查规范。")

        suffix = Path(review_rule_file.filename or "").suffix.lower()
        if suffix not in {".txt", ".md"}:
            raise ValueError("审查规范文件仅支持 .txt 或 .md。")
        try:
            uploaded_text = (await review_rule_file.read()).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("审查规范文件必须使用 UTF-8 编码。") from exc
        uploaded_text = uploaded_text.strip()
        if not uploaded_text:
            raise ValueError("上传的审查规范文件不能为空。")
        return uploaded_text

    def evaluation_rows(
        cases: list[Any],
        *,
        contract_index_version: str | None,
    ) -> list[dict[str, Any]]:
        rows = []
        for case in cases:
            index_version = getattr(case, "index_version", None)
            rows.append(
                {
                    "case_id": getattr(case, "case_id", None),
                    "question": getattr(case, "question", ""),
                    "expected_text": format_expected_indices(
                        list(
                            getattr(
                                case,
                                "expected_source_object_indices",
                                [],
                            )
                        )
                    ),
                    "stale": bool(
                        contract_index_version
                        and index_version
                        and index_version != contract_index_version
                    ),
                }
            )
        return rows

    def render_evaluation_page(
        request: Request,
        contract_id: str,
        *,
        run_id: str | None = None,
        error: str | None = None,
        submitted_rows: list[dict[str, Any]] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        selected_contract = active_service.get_contract(contract_id)
        if selected_contract is None:
            raise HTTPException(status_code=404, detail="contract not found")

        saved_cases: list[Any] = []
        default_cases: list[tuple[str, list[int]]] = []
        if selected_contract.status == "ready":
            saved_cases = active_service.list_evaluation_cases(contract_id)
            if not saved_cases:
                default_cases = active_service.default_evaluation_cases()

        if submitted_rows is not None:
            rows = submitted_rows
        elif saved_cases:
            rows = evaluation_rows(
                saved_cases,
                contract_index_version=selected_contract.index_version,
            )
        else:
            rows = [
                {
                    "case_id": None,
                    "question": question,
                    "expected_text": format_expected_indices(expected),
                    "stale": False,
                }
                for question, expected in default_cases
            ]

        latest_run = None
        if selected_contract.status == "ready":
            try:
                latest_run = (
                    active_service.get_evaluation_run_payload(run_id)
                    if run_id
                    else active_service.latest_evaluation_run_payload(
                        contract_id
                    )
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="run not found") from exc

        if selected_contract.status != "ready" and error is None:
            error = f"合同当前状态为 {status_label(selected_contract.status)}，完成入库后才能进行评测。"

        return templates.TemplateResponse(
            request=request,
            name="evaluation.html",
            status_code=status_code,
            context={
                "selected_contract": selected_contract,
                "evaluation_rows": rows,
                "latest_run": latest_run,
                "run_id": run_id,
                "error": error,
            },
        )

    def render_metadata_page(
        request: Request,
        contract_id: str,
    ) -> HTMLResponse:
        selected_contract = active_service.get_contract(contract_id)
        if selected_contract is None:
            raise HTTPException(status_code=404, detail="contract not found")

        try:
            source_objects = active_service.list_source_object_entries(
                contract_id
            )
        except EvaluationContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail="contract not found") from exc
        except EvaluationContractNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EvaluationMetadataNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="merged_content_list.json not found",
            ) from exc
        except EvaluationMetadataInvalidError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return templates.TemplateResponse(
            request=request,
            name="metadata.html",
            context={
                "selected_contract": selected_contract,
                "source_objects": source_objects,
            },
        )

    def render_retrieval_context_page(
        request: Request,
        contract_id: str,
    ) -> HTMLResponse:
        selected_contract = active_service.get_contract(contract_id)
        if selected_contract is None:
            raise HTTPException(status_code=404, detail="contract not found")

        try:
            context_entries = active_service.list_retrieval_context_entries(
                contract_id
            )
        except EvaluationContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail="contract not found") from exc
        except EvaluationContractNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EvaluationRetrievalContextNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="retrieval_context.json not found",
            ) from exc
        except EvaluationRetrievalContextInvalidError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return templates.TemplateResponse(
            request=request,
            name="retrieval_context.html",
            context={
                "selected_contract": selected_contract,
                "context_entries": context_entries,
            },
        )

    def evaluation_error_message(exc: Exception) -> str:
        if isinstance(exc, EvaluationStaleError):
            return str(exc)
        if isinstance(exc, ValueError):
            return str(exc)
        if isinstance(exc, EvaluationContractNotReadyError):
            return (
                f"合同当前状态为 {status_label(exc.record.status)}，没有可用的索引版本，"
                "请重新入库后再试。"
            )
        if isinstance(exc, EvaluationContractNotFoundError):
            return "合同不存在。"
        return "评测操作失败，请稍后重试。"

    def evaluation_error_status_code(exc: Exception) -> int:
        if isinstance(exc, EvaluationContractNotFoundError):
            return 404
        if isinstance(exc, (EvaluationContractNotReadyError, EvaluationStaleError)):
            return 409
        if isinstance(exc, ValueError):
            return 400
        return 500

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

    @application.post("/api/contracts/{contract_id}/reprocess", status_code=202)
    def reprocess_contract(
        contract_id: str,
        payload: ReprocessRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        try:
            record = active_service.reprocess_contract(contract_id, payload.mode)
        except ContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail="contract not found") from exc
        except ContractReprocessNotAllowedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ContractRawContentNotFoundError as exc:
            raise HTTPException(
                status_code=409,
                detail="已有解析结果不存在，请选择“从头解析合同”",
            ) from exc
        background_tasks.add_task(
            active_service.processor.process,
            record.contract_id,
            mode=payload.mode,
        )
        return _record_payload(record)

    @application.get("/api/contracts/{contract_id}")
    def get_contract(contract_id: str) -> dict[str, Any]:
        record = active_service.get_contract(contract_id)
        if record is None:
            raise HTTPException(status_code=404, detail="contract not found")
        return _record_payload(record)

    @application.get("/contracts/{contract_id}/images/{img_path:path}")
    def contract_image(contract_id: str, img_path: str) -> FileResponse:
        """Serve only image assets below the selected contract directory."""

        record = active_service.get_contract(contract_id)
        if record is None:
            raise HTTPException(status_code=404, detail="contract not found")

        reference = PurePosixPath(img_path.replace("\\", "/"))
        if reference.is_absolute() or ".." in reference.parts:
            raise HTTPException(status_code=404, detail="image not found")
        target_root = Path(record.storage_dir).expanduser().resolve()
        target = (target_root / Path(*reference.parts)).resolve()
        if not target.is_relative_to(target_root) or not target.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        media_type = mimetypes.guess_type(target.name)[0]
        if media_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(target, media_type=media_type)

    @application.get(
        "/api/contracts/{contract_id}/evaluation/runs/{run_id}"
    )
    def get_evaluation_run(contract_id: str, run_id: str) -> dict[str, Any]:
        try:
            payload = active_service.get_evaluation_run_payload(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="evaluation run not found") from exc
        if payload.get("contract_id") != contract_id:
            raise HTTPException(status_code=404, detail="evaluation run not found")
        return payload

    @application.get(
        "/api/contracts/{contract_id}/review/runs/{run_id}"
    )
    def get_contract_review_run(
        contract_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        try:
            return active_contract_review_web_service.get_run_payload(
                contract_id,
                run_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="review run not found",
            ) from exc

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

    @application.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return render_dashboard(request)

    @application.post("/upload", response_class=HTMLResponse)
    async def upload_page(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> Response:
        try:
            record = active_service.create_upload(
                file.filename or "",
                await file.read(),
            )
        except ValueError as exc:
            return render_dashboard(request, error=str(exc))
        background_tasks.add_task(active_service.processor.process, record.contract_id)
        return RedirectResponse(url="/", status_code=303)

    @application.post("/chat", response_class=HTMLResponse)
    def chat_page(
        request: Request,
        contract_id: str = Form(...),
        question: str = Form(...),
        debug: bool = Form(False),
    ) -> HTMLResponse:
        return render_chat_result(request, contract_id, question, debug)

    @application.get("/contracts/{contract_id}", response_class=HTMLResponse)
    def contract_page(request: Request, contract_id: str) -> HTMLResponse:
        return render_chat_page(request, contract_id)

    @application.get(
        "/contracts/{contract_id}/review",
        response_class=HTMLResponse,
    )
    def contract_review_page(
        request: Request,
        contract_id: str,
        run_id: str | None = None,
    ) -> HTMLResponse:
        return render_review_page(request, contract_id, run_id=run_id)

    @application.post(
        "/contracts/{contract_id}/review/runs",
        response_class=HTMLResponse,
    )
    async def create_contract_review_run(
        request: Request,
        contract_id: str,
        background_tasks: BackgroundTasks,
        review_rule_text: str = Form(default=""),
        review_rule_file: UploadFile | None = File(default=None),
    ) -> Response:
        try:
            normalized_text = await resolve_review_rule_text(
                review_rule_text,
                review_rule_file,
            )
            run = active_contract_review_web_service.create_run(
                contract_id,
                normalized_text,
            )
        except ContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail="contract not found") from exc
        except ContractNotReadyError as exc:
            return render_review_page(
                request,
                contract_id,
                review_rule_text=review_rule_text,
                error=(
                    f"合同当前状态为 {status_label(exc.record.status)}，"
                    "完成入库后才能进行风险评估。"
                ),
                status_code=409,
            )
        except ValueError as exc:
            return render_review_page(
                request,
                contract_id,
                review_rule_text=review_rule_text,
                error=str(exc),
                status_code=400,
            )

        background_tasks.add_task(
            active_contract_review_web_service.execute_run,
            run.run_id,
        )
        return RedirectResponse(
            url=f"/contracts/{contract_id}/review?run_id={run.run_id}",
            status_code=303,
        )

    @application.get(
        "/contracts/{contract_id}/evaluation",
        response_class=HTMLResponse,
    )
    def evaluation_page(
        request: Request,
        contract_id: str,
        run_id: str | None = None,
    ) -> HTMLResponse:
        return render_evaluation_page(request, contract_id, run_id=run_id)

    @application.get(
        "/contracts/{contract_id}/evaluation/metadata",
        response_class=HTMLResponse,
    )
    def metadata_page(request: Request, contract_id: str) -> HTMLResponse:
        return render_metadata_page(request, contract_id)

    @application.get(
        "/contracts/{contract_id}/evaluation/retrieval-context",
        response_class=HTMLResponse,
    )
    def retrieval_context_page(request: Request, contract_id: str) -> HTMLResponse:
        return render_retrieval_context_page(request, contract_id)

    @application.post(
        "/contracts/{contract_id}/evaluation/config",
        response_class=HTMLResponse,
    )
    def save_evaluation_config(
        request: Request,
        contract_id: str,
        question: list[str] = Form(default=[]),
        expected_source_object_indices: list[str] = Form(default=[]),
    ) -> Response:
        submitted_rows: list[dict[str, Any]] = []
        try:
            if len(question) != len(expected_source_object_indices):
                raise ValueError("评测问题和正确 Node ID 行数不一致")

            entries: list[tuple[str, list[int]]] = []
            for current_question, raw_indices in zip(
                question,
                expected_source_object_indices,
            ):
                normalized_question = current_question.strip()
                normalized_indices = parse_expected_indices(raw_indices)
                if not normalized_question and not raw_indices.strip():
                    continue
                if not normalized_question:
                    raise ValueError("问题不能为空")
                submitted_rows.append(
                    {
                        "case_id": None,
                        "question": normalized_question,
                        "expected_text": format_expected_indices(normalized_indices),
                        "stale": False,
                    }
                )
                entries.append((normalized_question, normalized_indices))

            active_service.save_evaluation_cases(contract_id, entries)
            return RedirectResponse(
                url=f"/contracts/{contract_id}/evaluation",
                status_code=303,
            )
        except (
            EvaluationContractNotFoundError,
            EvaluationContractNotReadyError,
            ValueError,
            EvaluationStaleError,
        ) as exc:
            if not submitted_rows:
                submitted_rows = [
                    {
                        "case_id": None,
                        "question": current_question,
                        "expected_text": raw_indices,
                        "stale": False,
                    }
                    for current_question, raw_indices in zip(
                        question,
                        expected_source_object_indices,
                    )
                ]
            return render_evaluation_page(
                request,
                contract_id,
                error=evaluation_error_message(exc),
                submitted_rows=submitted_rows,
                status_code=evaluation_error_status_code(exc),
            )

    @application.post(
        "/contracts/{contract_id}/evaluation/cases/{case_id}/run",
        response_class=HTMLResponse,
    )
    def run_single_evaluation(
        contract_id: str,
        case_id: int,
        background_tasks: BackgroundTasks,
    ) -> Response:
        try:
            run = active_service.create_single_evaluation_run(contract_id, case_id)
        except (
            EvaluationContractNotFoundError,
            EvaluationCaseNotFoundError,
        ) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EvaluationContractNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EvaluationStaleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        background_tasks.add_task(active_service.execute_evaluation_run, run.run_id)
        return RedirectResponse(
            url=f"/contracts/{contract_id}/evaluation?run_id={run.run_id}",
            status_code=303,
        )

    @application.post(
        "/contracts/{contract_id}/evaluation/run-all",
        response_class=HTMLResponse,
    )
    def run_all_evaluation(
        contract_id: str,
        background_tasks: BackgroundTasks,
    ) -> Response:
        try:
            run = active_service.create_all_evaluation_run(contract_id)
        except EvaluationContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EvaluationContractNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, EvaluationStaleError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        background_tasks.add_task(active_service.execute_evaluation_run, run.run_id)
        return RedirectResponse(
            url=f"/contracts/{contract_id}/evaluation?run_id={run.run_id}",
            status_code=303,
        )

    @application.post(
        "/contracts/{contract_id}/chat",
        response_class=HTMLResponse,
    )
    def contract_chat_page(
        request: Request,
        contract_id: str,
        question: str = Form(...),
        debug: bool = Form(False),
    ) -> HTMLResponse:
        return render_chat_result(request, contract_id, question, debug)

    return application
