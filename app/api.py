from __future__ import annotations

import logging
import mimetypes
import uuid
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
from pydantic import BaseModel

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
from app.evidence_review.presentation import (
    identifier_label,
    summarize_evidence_statuses,
)
from app.evidence_review.repository import (
    DecisionConflictError,
    EvidenceReviewRepository,
    RuleSetRecord,
    RuleSetTransitionError,
)
from app.evidence_review.rule_import import (
    RuleDocumentExtractor,
    RuleSetImportService,
    build_rule_parse_llm,
    validate_rule_upload,
)
from app.evidence_review.web_service import (
    EvidenceReviewWebService,
    RuleSetNotActiveError,
    RuleSetNotFoundError,
)
from app.index_manager import IndexManager
from app.markdown import render_markdown
from app.pipeline import ContractProcessor
from app.review_db import ContractReviewRepository, ReviewRunTransitionError
from app.review_service import ContractReviewWebService
from app.service import (
    ContractNotFoundError,
    ContractNotReadyError,
    ContractRawContentNotFoundError,
    ContractReprocessNotAllowedError,
    ContractService,
)
from app.status import processing_stage_label, status_label

logger = logging.getLogger(__name__)


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


def _rule_set_payload(record: RuleSetRecord) -> dict[str, Any]:
    """Return only fields that are safe and useful to browser clients."""
    return {
        "rule_set_id": record.rule_set_id,
        "name": record.name,
        "version": record.version,
        "status": record.status,
        "source_filename": record.source_filename,
        "parsed_rules": record.parsed_rules,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "activated_at": record.activated_at,
        "error_message": record.error_message,
    }


def create_app(
    *,
    settings: Settings | None = None,
    service: ContractService | None = None,
    contract_review_web_service: ContractReviewWebService | Any | None = None,
    rule_set_repository: EvidenceReviewRepository | None = None,
    rule_set_import_service: RuleSetImportService | Any | None = None,
    evidence_review_web_service: EvidenceReviewWebService | Any | None = None,
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
    active_rule_set_repository = rule_set_repository or EvidenceReviewRepository(
        settings.database_path
    )
    active_evidence_review_web_service = (
        evidence_review_web_service
        or EvidenceReviewWebService(
            contract_service=active_service,
            repository=active_rule_set_repository,
        )
    )
    active_evidence_review_web_service.recover_interrupted_runs()
    application = FastAPI(title="Contract Risk Review")
    templates = Jinja2Templates(
        directory=str(Path(__file__).resolve().parent / "templates")
    )
    templates.env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
    templates.env.filters["markdown"] = render_markdown
    templates.env.filters["status_label"] = status_label
    templates.env.filters["processing_stage_label"] = processing_stage_label

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

    def render_rule_sets_page(
        request: Request,
        *,
        error: str | None = None,
        status_code: int = 200,
        submitted_name: str = "",
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="rule_sets.html",
            status_code=status_code,
            context={
                "rule_sets": active_rule_set_repository.list_rule_sets(),
                "error": error,
                "submitted_name": submitted_name,
            },
        )

    def get_rule_set_or_404(rule_set_id: str) -> RuleSetRecord:
        record = active_rule_set_repository.get_rule_set(rule_set_id)
        if record is None:
            raise HTTPException(status_code=404, detail="rule set not found")
        return record

    def execute_rule_set_import(rule_set_id: str) -> None:
        importer = rule_set_import_service
        if importer is None:
            importer = RuleSetImportService(
                repository=active_rule_set_repository,
                extractor=RuleDocumentExtractor(
                    svr_url=settings.mineru_url,
                    backend=settings.mineru_backend,
                    server_url=settings.mineru_server_url,
                ),
                parse_llm=build_rule_parse_llm(),
                rule_sets_dir=Path(settings.rule_sets_dir),
                max_upload_bytes=settings.max_rule_upload_bytes,
            )
        importer.import_rule_set(rule_set_id)

    def render_evidence_review_page(
        request: Request,
        contract_id: str,
        *,
        run_id: str | None = None,
        error: str | None = None,
        status_code: int = 200,
        decision_item_id: str | None = None,
        decision_input: dict[str, str] | None = None,
    ) -> HTMLResponse:
        contract = active_service.get_contract(contract_id)
        if contract is None:
            raise HTTPException(status_code=404, detail="contract not found")
        run_payload = None
        if run_id:
            try:
                run_payload = (
                    active_evidence_review_web_service.get_run_payload(
                        contract_id,
                        run_id,
                    )
                )
            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="evidence review run not found",
                ) from exc
        if contract.status != "ready" and error is None:
            error = (
                f"合同当前状态为 {status_label(contract.status)}，"
                "完成入库后才能进行人工取证。"
            )
        active_rule_sets = [
            value
            for value in active_rule_set_repository.list_rule_sets()
            if value.status == "active"
        ]
        return templates.TemplateResponse(
            request=request,
            name="evidence_review.html",
            status_code=status_code,
            context={
                "selected_contract": contract,
                "active_rule_sets": active_rule_sets,
                "run": run_payload,
                "run_id": run_id,
                "error": error,
                "decision_item_id": decision_item_id,
                "decision_input": decision_input or {},
                "evidence_summary": summarize_evidence_statuses(
                    run_payload["items"]
                    if run_payload and run_payload["status"] == "ready"
                    else []
                ),
                "identifier_label": identifier_label,
            },
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

    @application.get("/api/rule-sets/{rule_set_id}")
    def get_rule_set(rule_set_id: str) -> dict[str, Any]:
        return _rule_set_payload(get_rule_set_or_404(rule_set_id))

    @application.get(
        "/api/contracts/{contract_id}/evidence-review/runs/{run_id}"
    )
    def get_evidence_review_run(
        contract_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        try:
            return active_evidence_review_web_service.get_run_payload(
                contract_id,
                run_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="evidence review run not found",
            ) from exc

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

    @application.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return render_dashboard(request)

    @application.get("/rule-sets", response_class=HTMLResponse)
    def rule_sets_page(request: Request) -> HTMLResponse:
        return render_rule_sets_page(request)

    @application.post("/rule-sets", response_class=HTMLResponse)
    async def upload_rule_set(
        request: Request,
        background_tasks: BackgroundTasks,
        name: str = Form(...),
        file: UploadFile = File(...),
    ) -> Response:
        normalized_name = name.strip()
        try:
            if not normalized_name:
                raise ValueError("规则集名称不能为空。")
            upload = validate_rule_upload(
                file.filename or "",
                await file.read(),
                max_bytes=settings.max_rule_upload_bytes,
            )
            upload_dir = (
                Path(settings.rule_sets_dir)
                / "uploads"
                / str(uuid.uuid4())
            )
            upload_dir.mkdir(parents=True, exist_ok=False)
            source_path = upload_dir / f"source{upload.suffix}"
            source_path.write_bytes(upload.content)
            record = active_rule_set_repository.create_rule_set(
                name=normalized_name,
                source_filename=upload.filename,
                source_path=source_path,
                source_sha256=upload.sha256,
            )
        except ValueError as exc:
            return render_rule_sets_page(
                request,
                error=str(exc),
                status_code=400,
                submitted_name=name,
            )

        background_tasks.add_task(execute_rule_set_import, record.rule_set_id)
        return RedirectResponse(
            url=f"/rule-sets/{record.rule_set_id}",
            status_code=303,
        )

    @application.get(
        "/rule-sets/{rule_set_id}",
        response_class=HTMLResponse,
    )
    def rule_set_detail_page(
        request: Request,
        rule_set_id: str,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="rule_set_detail.html",
            context={"rule_set": get_rule_set_or_404(rule_set_id)},
        )

    @application.post("/rule-sets/{rule_set_id}/activate")
    def activate_rule_set(rule_set_id: str) -> Response:
        get_rule_set_or_404(rule_set_id)
        try:
            active_rule_set_repository.activate_rule_set(rule_set_id)
        except RuleSetTransitionError as exc:
            raise HTTPException(
                status_code=409,
                detail="only a draft rule set can be activated",
            ) from exc
        return RedirectResponse(
            url=f"/rule-sets/{rule_set_id}",
            status_code=303,
        )

    @application.get(
        "/contracts/{contract_id}/evidence-review",
        response_class=HTMLResponse,
    )
    def evidence_review_page(
        request: Request,
        contract_id: str,
        run_id: str | None = None,
    ) -> HTMLResponse:
        return render_evidence_review_page(
            request,
            contract_id,
            run_id=run_id,
        )

    @application.post(
        "/contracts/{contract_id}/evidence-review/runs",
        response_class=HTMLResponse,
    )
    def create_evidence_review_run(
        request: Request,
        contract_id: str,
        background_tasks: BackgroundTasks,
        rule_set_id: str = Form(...),
    ) -> Response:
        try:
            run = active_evidence_review_web_service.create_run(
                contract_id,
                rule_set_id,
            )
        except ContractNotFoundError as exc:
            raise HTTPException(status_code=404, detail="contract not found") from exc
        except ContractNotReadyError:
            return render_evidence_review_page(
                request,
                contract_id,
                error="完成入库后才能进行人工取证。",
                status_code=409,
            )
        except RuleSetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="rule set not found") from exc
        except RuleSetNotActiveError:
            return render_evidence_review_page(
                request,
                contract_id,
                error="所选规则集尚未启用，请先确认并启用规则集。",
                status_code=409,
            )

        background_tasks.add_task(
            active_evidence_review_web_service.execute_run,
            run.run_id,
        )
        return RedirectResponse(
            url=(
                f"/contracts/{contract_id}/evidence-review"
                f"?run_id={run.run_id}"
            ),
            status_code=303,
        )

    @application.post(
        "/contracts/{contract_id}/evidence-review/runs/{run_id}"
        "/items/{item_id}/decision",
        response_class=HTMLResponse,
    )
    def save_evidence_review_decision(
        request: Request,
        contract_id: str,
        run_id: str,
        item_id: str,
        decision: str = Form(default=""),
        risk_level: str = Form(default=""),
        opinion: str = Form(default=""),
        research_status: str = Form(...),
        research_notes: str = Form(default=""),
        expected_updated_at: str = Form(...),
    ) -> Response:
        submitted = {
            "decision": decision,
            "risk_level": risk_level,
            "opinion": opinion,
            "research_status": research_status,
            "research_notes": research_notes,
            "expected_updated_at": expected_updated_at,
        }
        try:
            active_evidence_review_web_service.get_run_payload(
                contract_id,
                run_id,
            )
            active_evidence_review_web_service.save_human_decision(
                run_id,
                item_id,
                {
                    "human_review_status": (
                        "completed" if decision else "pending"
                    ),
                    "decision": decision or None,
                    "risk_level": risk_level or None,
                    "opinion": opinion,
                    "research_status": research_status,
                    "research_notes": research_notes,
                },
                expected_updated_at,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="evidence review item not found",
            ) from exc
        except DecisionConflictError:
            return render_evidence_review_page(
                request,
                contract_id,
                run_id=run_id,
                error="页面数据已更新，请核对最新内容后重新提交。",
                status_code=409,
                decision_item_id=item_id,
                decision_input=submitted,
            )
        except ValueError as exc:
            message = str(exc)
            if "pending research" in message:
                public_message = (
                    "外部或内部查询尚未完成，只能选择“暂无法判断”"
                    "或继续待审核。"
                )
            elif "opinion" in message:
                public_message = "完成人工判断时必须填写人工意见。"
            elif "risk_level" in message:
                public_message = "只有“有风险”结论可以填写风险等级。"
            else:
                public_message = "人工判断内容不完整，请检查后重试。"
            return render_evidence_review_page(
                request,
                contract_id,
                run_id=run_id,
                error=public_message,
                status_code=400,
                decision_item_id=item_id,
                decision_input=submitted,
            )

        return RedirectResponse(
            url=(
                f"/contracts/{contract_id}/evidence-review"
                f"?run_id={run_id}#item-{item_id}"
            ),
            status_code=303,
        )

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
            url=f"/contracts/{contract_id}/review/runs/{run.run_id}/rules",
            status_code=303,
        )

    @application.get(
        "/contracts/{contract_id}/review/runs/{run_id}/rules",
        response_class=HTMLResponse,
    )
    def review_rules_page(
        request: Request, contract_id: str, run_id: str
    ) -> Response:
        contract = active_service.get_contract(contract_id)
        if contract is None:
            raise HTTPException(status_code=404, detail="contract not found")
        try:
            run = active_contract_review_web_service.get_run_payload(
                contract_id, run_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="review run not found") from exc
        if run["status"] == "ready" or (
            run["status"] == "processing"
            and run["progress"].get("stage") not in {
                "parsing_rules", "rules_parsed", "rules_ready"
            }
        ):
            return RedirectResponse(
                url=f"/contracts/{contract_id}/review?run_id={run_id}",
                status_code=303,
            )
        return templates.TemplateResponse(
            request=request,
            name="review_rules.html",
            context={"selected_contract": contract, "review_run": run},
        )

    @application.post("/contracts/{contract_id}/review/runs/{run_id}/continue")
    def continue_contract_review_run(
        contract_id: str,
        run_id: str,
        background_tasks: BackgroundTasks,
    ) -> Response:
        try:
            active_contract_review_web_service.claim_continue_run(
                contract_id, run_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="review run not found") from exc
        except ReviewRunTransitionError as exc:
            raise HTTPException(
                status_code=409, detail="review run is not waiting to continue"
            ) from exc
        background_tasks.add_task(
            active_contract_review_web_service.execute_review, run_id
        )
        return RedirectResponse(
            url=f"/contracts/{contract_id}/review?run_id={run_id}",
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
        retrieval_mode: str = Form(default="vector"),
    ) -> Response:
        try:
            run = active_service.create_single_evaluation_run(
                contract_id,
                case_id,
                retrieval_mode=retrieval_mode,
            )
        except (
            EvaluationContractNotFoundError,
            EvaluationCaseNotFoundError,
        ) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EvaluationContractNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EvaluationStaleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        retrieval_mode: str = Form(default="vector"),
    ) -> Response:
        try:
            run = active_service.create_all_evaluation_run(
                contract_id,
                retrieval_mode=retrieval_mode,
            )
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

    return application
