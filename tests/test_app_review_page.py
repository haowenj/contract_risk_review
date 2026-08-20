import os
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")
os.environ.setdefault("LLM_MODEL", "test-answer-model")
os.environ.setdefault("LLM_RERANK_MODEL", "test-reranker")

from app.api import create_app
from app.config import Settings
from app.db import ContractRepository
from app.review_db import ContractReviewRepository
from app.review_service import ContractReviewWebService

REVIEW_RESULT = {
    "contract_id": "c1",
    "review_rule_text": "审查规范",
    "review_items": [
        {
            "id": "item_1",
            "name": "分包转包限制",
            "rule_basis": "不得违法分包",
            "review_goal": "核验限制条款",
            "retrieval_query": "内部查询不应展示",
        }
    ],
    "current_item_index": 3,
    "summary": {
        "total_items": 3,
        "risk_count": 1,
        "high_risk_count": 0,
        "medium_risk_count": 1,
        "low_risk_count": 0,
        "no_obvious_risk_count": 1,
        "needs_review_count": 1,
    },
    "review_results": [
        {
            "item_id": "item_1",
            "item_name": "分包转包限制",
            "risk_status": "risk",
            "risk_level": "medium",
            "evidence_status": "absence_verified",
            "finding": "未发现明确限制条款。",
            "risk_description": "存在违法分包风险。",
            "suggestion": "补充分包转包限制。",
            "evidence": [],
            "absence_check": {
                "primary_keywords": ["分包", "转包"],
                "secondary_keywords": ["第三方"],
                "candidate_count": 0,
            },
        },
        {
            "item_id": "item_2",
            "item_name": "付款期限",
            "risk_status": "no_obvious_risk",
            "risk_level": None,
            "evidence_status": "found",
            "finding": "付款期限为30日。",
            "risk_description": "未发现明显风险。",
            "suggestion": "维持现有约定。",
            "evidence": [
                {
                    "page_idx": 4,
                    "source_object_index": 27,
                    "node_type": "table",
                    "evidence_text": "付款期限：30日",
                    "score": 0.99,
                }
            ],
            "absence_check": None,
        },
        {
            "item_id": "item_3",
            "item_name": "知识产权归属",
            "risk_status": "needs_review",
            "risk_level": None,
            "evidence_status": "insufficient",
            "finding": "证据不足。",
            "risk_description": "无法可靠判断。",
            "suggestion": "请人工复核。",
            "evidence": [],
            "absence_check": None,
        },
    ],
}


class RecoveryStub:
    def __init__(self):
        self.calls = 0

    def recover_interrupted_runs(self):
        self.calls += 1


class ActiveContractService:
    def __init__(self, repository):
        self.repository = repository
        self.evaluation_service = RecoveryStub()

    def get_contract(self, contract_id):
        return self.repository.get(contract_id)

    def list_contracts(self):
        return self.repository.list()


class SuccessfulReviewService:
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback

    def run(self, contract_id, review_rule_text):
        self.progress_callback(
            "review_items_parsed",
            {"review_items": [{"id": "item_1", "name": "付款期限"}]},
        )
        self.progress_callback(
            "review_item_started",
            {
                "current_item_index": 0,
                "item": {"id": "item_1", "name": "付款期限"},
            },
        )
        self.progress_callback(
            "review_item_completed",
            {"result": {"item_name": "付款期限"}},
        )
        result = dict(REVIEW_RESULT)
        result["contract_id"] = contract_id
        result["review_rule_text"] = review_rule_text
        return result


def settings_for(root: Path) -> Settings:
    return Settings(
        project_dir=root,
        data_dir=root / "data",
        database_path=root / "data" / "contracts.db",
        contracts_dir=root / "data" / "contracts",
        mineru_url="http://mineru.test",
        mineru_backend="hybrid-engine",
        mineru_server_url=None,
    )


def build_client(root: Path, *, status="ready"):
    settings = settings_for(root)
    contracts = ContractRepository(settings.database_path)
    contract = contracts.create("contract.pdf", root / "contract", contract_id="c1")
    contracts.update_status(
        contract.contract_id,
        status,
        index_version="index-v1" if status == "ready" else None,
    )
    active_service = ActiveContractService(contracts)
    review_repository = ContractReviewRepository(settings.database_path)

    def factory(*, contract_service, progress_callback):
        assert contract_service is active_service
        return SuccessfulReviewService(progress_callback)

    web_service = ContractReviewWebService(
        contract_service=active_service,
        review_repository=review_repository,
        review_service_factory=factory,
    )
    application = create_app(
        settings=settings,
        service=active_service,
        contract_review_web_service=web_service,
    )
    return TestClient(application), contracts, review_repository, web_service


def run_id_from(response):
    return parse_qs(urlparse(response.headers["location"]).query)["run_id"][0]


def test_ready_review_page_shows_text_and_txt_md_inputs():
    with TemporaryDirectory() as temp_dir:
        client, _, _, _ = build_client(Path(temp_dir))

        response = client.get("/contracts/c1/review")

    assert response.status_code == 200
    assert 'name="review_rule_text"' in response.text
    assert 'name="review_rule_file"' in response.text
    assert 'accept=".txt,.md,text/plain,text/markdown"' in response.text
    assert "只选择一种输入方式" in response.text


def test_review_page_includes_back_to_top_button():
    with TemporaryDirectory() as temp_dir:
        client, _, _, _ = build_client(Path(temp_dir))

        response = client.get("/contracts/c1/review")

    assert response.status_code == 200
    assert 'id="back-to-top"' in response.text
    assert 'aria-label="回到顶部"' in response.text
    assert "position: fixed" in response.text
    assert "window.scrollTo" in response.text
    assert "prefers-reduced-motion" in response.text


def test_text_submission_creates_run_executes_in_background_and_redirects():
    with TemporaryDirectory() as temp_dir:
        client, _, repository, _ = build_client(Path(temp_dir))

        response = client.post(
            "/contracts/c1/review/runs",
            data={"review_rule_text": "  付款期限不得超过90日  "},
            follow_redirects=False,
        )
        run_id = run_id_from(response)
        run = repository.get_run(run_id)

    assert response.status_code == 303
    assert response.headers["location"] == f"/contracts/c1/review?run_id={run_id}"
    assert run.status == "ready"
    assert run.review_rule_text == "付款期限不得超过90日"
    assert run.result["contract_id"] == "c1"


@pytest.mark.parametrize("filename", ["rules.txt", "rules.md"])
def test_txt_and_md_uploads_are_read_as_review_rule_text(filename):
    with TemporaryDirectory() as temp_dir:
        client, _, repository, _ = build_client(Path(temp_dir))

        response = client.post(
            "/contracts/c1/review/runs",
            data={"review_rule_text": ""},
            files={
                "review_rule_file": (
                    filename,
                    "\ufeff上传的审查规范".encode(),
                    "text/plain",
                )
            },
            follow_redirects=False,
        )
        run = repository.get_run(run_id_from(response))

    assert response.status_code == 303
    assert run.review_rule_text == "上传的审查规范"


@pytest.mark.parametrize(
    ("data", "files", "expected_message"),
    [
        ({"review_rule_text": ""}, None, "请输入或上传审查规范"),
        (
            {"review_rule_text": "手动规范"},
            {"review_rule_file": ("rules.txt", b"file rules", "text/plain")},
            "请只选择一种审查规范输入方式",
        ),
        (
            {"review_rule_text": ""},
            {"review_rule_file": ("rules.pdf", b"pdf", "application/pdf")},
            "审查规范文件仅支持 .txt 或 .md",
        ),
        (
            {"review_rule_text": ""},
            {"review_rule_file": ("rules.txt", b"\xff", "text/plain")},
            "审查规范文件必须使用 UTF-8 编码",
        ),
    ],
)
def test_submission_validation_returns_server_rendered_error(
    data,
    files,
    expected_message,
):
    with TemporaryDirectory() as temp_dir:
        client, _, _, _ = build_client(Path(temp_dir))

        response = client.post(
            "/contracts/c1/review/runs",
            data=data,
            files=files,
            follow_redirects=False,
        )

    assert response.status_code == 400
    assert expected_message in response.text


def test_non_ready_contract_cannot_create_review_run():
    with TemporaryDirectory() as temp_dir:
        client, _, repository, _ = build_client(Path(temp_dir), status="processing")

        page = client.get("/contracts/c1/review")
        response = client.post(
            "/contracts/c1/review/runs",
            data={"review_rule_text": "审查规范"},
            follow_redirects=False,
        )
        with sqlite3.connect(repository.database_path) as connection:
            run_count = connection.execute(
                "SELECT COUNT(*) FROM review_runs"
            ).fetchone()[0]

    assert page.status_code == 200
    assert "完成入库后才能进行风险评估" in page.text
    assert response.status_code == 409
    assert "完成入库后才能进行风险评估" in response.text
    assert run_count == 0


def test_page_and_api_hide_run_owned_by_another_contract():
    with TemporaryDirectory() as temp_dir:
        client, contracts, repository, _ = build_client(Path(temp_dir))
        other = contracts.create("other.pdf", Path(temp_dir) / "other", contract_id="c2")
        contracts.update_status(other.contract_id, "ready", index_version="index-v2")
        run = repository.create_run("c1", "审查规范")

        page = client.get(f"/contracts/c2/review?run_id={run.run_id}")
        api = client.get(f"/api/contracts/c2/review/runs/{run.run_id}")

    assert page.status_code == 404
    assert api.status_code == 404


def test_processing_page_and_status_api_expose_only_business_progress():
    with TemporaryDirectory() as temp_dir:
        client, _, repository, _ = build_client(Path(temp_dir))
        run = repository.create_run("c1", "审查规范")
        repository.mark_processing(
            run.run_id,
            {
                "stage": "reviewing",
                "message": "正在审查 2 / 3：付款期限",
                "current": 2,
                "total": 3,
                "item_name": "付款期限",
            },
        )

        page = client.get(f"/contracts/c1/review?run_id={run.run_id}")
        api = client.get(f"/api/contracts/c1/review/runs/{run.run_id}")

    assert page.status_code == 200
    assert "正在审查 2 / 3：付款期限" in page.text
    assert api.status_code == 200
    assert api.json()["progress"]["stage"] == "reviewing"
    assert api.json()["result"] == {}


def test_ready_page_renders_summary_results_evidence_and_absence_audit():
    with TemporaryDirectory() as temp_dir:
        client, _, repository, _ = build_client(Path(temp_dir))
        run = repository.create_run("c1", "审查规范")
        repository.mark_processing(
            run.run_id,
            {"stage": "parsing_rules", "message": "正在解析审查规范"},
        )
        repository.mark_ready(
            run.run_id,
            REVIEW_RESULT,
            {
                "stage": "completed",
                "message": "已完成 3 / 3",
                "current": 3,
                "total": 3,
            },
        )

        response = client.get(f"/contracts/c1/review?run_id={run.run_id}")
        api = client.get(f"/api/contracts/c1/review/runs/{run.run_id}")

    assert response.status_code == 200
    for value in ["总审查项", "风险", "未发现明显风险", "需人工复核"]:
        assert value in response.text
    for value in [
        "分包转包限制",
        "付款期限",
        "知识产权归属",
        "未分级",
        "付款期限：30日",
        "page_idx：4",
        "source_object_index：27",
        "node_type：table",
        "primary_keywords",
        "分包、转包",
        "secondary_keywords",
        "第三方",
        "candidate_count：0",
    ]:
        assert value in response.text
    assert "内部查询不应展示" not in response.text
    assert "0.99" not in response.text
    assert "Rerank Top3" not in response.text
    assert api.status_code == 200
    assert "review_rule_text" not in api.json()
    assert "review_items" not in api.json()["result"]
    assert "score" not in str(api.json()["result"])
    assert "内部查询不应展示" not in str(api.json())


def test_failed_run_page_displays_saved_error():
    with TemporaryDirectory() as temp_dir:
        client, _, repository, _ = build_client(Path(temp_dir))
        run = repository.create_run("c1", "审查规范")
        repository.mark_processing(
            run.run_id,
            {"stage": "parsing_rules", "message": "正在解析审查规范"},
        )
        repository.mark_failed(run.run_id, "review model unavailable")

        response = client.get(f"/contracts/c1/review?run_id={run.run_id}")

    assert response.status_code == 200
    assert "风险审查执行失败，请查看服务日志" in response.text
    assert "review model unavailable" not in response.text


def test_app_creation_recovers_interrupted_review_runs():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        settings = settings_for(root)
        contracts = ContractRepository(settings.database_path)
        contract = contracts.create("contract.pdf", root / "contract", contract_id="c1")
        contracts.update_status(contract.contract_id, "ready", index_version="index-v1")
        repository = ContractReviewRepository(settings.database_path)
        run = repository.create_run("c1", "审查规范")
        active_service = ActiveContractService(contracts)
        web_service = ContractReviewWebService(
            contract_service=active_service,
            review_repository=repository,
        )

        create_app(
            settings=settings,
            service=active_service,
            contract_review_web_service=web_service,
        )
        recovered = repository.get_run(run.run_id)

    assert recovered.status == "failed"
    assert recovered.error_message == "服务重启导致风险审查任务中断"


def test_app_enables_file_diagnostics_under_application_data_directory():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        settings = settings_for(root)
        contracts = ContractRepository(settings.database_path)
        active_service = ActiveContractService(contracts)

        with patch("app.api.ContractReviewWebService") as service_class:
            create_app(settings=settings, service=active_service)

    service_class.assert_called_once_with(
        contract_service=active_service,
        review_repository=ANY,
        review_runs_dir=settings.data_dir / "review_runs",
    )
