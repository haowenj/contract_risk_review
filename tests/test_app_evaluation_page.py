import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")
os.environ.setdefault("LLM_MODEL", "test-answer-model")
os.environ.setdefault("LLM_RERANK_MODEL", "qwen3-rerank")

from app.config import Settings
from main import create_app


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


def build_client(root: Path, *, status: str = "ready"):
    contract = SimpleNamespace(
        contract_id="c1",
        filename="contract.pdf",
        storage_dir=str(root / "contract"),
        index_version="index-v1" if status == "ready" else None,
        status=status,
        error_message=None,
        created_at="2026-08-17T00:00:00+00:00",
        updated_at="2026-08-17T00:00:00+00:00",
    )
    service = Mock()
    service.evaluation_service = Mock()
    service.list_contracts.return_value = [contract]
    service.get_contract.return_value = contract
    service.list_evaluation_cases.return_value = []
    service.default_evaluation_cases.return_value = [("默认问题", [111, 112])]
    service.latest_evaluation_run_payload.return_value = None
    return TestClient(create_app(settings=settings_for(root), service=service)), service


def test_ready_contract_has_evaluation_entry_and_page_shows_default_case():
    with TemporaryDirectory() as temp_dir:
        client, _ = build_client(Path(temp_dir))

        home_response = client.get("/")
        page_response = client.get("/contracts/c1/evaluation")

    assert home_response.status_code == 200
    assert "/contracts/c1/evaluation" in home_response.text
    assert page_response.status_code == 200
    assert "默认问题" in page_response.text
    assert 'name="question"' in page_response.text
    assert 'name="expected_source_object_indices"' in page_response.text


def test_config_route_parses_numeric_ids_and_redirects():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))

        response = client.post(
            "/contracts/c1/evaluation/config",
            data={
                "question": ["付款方式？"],
                "expected_source_object_indices": ["111, 112\n113"],
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    service.save_evaluation_cases.assert_called_once_with(
        "c1",
        [("付款方式？", [111, 112, 113])],
    )


def test_all_run_route_schedules_background_execution():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.create_all_evaluation_run.return_value = SimpleNamespace(
            run_id="run-1"
        )

        response = client.post(
            "/contracts/c1/evaluation/run-all",
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert "run_id=run-1" in response.headers["location"]
    service.execute_evaluation_run.assert_called_once_with("run-1")


def test_single_run_route_schedules_requested_case():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.create_single_evaluation_run.return_value = SimpleNamespace(
            run_id="run-single"
        )

        response = client.post(
            "/contracts/c1/evaluation/cases/7/run",
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert "run_id=run-single" in response.headers["location"]
    service.create_single_evaluation_run.assert_called_once_with("c1", 7)
    service.execute_evaluation_run.assert_called_once_with("run-single")


def test_run_status_endpoint_returns_serialized_results():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.get_evaluation_run_payload.return_value = {
            "run_id": "run-1",
            "contract_id": "c1",
            "status": "ready",
            "items": [
                {
                    "case_id": 7,
                    "question": "问题",
                    "expected_source_object_indices": [111],
                    "result": {
                        "vector_source_object_indices": [111],
                        "rerank_recall_at_10": 1.0,
                    },
                }
            ],
        }

        response = client.get(
            "/api/contracts/c1/evaluation/runs/run-1"
        )

    assert response.status_code == 200
    assert response.json()["items"][0]["result"]["rerank_recall_at_10"] == 1.0


def test_parse_expected_indices_accepts_common_separators_and_rejects_text():
    from app.evaluation_forms import parse_expected_indices

    assert parse_expected_indices("111, 112\n112 113") == [111, 112, 113]
    with pytest.raises(ValueError, match="数字"):
        parse_expected_indices("111, node-2")
