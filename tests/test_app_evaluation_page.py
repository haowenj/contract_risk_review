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
os.environ.setdefault("LLM_RERANK_MODEL", "test-reranker")

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
    assert "/contracts/c1/evaluation/metadata" in page_response.text
    assert "查看解析对象" in page_response.text
    assert "查看检索上下文" in page_response.text
    assert "/contracts/c1/evaluation/retrieval-context" in page_response.text
    assert ">召回测试<" in home_response.text
    assert "evaluationLink.textContent = '召回测试';" in home_response.text
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


def test_config_route_returns_bad_request_when_save_fails_validation():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.save_evaluation_cases.side_effect = ValueError("问题不能为空")

        response = client.post(
            "/contracts/c1/evaluation/config",
            data={
                "question": ["付款方式？"],
                "expected_source_object_indices": ["111"],
            },
            follow_redirects=False,
        )

    assert response.status_code == 400
    assert "问题不能为空" in response.text


def test_metadata_page_shows_complete_source_objects_with_indices():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.list_source_object_entries.return_value = [
            {
                "source_object_index": 7,
                "object": {"type": "text", "text": "完整元数据内容"},
            }
        ]

        response = client.get("/contracts/c1/evaluation/metadata")

    assert response.status_code == 200
    assert "解析对象" in response.text
    assert "source_object_index = 7" in response.text
    assert "完整元数据内容" in response.text


def test_retrieval_context_page_shows_saved_contexts_with_indices():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.list_retrieval_context_entries.return_value = [
            {
                "source_object_index": 7,
                "retrieval_context": "文档章节：付款条款",
            }
        ]

        response = client.get("/contracts/c1/evaluation/retrieval-context")

    assert response.status_code == 200
    assert "检索上下文" in response.text
    assert "source_object_index = 7" in response.text
    assert "文档章节：付款条款" in response.text


def test_evaluation_page_renders_items_from_latest_run_payload():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.latest_evaluation_run_payload.return_value = {
            "run_id": "run-1",
            "status": "ready",
            "items": [
                {
                    "question": "测试问题",
                    "expected_source_object_indices": [7],
                    "result": {
                        "vector_source_object_indices": [7],
                        "rerank_source_object_indices": [7],
                        "vector_recall_at_5": 1.0,
                        "vector_recall_at_10": 1.0,
                        "rerank_recall_at_5": 1.0,
                        "rerank_recall_at_10": 1.0,
                        "vector_results": [],
                        "reranked_results": [],
                        "selected_nodes": [],
                        "llm_summary": {"answer": "测试答案"},
                    },
                }
            ],
        }

        response = client.get("/contracts/c1/evaluation")

    assert response.status_code == 200
    assert "测试问题" in response.text
    assert "测试答案" in response.text


def test_evaluation_page_renders_pending_items_before_background_run_finishes():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        service.get_evaluation_run_payload.return_value = {
            "run_id": "run-processing",
            "status": "processing",
            "items": [
                {
                    "question": "尚未完成的问题",
                    "expected_source_object_indices": [7],
                    "result": {},
                }
            ],
        }

        response = client.get("/contracts/c1/evaluation?run_id=run-processing")

    assert response.status_code == 200
    assert "尚未完成的问题" in response.text
    assert "等待测试完成" in response.text


def test_evaluation_page_renders_image_fields_in_all_retrieval_stages():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        image = {
            "node_type": "image",
            "source_object_index": 12,
            "page_idx": 4,
            "img_path": "images/account.jpg",
            "image_type": "bank_account",
            "structured_data": {"account_number": "110914414810101"},
            "verification_status": "verified",
            "evidence_text": "银行账号：110914414810101",
        }
        service.latest_evaluation_run_payload.return_value = {
            "run_id": "run-image",
            "status": "ready",
            "items": [
                {
                    "question": "账号？",
                    "expected_source_object_indices": [12],
                    "result": {
                        "vector_source_object_indices": [12],
                        "rerank_source_object_indices": [12],
                        "vector_recall_at_5": 1.0,
                        "vector_recall_at_10": 1.0,
                        "rerank_recall_at_5": 1.0,
                        "rerank_recall_at_10": 1.0,
                        "vector_results": [image],
                        "reranked_results": [image],
                        "selected_nodes": [image],
                        "llm_summary": {"answer": "账号为110914414810101。"},
                    },
                }
            ],
        }

        response = client.get("/contracts/c1/evaluation")

    assert response.status_code == 200
    assert response.text.count("图片证据") >= 3
    assert response.text.count("images/account.jpg") >= 3
    assert response.text.count("verified") >= 3


def test_evaluation_page_renders_table_image_in_all_retrieval_stages():
    with TemporaryDirectory() as temp_dir:
        client, service = build_client(Path(temp_dir))
        table = {
            "node_type": "table",
            "source_object_index": 7,
            "page_idx": 2,
            "img_path": "images/payment-table.jpg",
            "image_type": "general",
            "structured_data": {
                "visible_text": "付款比例30%",
                "content_description": "付款计划表",
            },
            "verification_status": "not_required",
            "evidence_text": "第1行：付款比例 | 30%",
        }
        service.latest_evaluation_run_payload.return_value = {
            "run_id": "run-table",
            "status": "ready",
            "items": [
                {
                    "question": "付款比例？",
                    "expected_source_object_indices": [7],
                    "result": {
                        "vector_source_object_indices": [7],
                        "rerank_source_object_indices": [7],
                        "vector_recall_at_5": 1.0,
                        "vector_recall_at_10": 1.0,
                        "rerank_recall_at_5": 1.0,
                        "rerank_recall_at_10": 1.0,
                        "vector_results": [table],
                        "reranked_results": [table],
                        "selected_nodes": [table],
                        "llm_summary": {"answer": "付款比例为30%。"},
                    },
                }
            ],
        }

        response = client.get("/contracts/c1/evaluation")

    assert response.status_code == 200
    assert response.text.count("表格原图") >= 3
    assert response.text.count("images/payment-table.jpg") >= 3
    assert response.text.count("付款计划表") >= 3


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
