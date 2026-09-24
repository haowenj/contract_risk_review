import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")
os.environ.setdefault("LLM_MODEL", "test-answer-model")
os.environ.setdefault("LLM_RERANK_MODEL", "test-reranker")

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import ContractRepository
from app.service import ContractService
from main import create_app


class ServerRenderedPageTest(TestCase):
    def test_processing_contract_shows_stage_progress_until_finished(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                project_dir=root,
                data_dir=root / "data",
                database_path=root / "data" / "contracts.db",
                contracts_dir=root / "data" / "contracts",
                mineru_url="http://mineru.test",
                mineru_backend="hybrid-engine",
                mineru_server_url=None,
            )
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", root / "contract")
            service = ContractService(repository, settings, Mock(), Mock())
            client = TestClient(create_app(settings=settings, service=service))

            for stage, label in (
                ("document_parsing", "正在完成文档解析"),
                ("vector_index", "正在构建向量索引"),
                ("bm25_index", "正在构建 BM25 索引"),
            ):
                with self.subTest(stage=stage):
                    repository.update_status(
                        contract.contract_id, "processing", processing_stage=stage
                    )
                    response = client.get("/")
                    self.assertIn(label, response.text)
                    self.assertIn(
                        f'class="stage-progress" data-stage="{stage}"',
                        response.text,
                    )

            repository.update_status(
                contract.contract_id, "ready", processing_stage="completed"
            )
            self.assertNotIn('class="stage-progress"', client.get("/").text)

    def test_home_page_contains_task_list_without_chat_form(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                project_dir=root,
                data_dir=root / "data",
                database_path=root / "data" / "contracts.db",
                contracts_dir=root / "data" / "contracts",
                mineru_url="http://mineru.test",
                mineru_backend="hybrid-engine",
                mineru_server_url=None,
            )
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", root / "contract")
            service = ContractService(repository, settings, Mock(), Mock())
            response = TestClient(create_app(settings=settings, service=service)).get(
                "/"
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn('enctype="multipart/form-data"', response.text)
            self.assertIn('class="upload-zone"', response.text)
            self.assertIn('data-upload-loading', response.text)
            self.assertIn('data-upload-status', response.text)
            self.assertIn('aria-live="polite"', response.text)
            self.assertNotIn('id="open-upload"', response.text)
            self.assertIn("contract.pdf", response.text)
            self.assertIn("合同任务", response.text)
            self.assertIn(">排队中<", response.text)
            self.assertNotIn("MinerU · 合同任务队列", response.text)
            self.assertNotIn('name="question"', response.text)
            self.assertNotIn("请选择一份合同", response.text)
            self.assertIn('data-poll-interval="10000"', response.text)
            self.assertNotIn(
                f'data-delete-contract-id="{contract.contract_id}"',
                response.text,
            )

    def test_ready_contract_shows_rule_selection_without_test_entries(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                project_dir=root,
                data_dir=root / "data",
                database_path=root / "data" / "contracts.db",
                contracts_dir=root / "data" / "contracts",
                mineru_url="http://mineru.test",
                mineru_backend="hybrid-engine",
                mineru_server_url=None,
            )
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", root / "contract")
            repository.update_status(contract.contract_id, "ready")
            service = ContractService(repository, settings, Mock(), Mock())
            client = TestClient(create_app(settings=settings, service=service))

            home_response = client.get("/")
            legacy_page_response = client.get(f"/contracts/{contract.contract_id}")

            self.assertEqual(home_response.status_code, 200)
            self.assertNotIn(f'href="/contracts/{contract.contract_id}"', home_response.text)
            self.assertNotIn("开始问答", home_response.text)
            self.assertIn(
                f"/contracts/{contract.contract_id}/evidence-review",
                home_response.text,
            )
            self.assertIn("选择规则并校验", home_response.text)
            self.assertNotIn("召回测试", home_response.text)
            self.assertNotIn("实验性自动判断", home_response.text)
            self.assertIn("重新解析", home_response.text)
            self.assertIn(
                f'data-delete-contract-id="{contract.contract_id}"',
                home_response.text,
            )
            self.assertIn("删除合同", home_response.text)
            self.assertIn(">已就绪<", home_response.text)
            self.assertNotIn("可问答", home_response.text)
            self.assertEqual(legacy_page_response.status_code, 404)

    def test_failed_contract_shows_reprocess_action_and_chinese_status(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                project_dir=root,
                data_dir=root / "data",
                database_path=root / "data" / "contracts.db",
                contracts_dir=root / "data" / "contracts",
                mineru_url="http://mineru.test",
                mineru_backend="hybrid-engine",
                mineru_server_url=None,
            )
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", root / "contract")
            repository.update_status(contract.contract_id, "failed", "MinerU down")
            service = ContractService(repository, settings, Mock(), Mock())

            response = TestClient(create_app(settings=settings, service=service)).get(
                "/"
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(">失败<", response.text)
        self.assertIn("重新解析", response.text)
        self.assertNotIn("重试", response.text)
        self.assertIn(
            f'data-reprocess-contract-id="{contract.contract_id}"',
            response.text,
        )
        self.assertIn(
            f'data-delete-contract-id="{contract.contract_id}"',
            response.text,
        )
