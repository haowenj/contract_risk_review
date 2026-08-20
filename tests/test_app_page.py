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
            self.assertNotIn('id="open-upload"', response.text)
            self.assertIn("contract.pdf", response.text)
            self.assertIn("合同任务", response.text)
            self.assertIn(">排队中<", response.text)
            self.assertNotIn("MinerU · 合同任务队列", response.text)
            self.assertNotIn('name="question"', response.text)
            self.assertNotIn("请选择一份合同", response.text)
            self.assertIn('data-poll-interval="10000"', response.text)

    def test_ready_contract_shows_evaluation_entry_without_chat_module(self):
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
                f"/contracts/{contract.contract_id}/review",
                home_response.text,
            )
            self.assertIn("风险评估", home_response.text)
            self.assertIn("召回测试", home_response.text)
            self.assertIn("重新解析", home_response.text)
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
