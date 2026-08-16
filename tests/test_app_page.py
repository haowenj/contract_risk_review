import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")
os.environ.setdefault("LLM_MODEL", "test-answer-model")
os.environ.setdefault("LLM_RERANK_MODEL", "qwen3-rerank")

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import ContractRepository
from app.service import ContractService
from main import create_app


class ServerRenderedPageTest(TestCase):
    def test_home_page_contains_upload_contract_list_and_chat_form(self):
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
                f"/?contract_id={contract.contract_id}"
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn('enctype="multipart/form-data"', response.text)
            self.assertIn("contract.pdf", response.text)
            self.assertIn('name="question"', response.text)

    def test_page_renders_debug_sections_from_chat_result(self):
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
            contract_dir = root / "contract"
            contract = repository.create("contract.pdf", contract_dir)
            repository.update_status(contract.contract_id, "ready")
            service = Mock()
            service.list_contracts.return_value = [repository.get(contract.contract_id)]
            service.get_contract.return_value = repository.get(contract.contract_id)
            service.ask.return_value = {
                "answer": "答案",
                "evidence": [{"text": "证据"}],
                "debug": {
                    "rerank_top10": [{"text": "候选"}],
                    "selected_evidence": [{"text": "证据"}],
                    "final_answer": "答案",
                },
            }
            response = TestClient(create_app(settings=settings, service=service)).post(
                "/chat",
                data={"contract_id": contract.contract_id, "question": "问题", "debug": "on"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("Rerank Top10", response.text)
            self.assertIn("Selected Evidence", response.text)
            self.assertIn("答案", response.text)
