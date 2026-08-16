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


class AppAPITest(TestCase):
    def _build(self, root: Path):
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
        processor = Mock()
        service = ContractService(repository, settings, processor, Mock())
        return create_app(settings=settings, service=service), service

    def test_upload_rejects_non_pdf(self):
        with TemporaryDirectory() as temp_dir:
            app, _ = self._build(Path(temp_dir))
            response = TestClient(app).post(
                "/api/contracts",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )

        self.assertEqual(response.status_code, 400)

    def test_upload_creates_contract_and_schedules_processor(self):
        with TemporaryDirectory() as temp_dir:
            app, service = self._build(Path(temp_dir))
            response = TestClient(app).post(
                "/api/contracts",
                files={"file": ("contract.pdf", b"%PDF-test", "application/pdf")},
            )
            self.assertEqual(response.status_code, 202)
            payload = response.json()
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(Path(payload["storage_dir"]).name, payload["contract_id"])
            self.assertTrue(Path(payload["storage_dir"]).joinpath("source.pdf").is_file())
            service.processor.process.assert_called_once_with(payload["contract_id"])

    def test_chat_returns_answer_from_service(self):
        with TemporaryDirectory() as temp_dir:
            app, service = self._build(Path(temp_dir))
            contract = service.repository.create("contract.pdf", Path(temp_dir) / "contract")
            service.repository.update_status(contract.contract_id, "ready")
            service.ask = Mock(
                return_value={
                    "contract_id": contract.contract_id,
                    "question": "问题",
                    "answer": "答案",
                    "evidence": [],
                    "debug": None,
                }
            )

            response = TestClient(app).post(
                f"/api/contracts/{contract.contract_id}/chat",
                json={"question": "问题"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "答案")
        service.ask.assert_called_once_with(contract.contract_id, "问题", False)

    def test_chat_rejects_missing_contract(self):
        with TemporaryDirectory() as temp_dir:
            app, _ = self._build(Path(temp_dir))
            response = TestClient(app).post(
                "/api/contracts/missing/chat",
                json={"question": "问题"},
            )

        self.assertEqual(response.status_code, 404)
