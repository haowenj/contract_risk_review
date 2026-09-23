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
from app.evaluation_db import EvaluationRepository
from app.evidence_review.repository import EvidenceReviewRepository
from app.review_db import ContractReviewRepository
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

    def test_reprocess_contract_reuses_existing_parse_and_schedules_processor(self):
        with TemporaryDirectory() as temp_dir:
            app, service = self._build(Path(temp_dir))
            contract_dir = Path(temp_dir) / "contract"
            contract_dir.mkdir(parents=True)
            (contract_dir / "raw_content_list.json").write_text("[]", encoding="utf-8")
            contract = service.repository.create(
                "contract.pdf",
                contract_dir,
            )
            service.repository.update_status(
                contract.contract_id,
                "ready",
                index_version="index-v1",
            )

            response = TestClient(app).post(
                f"/api/contracts/{contract.contract_id}/reprocess",
                json={"mode": "reuse_existing"},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        service.processor.process.assert_called_once_with(
            contract.contract_id,
            mode="reuse_existing",
        )

    def test_delete_finished_contract_removes_files_record_and_review_history(self):
        with TemporaryDirectory() as temp_dir:
            app, service = self._build(Path(temp_dir))
            contract = service.create_upload("contract.pdf", b"%PDF-test")
            service.repository.update_status(
                contract.contract_id, "ready", index_version="v1"
            )
            history = EvidenceReviewRepository(service.settings.database_path)
            run = history.create_evidence_run(
                contract_id=contract.contract_id,
                rule_set_id="rule-1",
                rule_snapshot={"review_items": []},
            )
            history.mark_evidence_run_processing(run.run_id)
            history.complete_evidence_run(run.run_id, [], progress={"completed": 0})
            old_review = ContractReviewRepository(service.settings.database_path)
            old_run = old_review.create_run(contract.contract_id, "测试规则")
            old_review.mark_processing(old_run.run_id, {"stage": "parsing_rules"})
            old_review.mark_failed(old_run.run_id, "结束")
            evaluations = EvaluationRepository(service.settings.database_path)
            evaluation = evaluations.create_run(
                contract.contract_id,
                "all",
                "v1",
                {"pipeline_version": "v1"},
                [],
            )
            evaluations.mark_ready(evaluation.run_id)
            service.index_manager.put = Mock()
            service.index_manager.clear = Mock()
            client = TestClient(app)

            response = client.delete(f"/api/contracts/{contract.contract_id}")

            self.assertEqual(response.status_code, 204)
            self.assertIsNone(service.repository.get(contract.contract_id))
            self.assertFalse(Path(contract.storage_dir).exists())
            self.assertIsNone(history.get_evidence_run(run.run_id))
            self.assertIsNone(old_review.get_run(old_run.run_id))
            self.assertIsNone(evaluations.get_run(evaluation.run_id))
            service.index_manager.clear.assert_called_once_with(contract.contract_id)
            self.assertEqual(client.get(f"/api/contracts/{contract.contract_id}").status_code, 404)

    def test_delete_rejects_processing_contract_and_active_review(self):
        with TemporaryDirectory() as temp_dir:
            app, service = self._build(Path(temp_dir))
            contract = service.create_upload("contract.pdf", b"%PDF-test")
            client = TestClient(app)
            endpoint = f"/api/contracts/{contract.contract_id}"
            self.assertEqual(client.delete(endpoint).status_code, 409)
            service.repository.update_status(contract.contract_id, "ready", index_version="v1")
            history = EvidenceReviewRepository(service.settings.database_path)
            run = history.create_evidence_run(
                contract_id=contract.contract_id,
                rule_set_id="rule-1",
                rule_snapshot={"review_items": []},
            )

            self.assertEqual(client.delete(endpoint).status_code, 409)
            self.assertIsNotNone(service.repository.get(contract.contract_id))
            self.assertTrue(Path(contract.storage_dir).exists())
            self.assertIsNotNone(history.get_evidence_run(run.run_id))

    def test_delete_rejects_unmanaged_storage_path(self):
        with TemporaryDirectory() as temp_dir:
            app, service = self._build(Path(temp_dir))
            outside = Path(temp_dir) / "outside"
            outside.mkdir()
            (outside / "keep.txt").write_text("keep", encoding="utf-8")
            contract = service.repository.create("contract.pdf", outside)
            service.repository.update_status(contract.contract_id, "ready", index_version="v1")

            response = TestClient(app).delete(f"/api/contracts/{contract.contract_id}")

            self.assertEqual(response.status_code, 409)
            self.assertEqual((outside / "keep.txt").read_text(), "keep")
            self.assertIsNotNone(service.repository.get(contract.contract_id))

    def test_chat_routes_are_removed(self):
        with TemporaryDirectory() as temp_dir:
            app, service = self._build(Path(temp_dir))
            contract = service.repository.create(
                "contract.pdf",
                Path(temp_dir) / "contract",
            )
            service.repository.update_status(contract.contract_id, "ready")
            client = TestClient(app)

            html_response = client.get(f"/contracts/{contract.contract_id}")
            form_response = client.post(
                "/chat",
                data={"contract_id": contract.contract_id, "question": "问题"},
            )
            api_response = client.post(
                f"/api/contracts/{contract.contract_id}/chat",
                json={"question": "问题"},
            )

        self.assertEqual(html_response.status_code, 404)
        self.assertEqual(form_response.status_code, 404)
        self.assertEqual(api_response.status_code, 404)
