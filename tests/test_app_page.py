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

    def test_ready_contract_opens_independent_chat_page(self):
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
            chat_response = client.get(f"/contracts/{contract.contract_id}")

            self.assertEqual(home_response.status_code, 200)
            self.assertIn(f"/contracts/{contract.contract_id}", home_response.text)
            self.assertIn("开始问答", home_response.text)
            self.assertIn(">可问答<", home_response.text)
            self.assertEqual(chat_response.status_code, 200)
            self.assertIn("状态：可问答", chat_response.text)
            self.assertIn('name="question"', chat_response.text)
            self.assertIn(
                f'action="/contracts/{contract.contract_id}/chat"',
                chat_response.text,
            )

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
                "answer": "**答案**",
                "evidence": [{"text": "证据"}],
                "debug": {
                    "rerank_top10": [{"text": "候选"}],
                    "selected_evidence": [{"text": "证据"}],
                    "final_answer": "**答案**",
                },
            }
            response = TestClient(create_app(settings=settings, service=service)).post(
                f"/contracts/{contract.contract_id}/chat",
                data={"contract_id": contract.contract_id, "question": "问题", "debug": "on"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("Rerank Top10", response.text)
            self.assertIn("Selected Evidence", response.text)
            self.assertIn("<strong>答案</strong>", response.text)
            self.assertNotIn("**答案**", response.text)

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

    def test_page_renders_image_evidence_with_original_reference(self):
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
            (contract_dir / "images").mkdir(parents=True)
            (contract_dir / "images/account.jpg").write_bytes(b"jpeg")
            contract = repository.create("contract.pdf", contract_dir)
            repository.update_status(contract.contract_id, "ready")
            service = Mock()
            service.list_contracts.return_value = [repository.get(contract.contract_id)]
            service.get_contract.return_value = repository.get(contract.contract_id)
            service.ask.return_value = {
                "answer": "账号为110914414810101。",
                "evidence": [
                    {
                        "node_type": "image",
                        "source_object_index": 12,
                        "page_idx": 4,
                        "img_path": "images/account.jpg",
                        "image_type": "bank_account",
                        "structured_data": {"account_number": "110914414810101"},
                        "verification_status": "verified",
                        "evidence_text": "银行账号：110914414810101",
                    }
                ],
                "debug": None,
            }
            client = TestClient(create_app(settings=settings, service=service))

            response = client.post(
                f"/contracts/{contract.contract_id}/chat",
                data={"contract_id": contract.contract_id, "question": "账号？"},
            )
            image_response = client.get(
                f"/contracts/{contract.contract_id}/images/images/account.jpg"
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("图片证据", response.text)
        self.assertIn("images/account.jpg", response.text)
        self.assertIn("verification_status", response.text)
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.content, b"jpeg")
