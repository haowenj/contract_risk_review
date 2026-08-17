from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from app.config import Settings
from app.db import ContractRepository
from app.service import ContractService
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


def test_chat_service_uses_injected_shared_pipeline():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        settings = settings_for(root)
        repository = ContractRepository(settings.database_path)
        contract = repository.create("contract.pdf", root / "contract")
        repository.update_status(
            contract.contract_id,
            "ready",
            index_version="index-v1",
        )
        index_manager = Mock()
        index_manager.get.return_value = object()
        pipeline = Mock()
        pipeline.run.return_value = {
            "query": "问题",
            "vector_results": [],
            "reranked_results": [],
            "selected_nodes": [],
            "selected_indices": [],
            "llm_summary": {"answer": "答案", "evidence_indices": []},
        }
        service = ContractService(
            repository,
            settings,
            Mock(),
            index_manager,
            rag_pipeline=pipeline,
        )

        service.ask(contract.contract_id, "问题")

    pipeline.run.assert_called_once()


def test_app_creation_recovers_interrupted_runs():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        settings = settings_for(root)
        evaluation_service = Mock()
        service = Mock()
        service.evaluation_service = evaluation_service

        create_app(settings=settings, service=service)

    evaluation_service.recover_interrupted_runs.assert_called_once_with()
