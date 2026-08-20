from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

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


def test_app_creation_recovers_interrupted_runs():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        settings = settings_for(root)
        evaluation_service = Mock()
        service = Mock()
        service.evaluation_service = evaluation_service

        create_app(settings=settings, service=service)

    evaluation_service.recover_interrupted_runs.assert_called_once_with()
