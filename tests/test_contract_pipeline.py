from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from app.config import Settings
from app.db import ContractRepository
from app.index_manager import IndexManager
from app.pipeline import ContractProcessor


class ContractProcessorTest(TestCase):
    def _settings(self, root: Path) -> Settings:
        return Settings(
            project_dir=root,
            data_dir=root / "data",
            database_path=root / "data" / "contracts.db",
            contracts_dir=root / "data" / "contracts",
            mineru_url="http://mineru.test",
            mineru_backend="hybrid-engine",
            mineru_server_url=None,
        )

    def test_success_marks_ready_only_after_index_is_persisted(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            contract_dir = settings.contracts_dir / "c1"
            contract_dir.mkdir(parents=True)
            (contract_dir / "source.pdf").write_bytes(b"%PDF")
            for path in (
                "raw_content_list.json",
                "cleaned_content_list.json",
                "merged_content_list.json",
            ):
                (contract_dir / path).write_text("[]", encoding="utf-8")
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", contract_dir)
            manager = IndexManager(object())
            call_order = []
            persist = Mock(side_effect=lambda **kwargs: call_order.append("persist"))
            fake_index = SimpleNamespace(storage_context=SimpleNamespace(persist=persist))

            def record(name, value=None):
                call_order.append(name)
                return value

            with patch("app.pipeline.run_parse", side_effect=lambda *a, **k: record("parse")), patch(
                "app.pipeline.clean_content_list_file", side_effect=lambda *a, **k: record("clean")), patch(
                "app.pipeline.merge_content_list_file", side_effect=lambda *a, **k: record("merge")), patch(
                "app.pipeline.generate_contexts", side_effect=lambda *a, **k: record("context", {})), patch(
                "app.pipeline.save_retrieval_contexts", side_effect=lambda *a, **k: record("save_context")), patch(
                "app.pipeline.build_nodes", side_effect=lambda *a, **k: record("nodes", [])), patch(
                "app.pipeline.VectorStoreIndex", side_effect=lambda *a, **k: record("index", fake_index)), patch.object(
                    manager, "put", side_effect=lambda *a, **k: record("cache")
                ):
                processor = ContractProcessor(
                    repository,
                    settings,
                    manager,
                    embedding_model=object(),
                )
                result = processor.process(contract.contract_id)

        self.assertEqual(result.status, "ready")
        self.assertEqual(
            call_order,
            ["parse", "clean", "merge", "context", "save_context", "nodes", "index", "persist", "cache"],
        )
        self.assertIsNotNone(result.index_version)
        persist.assert_called_once()

    def test_failure_marks_failed_and_does_not_mark_ready(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            contract_dir = settings.contracts_dir / "c1"
            contract_dir.mkdir(parents=True)
            (contract_dir / "source.pdf").write_bytes(b"%PDF")
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", contract_dir)
            processor = ContractProcessor(
                repository,
                settings,
                IndexManager(object()),
                embedding_model=object(),
            )

            with patch("app.pipeline.run_parse", side_effect=RuntimeError("parse failed")):
                result = processor.process(contract.contract_id)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_message, "parse failed")
