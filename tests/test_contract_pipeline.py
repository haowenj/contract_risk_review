import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import mineru_to_nodes
import retrieval_context_preprocess
from app.config import Settings
from app.db import ContractRepository
from app.image_ingestion import write_json_atomic
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

    def test_reuse_existing_mode_skips_parse_and_runs_downstream_pipeline(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            contract_dir = settings.contracts_dir / "c1"
            contract_dir.mkdir(parents=True)
            (contract_dir / "raw_content_list.json").write_text("[]", encoding="utf-8")
            (contract_dir / "merged_content_list.json").write_text("[]", encoding="utf-8")
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", contract_dir)
            manager = IndexManager(object())
            call_order = []
            persist = Mock(side_effect=lambda **kwargs: call_order.append("persist"))
            fake_index = SimpleNamespace(storage_context=SimpleNamespace(persist=persist))

            def record(name, value=None):
                call_order.append(name)
                return value

            with patch("app.pipeline.run_parse", side_effect=AssertionError("parse should be skipped")), patch(
                "app.pipeline.clean_content_list_file", side_effect=lambda *a, **k: record("clean")
            ), patch(
                "app.pipeline.merge_content_list_file", side_effect=lambda *a, **k: record("merge")
            ), patch(
                "app.pipeline.generate_contexts", side_effect=lambda *a, **k: record("context", {})
            ), patch(
                "app.pipeline.save_retrieval_contexts", side_effect=lambda *a, **k: record("save_context")
            ), patch(
                "app.pipeline.build_nodes", side_effect=lambda *a, **k: record("nodes", [])
            ), patch(
                "app.pipeline.VectorStoreIndex", side_effect=lambda *a, **k: record("index", fake_index)
            ), patch.object(
                manager, "put", side_effect=lambda *a, **k: record("cache")
            ):
                processor = ContractProcessor(
                    repository,
                    settings,
                    manager,
                    embedding_model=object(),
                )
                result = processor.process(
                    contract.contract_id,
                    mode="reuse_existing",
                )

        self.assertEqual(result.status, "ready")
        self.assertEqual(
            call_order,
            ["clean", "merge", "context", "save_context", "nodes", "index", "persist", "cache"],
        )

    def test_process_builds_text_and_table_nodes_with_contexts(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            contract_dir = settings.contracts_dir / "c1"
            contract_dir.mkdir(parents=True)
            (contract_dir / "raw_content_list.json").write_text("[]", encoding="utf-8")
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", contract_dir)
            manager = IndexManager(object())
            fake_index = SimpleNamespace(
                storage_context=SimpleNamespace(persist=Mock())
            )
            merged_objects = [
                {"type": "text", "text": "第四条 付款方式", "text_level": 2},
                {
                    "type": "table",
                    "table_body": (
                        "<table><tr><th>比例</th></tr>"
                        "<tr><td>30%</td></tr></table>"
                    ),
                    "table_caption": ["付款计划"],
                    "table_footnote": ["以到账为准"],
                    "page_idx": 2,
                    "bbox": [1, 2, 3, 4],
                },
            ]
            captured = {}

            def write_merged(_cleaned, merged, _log):
                merged.write_text(
                    json.dumps(merged_objects, ensure_ascii=False),
                    encoding="utf-8",
                )

            def generate_contexts(objects):
                contexts = retrieval_context_preprocess.generate_contexts(
                    objects,
                    context_generator=lambda text, _path: (
                        "表格上下文" if "30%" in text else "正文上下文"
                    ),
                    concurrency=1,
                )
                captured["contexts"] = contexts
                return contexts

            def build_nodes(objects, *, retrieval_contexts):
                nodes = mineru_to_nodes.build_nodes(
                    objects,
                    retrieval_contexts=retrieval_contexts,
                )
                captured["nodes"] = nodes
                return nodes

            with patch("app.pipeline.clean_content_list_file"), patch(
                "app.pipeline.merge_content_list_file",
                side_effect=write_merged,
            ), patch(
                "app.pipeline.generate_contexts",
                side_effect=generate_contexts,
            ), patch(
                "app.pipeline.save_retrieval_contexts"
            ), patch(
                "app.pipeline.build_nodes",
                side_effect=build_nodes,
            ), patch(
                "app.pipeline.VectorStoreIndex",
                return_value=fake_index,
            ), patch.object(manager, "put"):
                processor = ContractProcessor(
                    repository,
                    settings,
                    manager,
                    embedding_model=object(),
                )
                result = processor.process(
                    contract.contract_id,
                    mode="reuse_existing",
                )

        self.assertEqual(result.status, "ready")
        self.assertEqual(captured["contexts"], {0: "正文上下文", 1: "表格上下文"})
        self.assertEqual(len(captured["nodes"]), 2)
        self.assertEqual(captured["nodes"][0].metadata.get("source_object_index"), 0)
        self.assertEqual(captured["nodes"][1].metadata["node_type"], "table")
        self.assertEqual(captured["nodes"][1].metadata["page_idx"], 2)

    def test_process_enriches_table_images_before_context_and_nodes(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = self._settings(root)
            contract_dir = settings.contracts_dir / "c1"
            contract_dir.mkdir(parents=True)
            (contract_dir / "raw_content_list.json").write_text("[]", encoding="utf-8")
            repository = ContractRepository(settings.database_path)
            contract = repository.create("contract.pdf", contract_dir)
            manager = IndexManager(object())
            fake_index = SimpleNamespace(
                storage_context=SimpleNamespace(persist=Mock())
            )
            merged_objects = [
                {
                    "type": "table",
                    "img_path": "images/general-table.jpg",
                    "table_body": "<table><tr><td>营业执照</td></tr></table>",
                    "page_idx": 1,
                }
            ]
            enriched_objects = [
                {
                    **merged_objects[0],
                    "image_processing_status": "ready",
                    "image_type": "general",
                    "structured_data": {
                        "visible_text": "营业执照",
                        "content_description": "证照图片",
                    },
                    "ocr_status": "not_required",
                    "verification_status": "not_required",
                    "verification_details": {},
                }
            ]
            captured = {}
            image_service = Mock()
            image_service.enrich_images.return_value = enriched_objects

            def write_merged(_cleaned, merged, _log):
                merged.write_text(
                    json.dumps(merged_objects, ensure_ascii=False),
                    encoding="utf-8",
                )

            def capture_contexts(objects):
                captured["context_objects"] = objects
                return {0: "证照图片所在章节"}

            def capture_nodes(objects, *, retrieval_contexts):
                captured["node_objects"] = objects
                captured["contexts"] = retrieval_contexts
                return []

            with patch("app.pipeline.clean_content_list_file"), patch(
                "app.pipeline.merge_content_list_file",
                side_effect=write_merged,
            ), patch(
                "app.pipeline.generate_contexts",
                side_effect=capture_contexts,
            ), patch(
                "app.pipeline.save_retrieval_contexts"
            ), patch(
                "app.pipeline.build_nodes",
                side_effect=capture_nodes,
            ), patch(
                "app.pipeline.VectorStoreIndex",
                return_value=fake_index,
            ), patch.object(manager, "put"), patch(
                "app.pipeline.write_json_atomic",
                wraps=write_json_atomic,
            ) as write_json:
                processor = ContractProcessor(
                    repository,
                    settings,
                    manager,
                    embedding_model=object(),
                    image_ingestion_service=image_service,
                )
                result = processor.process(
                    contract.contract_id,
                    mode="reuse_existing",
                )

            self.assertEqual(result.status, "ready")
            image_service.enrich_images.assert_called_once_with(
                merged_objects,
                storage_dir=contract_dir,
            )
            self.assertEqual(captured["context_objects"], enriched_objects)
            self.assertEqual(captured["node_objects"], enriched_objects)
            write_json.assert_called_once()
            self.assertEqual(
                json.loads(
                    (contract_dir / "merged_content_list.json").read_text(
                        encoding="utf-8"
                    )
                ),
                enriched_objects,
            )
