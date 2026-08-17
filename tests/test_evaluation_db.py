import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from app.db import ContractRepository
from app.evaluation_db import EvaluationRepository


class EvaluationRepositoryTest(TestCase):
    def _build(self, root: Path):
        database_path = root / "contracts.db"
        contracts = ContractRepository(database_path)
        contract = contracts.create("contract.pdf", root / "contract", contract_id="c1")
        contracts.update_status(contract.contract_id, "ready", index_version="index-v3")
        return contracts, EvaluationRepository(database_path)

    def test_replace_cases_binds_current_index_version_and_preserves_order(self):
        with TemporaryDirectory() as temp_dir:
            _, repository = self._build(Path(temp_dir))

            cases = repository.replace_cases(
                "c1",
                "index-v3",
                [("第二题", [20, 21]), ("第一题", [10])],
            )

        self.assertEqual([case.question for case in cases], ["第二题", "第一题"])
        self.assertEqual(
            [case.index_version for case in cases],
            ["index-v3", "index-v3"],
        )
        self.assertEqual(cases[0].expected_source_object_indices, [20, 21])

    def test_run_snapshot_and_recovery_are_persisted(self):
        with TemporaryDirectory() as temp_dir:
            _, repository = self._build(Path(temp_dir))
            cases = repository.replace_cases(
                "c1",
                "index-v3",
                [("第二题", [20, 21]), ("第一题", [10])],
            )
            config_snapshot = {
                "vector_top_k": 10,
                "rerank_top_k": 10,
                "pipeline_version": "rag-v1",
            }
            run = repository.create_run(
                "c1",
                "all",
                "index-v3",
                config_snapshot,
                cases,
            )
            repository.mark_processing(run.run_id)
            repository.save_item(
                run.run_id,
                cases[0],
                {"query": "第二题", "vector_results": []},
            )
            repository.recover_incomplete_runs("service restarted")

            loaded = repository.get_run(run.run_id)
            items = repository.list_run_items(run.run_id)

        self.assertEqual(loaded.status, "failed")
        self.assertEqual(loaded.error_message, "service restarted")
        self.assertEqual(loaded.config_snapshot["pipeline_version"], "rag-v1")
        self.assertEqual(items[0].result["query"], "第二题")

    def test_run_item_json_is_deserialized_as_a_fresh_object(self):
        with TemporaryDirectory() as temp_dir:
            _, repository = self._build(Path(temp_dir))
            case = repository.replace_cases("c1", "index-v3", [("问题", [1])])[0]
            run = repository.create_run(
                "c1",
                "single",
                "index-v3",
                {"pipeline_version": "rag-v1"},
                [case],
            )
            repository.save_item(run.run_id, case, {"nested": {"value": 1}})

            first = repository.list_run_items(run.run_id)[0].result
            first["nested"]["value"] = 2
            second = repository.list_run_items(run.run_id)[0].result

        self.assertEqual(second["nested"]["value"], 1)

