import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from app.db import ContractRepository
from app.review_db import ContractReviewRepository, ReviewRunTransitionError


class ContractReviewRepositoryTest(TestCase):
    def _build(self, root: Path):
        database_path = root / "contracts.db"
        contracts = ContractRepository(database_path)
        contract = contracts.create(
            "contract.pdf",
            root / "contract",
            contract_id="c1",
        )
        contracts.update_status(
            contract.contract_id,
            "ready",
            index_version="index-v1",
        )
        return ContractReviewRepository(database_path)

    def test_run_lifecycle_persists_progress_and_full_result(self):
        with TemporaryDirectory() as temp_dir:
            repository = self._build(Path(temp_dir))
            run = repository.create_run("c1", "付款期限不得超过90日")

            self.assertEqual(run.status, "queued")
            self.assertEqual(run.review_rule_text, "付款期限不得超过90日")
            self.assertEqual(
                run.progress,
                {"stage": "queued", "message": "等待开始风险审查"},
            )

            repository.mark_processing(
                run.run_id,
                {"stage": "parsing_rules", "message": "正在解析审查规范"},
            )
            repository.update_progress(
                run.run_id,
                {
                    "stage": "reviewing",
                    "message": "正在审查 1 / 1：付款期限",
                    "current": 1,
                    "total": 1,
                    "item_name": "付款期限",
                },
            )
            result = {
                "summary": {"total_items": 1, "risk_count": 1},
                "review_results": [{"item_name": "付款期限"}],
            }
            repository.mark_ready(
                run.run_id,
                result,
                {"stage": "completed", "message": "已完成 1 / 1"},
            )
            loaded = repository.get_run(run.run_id)

        self.assertEqual(loaded.status, "ready")
        self.assertIsNotNone(loaded.started_at)
        self.assertIsNotNone(loaded.completed_at)
        self.assertEqual(loaded.progress["message"], "已完成 1 / 1")
        self.assertEqual(loaded.result, result)
        json.dumps(loaded.result, ensure_ascii=False)

    def test_failed_and_interrupted_runs_are_persisted(self):
        with TemporaryDirectory() as temp_dir:
            repository = self._build(Path(temp_dir))
            failed = repository.create_run("c1", "规范一")
            interrupted = repository.create_run("c1", "规范二")
            repository.mark_processing(
                failed.run_id,
                {"stage": "parsing_rules", "message": "正在解析审查规范"},
            )
            repository.mark_failed(failed.run_id, "LLM unavailable")

            recovered = repository.recover_incomplete_runs("服务重启导致风险审查任务中断")
            failed_loaded = repository.get_run(failed.run_id)
            interrupted_loaded = repository.get_run(interrupted.run_id)

        self.assertEqual(recovered, 1)
        self.assertEqual(failed_loaded.status, "failed")
        self.assertEqual(failed_loaded.error_message, "LLM unavailable")
        self.assertEqual(interrupted_loaded.status, "failed")
        self.assertEqual(
            interrupted_loaded.error_message,
            "服务重启导致风险审查任务中断",
        )

    def test_json_columns_are_deserialized_as_fresh_objects(self):
        with TemporaryDirectory() as temp_dir:
            repository = self._build(Path(temp_dir))
            run = repository.create_run("c1", "规范")
            repository.mark_processing(
                run.run_id,
                {"stage": "parsing_rules", "message": "正在解析审查规范"},
            )
            repository.update_progress(
                run.run_id,
                {"stage": "reviewing", "nested": {"current": 1}},
            )

            first = repository.get_run(run.run_id).progress
            first["nested"]["current"] = 99
            second = repository.get_run(run.run_id).progress

        self.assertEqual(second["nested"]["current"], 1)

    def test_terminal_status_cannot_be_restarted_or_overwritten(self):
        with TemporaryDirectory() as temp_dir:
            repository = self._build(Path(temp_dir))
            run = repository.create_run("c1", "规范")
            repository.mark_processing(
                run.run_id,
                {"stage": "parsing_rules", "message": "正在解析审查规范"},
            )

            with self.assertRaises(ReviewRunTransitionError):
                repository.mark_processing(
                    run.run_id,
                    {"stage": "parsing_rules", "message": "重复执行"},
                )

            repository.mark_failed(run.run_id, "service restarted")
            with self.assertRaises(ReviewRunTransitionError):
                repository.mark_ready(
                    run.run_id,
                    {"summary": {"total_items": 1}},
                    {"stage": "completed", "message": "已完成 1 / 1"},
                )
            with self.assertRaises(ReviewRunTransitionError):
                repository.update_progress(
                    run.run_id,
                    {"stage": "reviewing", "message": "迟到的进度"},
                )

            loaded = repository.get_run(run.run_id)

        self.assertEqual(loaded.status, "failed")
        self.assertEqual(loaded.error_message, "service restarted")
