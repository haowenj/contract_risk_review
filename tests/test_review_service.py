import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.db import ContractRepository
from app.review_db import ContractReviewRepository
from app.review_logging import ReviewRunJournal
from app.review_service import ContractReviewWebService
from app.service import ContractNotReadyError

RESULT = {
    "contract_id": "c1",
    "review_rule_text": "审查规范",
    "summary": {
        "total_items": 2,
        "risk_count": 1,
        "high_risk_count": 0,
        "medium_risk_count": 1,
        "low_risk_count": 0,
        "no_obvious_risk_count": 1,
        "needs_review_count": 0,
    },
    "review_results": [
        {"item_name": "付款期限"},
        {"item_name": "分包转包限制"},
    ],
}


class RecordingReviewRepository(ContractReviewRepository):
    def __init__(self, database_path: Path):
        self.progress_updates = []
        super().__init__(database_path)

    def mark_processing(self, run_id, progress):
        run = super().mark_processing(run_id, progress)
        self.progress_updates.append(progress)
        return run

    def update_progress(self, run_id, progress):
        run = super().update_progress(run_id, progress)
        self.progress_updates.append(progress)
        return run


class FakeContractReviewService:
    def __init__(self, progress_callback, *, failure=None):
        self.progress_callback = progress_callback
        self.failure = failure

    def run(self, contract_id, review_rule_text):
        if self.failure is not None:
            raise self.failure
        self.progress_callback(
            "review_items_parsed",
            {
                "review_items": [
                    {"id": "item_1", "name": "付款期限"},
                    {"id": "item_2", "name": "分包转包限制"},
                ]
            },
        )
        self.progress_callback(
            "review_item_started",
            {
                "current_item_index": 0,
                "item": {"id": "item_1", "name": "付款期限"},
            },
        )
        self.progress_callback(
            "retrieval_query_rewritten",
            {"item_id": "item_1", "retrieval_query": "内部查询"},
        )
        self.progress_callback(
            "review_item_completed",
            {"result": {"item_name": "付款期限"}},
        )
        self.progress_callback(
            "review_item_started",
            {
                "current_item_index": 1,
                "item": {"id": "item_2", "name": "分包转包限制"},
            },
        )
        self.progress_callback(
            "absence_check_started",
            {"item_id": "item_2"},
        )
        self.progress_callback(
            "empty_evidence_rerank_debug",
            {"rerank_top3": [{"score": 0.99}], "retrieval_query": "debug"},
        )
        self.progress_callback(
            "review_item_completed",
            {"result": {"item_name": "分包转包限制"}},
        )
        return RESULT


class ContractLookup:
    def __init__(self, repository):
        self.repository = repository

    def get_contract(self, contract_id):
        return self.repository.get(contract_id)


def build_service(
    root: Path,
    *,
    status="ready",
    failure=None,
    review_service_factory=None,
    diagnostics=False,
):
    contracts = ContractRepository(root / "contracts.db")
    contract = contracts.create("contract.pdf", root / "contract", contract_id="c1")
    contracts.update_status(
        contract.contract_id,
        status,
        index_version="index-v1" if status == "ready" else None,
    )
    repository = RecordingReviewRepository(root / "contracts.db")
    contract_service = ContractLookup(contracts)
    factory_calls = []

    def default_factory(*, contract_service, progress_callback):
        factory_calls.append(contract_service)
        return FakeContractReviewService(progress_callback, failure=failure)

    factory = review_service_factory or default_factory
    service_kwargs = {}
    if diagnostics:
        service_kwargs["review_runs_dir"] = (
            diagnostics if isinstance(diagnostics, Path) else root / "review_runs"
        )
    service = ContractReviewWebService(
        contract_service=contract_service,
        review_repository=repository,
        review_service_factory=factory,
        **service_kwargs,
    )
    return service, contract_service, repository, factory_calls


def read_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_second_retrieval_success_writes_only_compact_diagnostics():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        evidence = {
            "source_object_index": 18,
            "page_idx": 3,
            "node_type": "text",
            "evidence_text": "合同中存在明确约定",
            "score": 0.98,
        }
        result = {
            "contract_id": "c1",
            "review_rule_text": "审查规范",
            "summary": {
                "total_items": 1,
                "risk_count": 1,
                "high_risk_count": 1,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "no_obvious_risk_count": 0,
                "needs_review_count": 0,
            },
            "review_results": [
                {
                    "item_id": "item_1",
                    "item_name": "分包转包限制",
                    "risk_status": "risk",
                    "risk_level": "high",
                    "evidence_status": "found",
                    "finding": "存在风险",
                    "risk_description": "允许转包",
                    "suggestion": "删除相关约定",
                    "evidence": [evidence],
                    "absence_check": None,
                }
            ],
        }

        class SecondRetrievalService:
            def __init__(self, progress_callback):
                self.progress_callback = progress_callback

            def run(self, contract_id, review_rule_text):
                self.progress_callback(
                    "review_items_parsed",
                    {"review_items": [{"id": "item_1", "name": "分包转包限制"}]},
                )
                self.progress_callback(
                    "review_item_started",
                    {
                        "current_item_index": 0,
                        "item": {"id": "item_1", "name": "分包转包限制"},
                    },
                )
                self.progress_callback(
                    "evidence_retrieved",
                    {
                        "item_id": "item_1",
                        "attempt": 1,
                        "retrieval_query": "第一次内部查询",
                        "evidence": [],
                    },
                )
                self.progress_callback(
                    "empty_evidence_rerank_debug",
                    {
                        "item_id": "item_1",
                        "attempt": 1,
                        "retrieval_query": "第一次内部查询",
                        "rerank_top3": [{"score": 0.4}],
                    },
                )
                self.progress_callback(
                    "retrieval_query_rewritten",
                    {
                        "item_id": "item_1",
                        "next_attempt": 2,
                        "previous_query": "第一次内部查询",
                        "retrieval_query": "第二次强化查询",
                        "reason": "补充同义词",
                    },
                )
                self.progress_callback(
                    "evidence_retrieved",
                    {
                        "item_id": "item_1",
                        "attempt": 2,
                        "retrieval_query": "第二次强化查询",
                        "evidence": [evidence],
                    },
                )
                self.progress_callback(
                    "review_item_completed",
                    {"result": result["review_results"][0]},
                )
                return result

        def factory(*, contract_service, progress_callback):
            return SecondRetrievalService(progress_callback)

        service, _, _, _ = build_service(
            root,
            review_service_factory=factory,
            diagnostics=True,
        )
        run = service.create_run("c1", "审查规范")

        completed = service.execute_run(run.run_id)
        run_dir = root / "review_runs" / run.run_id
        events = read_json_lines(run_dir / "events.jsonl")
        persisted_result = json.loads((run_dir / "result.json").read_text())
        current_item_exists = (run_dir / "current_item.jsonl.tmp").exists()

    assert completed.status == "ready"
    item_event = next(event for event in events if event["event"] == "item_completed")
    summary_event = next(
        event for event in events if event["event"] == "diagnostic_summary"
    )
    assert item_event["detail"] == "summary"
    assert item_event["retrieval_attempts"] == 2
    assert item_event["recovered_by_second_retrieval"] is True
    assert item_event["evidence_locations"] == [
        {"page_idx": 3, "source_object_index": 18, "node_type": "text"}
    ]
    serialized_events = json.dumps(events, ensure_ascii=False)
    assert "第一次内部查询" not in serialized_events
    assert "第二次强化查询" not in serialized_events
    assert "合同中存在明确约定" not in serialized_events
    assert summary_event == {
        "timestamp": summary_event["timestamp"],
        "event": "diagnostic_summary",
        "first_retrieval_success": 0,
        "second_retrieval_success": 1,
        "insufficient": 0,
        "absence_verified": 0,
    }
    assert not current_item_exists
    assert persisted_result == result


def test_create_run_writes_normalized_diagnostic_input_snapshot():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        service, _, _, _ = build_service(root, diagnostics=True)

        run = service.create_run("c1", "  审查规范\n第二项  ")
        input_payload = json.loads(
            (
                root
                / "review_runs"
                / run.run_id
                / "input.json"
            ).read_text()
        )

    assert input_payload == {
        "run_id": run.run_id,
        "contract_id": "c1",
        "review_rule_text": "审查规范\n第二项",
        "created_at": run.created_at,
    }


def test_absence_diagnostics_keep_detail_with_bounded_candidate_payloads():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        journal = ReviewRunJournal(root / "review_runs", "run-1")
        candidates = [
            {
                "source_object_index": index,
                "page_idx": index,
                "node_type": "text",
                "evidence_text": f"候选{index}-" + "证据" * 2_000,
            }
            for index in range(25)
        ]
        journal.started("c1")
        journal.record(
            "review_item_started",
            {
                "current_item_index": 0,
                "item": {"id": "item_1", "name": "分包转包限制"},
            },
        )
        journal.record(
            "evidence_retrieved",
            {
                "item_id": "item_1",
                "attempt": 2,
                "retrieval_query": "第二次仍未命中的查询",
                "evidence": [],
            },
        )
        journal.record("absence_check_started", {"item_id": "item_1"})
        journal.record(
            "absence_candidates_found",
            {
                "item_id": "item_1",
                "candidate_count": 25,
                "candidates": candidates,
            },
        )
        journal.record(
            "review_item_completed",
            {
                "result": {
                    "item_id": "item_1",
                    "item_name": "分包转包限制",
                    "risk_status": "risk",
                    "risk_level": "high",
                    "evidence_status": "found",
                    "evidence": [candidates[0]],
                }
            },
        )
        run_dir = root / "review_runs" / "run-1"
        events = read_json_lines(run_dir / "events.jsonl")
        current_item_exists = (run_dir / "current_item.jsonl.tmp").exists()

    candidate_event = next(
        event for event in events if event["event"] == "absence_candidates_found"
    )
    retained = candidate_event["payload"]["candidates"]
    assert len(retained) == 21
    assert retained[-1] == {"_truncated": True, "_omitted_count": 5}
    assert len(retained[0]["evidence_text"]) == 2_000
    assert "第二次仍未命中的查询" in json.dumps(events, ensure_ascii=False)
    assert not current_item_exists


def test_active_item_diagnostics_stop_at_size_limit_and_survive_interruption():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        journal = ReviewRunJournal(root / "review_runs", "run-1")
        journal.started("c1")
        journal.record(
            "review_item_started",
            {
                "current_item_index": 0,
                "item": {"id": "item_1", "name": "超大异常审查项"},
            },
        )
        candidates = [
            {
                "source_object_index": index,
                "page_idx": index,
                "node_type": "text",
                "evidence_text": "异常证据" * 1_000,
            }
            for index in range(20)
        ]
        for _ in range(10):
            journal.record(
                "absence_candidates_found",
                {
                    "item_id": "item_1",
                    "candidate_count": 20,
                    "candidates": candidates,
                },
            )

        current_item_path = (
            root / "review_runs" / "run-1" / "current_item.jsonl.tmp"
        )
        events = read_json_lines(current_item_path)
        persisted_size = current_item_path.stat().st_size

    assert persisted_size <= 270_000
    assert any(event["event"] == "diagnostic_truncated" for event in events)


def test_background_failure_flushes_current_item_and_writes_failure_details():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        class FailingReviewService:
            def __init__(self, progress_callback):
                self.progress_callback = progress_callback

            def run(self, contract_id, review_rule_text):
                self.progress_callback(
                    "review_items_parsed",
                    {"review_items": [{"id": "item_1", "name": "付款期限"}]},
                )
                self.progress_callback(
                    "review_item_started",
                    {
                        "current_item_index": 0,
                        "item": {"id": "item_1", "name": "付款期限"},
                    },
                )
                self.progress_callback(
                    "evidence_retrieved",
                    {
                        "item_id": "item_1",
                        "attempt": 1,
                        "retrieval_query": "失败前的检索问题",
                        "evidence": [],
                    },
                )
                raise RuntimeError("risk_decision item_1 failed")

        def factory(*, contract_service, progress_callback):
            return FailingReviewService(progress_callback)

        service, _, repository, _ = build_service(
            root,
            review_service_factory=factory,
            diagnostics=True,
        )
        run = service.create_run("c1", "审查规范")

        completed = service.execute_run(run.run_id)
        run_dir = root / "review_runs" / run.run_id
        events = read_json_lines(run_dir / "events.jsonl")
        failure = json.loads((run_dir / "failure.json").read_text())
        current_item_exists = (run_dir / "current_item.jsonl.tmp").exists()
        loaded = repository.get_run(run.run_id)

    assert completed.status == "failed"
    assert loaded.status == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["message"] == "risk_decision item_1 failed"
    assert "RuntimeError: risk_decision item_1 failed" in failure["traceback"]
    assert "失败前的检索问题" in json.dumps(events, ensure_ascii=False)
    assert events[-1]["event"] == "run_failed"
    assert not current_item_exists


def test_needs_review_keeps_detail_even_when_retrieval_found_evidence():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        journal = ReviewRunJournal(root / "review_runs", "run-1")
        evidence = {
            "source_object_index": 7,
            "page_idx": 2,
            "node_type": "text",
            "evidence_text": "条款表述存在歧义",
        }
        journal.started("c1")
        journal.record(
            "review_item_started",
            {
                "current_item_index": 0,
                "item": {"id": "item_1", "name": "付款条件"},
            },
        )
        journal.record(
            "evidence_retrieved",
            {
                "item_id": "item_1",
                "attempt": 1,
                "retrieval_query": "需要人工判断的查询",
                "evidence": [evidence],
            },
        )
        journal.record(
            "review_item_completed",
            {
                "result": {
                    "item_id": "item_1",
                    "item_name": "付款条件",
                    "risk_status": "needs_review",
                    "risk_level": None,
                    "evidence_status": "found",
                    "evidence": [evidence],
                }
            },
        )
        events = read_json_lines(root / "review_runs" / "run-1" / "events.jsonl")

    assert "需要人工判断的查询" in json.dumps(events, ensure_ascii=False)
    completed_event = next(
        event for event in events if event["event"] == "review_item_completed"
    )
    assert completed_event["payload"]["result"]["risk_status"] == "needs_review"


def test_diagnostic_write_failure_does_not_change_review_result():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        blocked_path = root / "blocked"
        blocked_path.write_text("not a directory")
        service, _, repository, _ = build_service(
            root,
            diagnostics=blocked_path,
        )
        run = service.create_run("c1", "审查规范")

        completed = service.execute_run(run.run_id)
        loaded = repository.get_run(run.run_id)

    assert completed.status == "ready"
    assert loaded.status == "ready"
    assert loaded.result == RESULT


def test_background_success_reuses_contract_service_and_persists_safe_progress():
    with TemporaryDirectory() as temp_dir:
        service, contracts, repository, factory_calls = build_service(Path(temp_dir))
        run = service.create_run("c1", "  审查规范  ")

        completed = service.execute_run(run.run_id)
        duplicate = service.execute_run(run.run_id)
        payload = service.get_run_payload("c1", run.run_id)

    assert completed.status == "ready"
    assert duplicate.status == "ready"
    assert payload["result"]["summary"] == RESULT["summary"]
    assert payload["result"]["review_results"] == [
        {
            "item_name": "付款期限",
            "evidence": [],
            "absence_check": None,
        },
        {
            "item_name": "分包转包限制",
            "evidence": [],
            "absence_check": None,
        },
    ]
    assert "review_rule_text" not in payload
    assert factory_calls == [contracts]
    assert [value["message"] for value in repository.progress_updates] == [
        "正在解析审查规范",
        "已解析 2 个审查项",
        "正在审查 1 / 2：付款期限",
        "正在进行第二次证据检索",
        "已完成 1 / 2",
        "正在审查 2 / 2：分包转包限制",
        "正在执行全文缺失核验",
        "已完成 2 / 2",
    ]
    assert "内部查询" not in str(repository.progress_updates)
    assert "rerank_top3" not in str(repository.progress_updates)
    assert "score" not in str(repository.progress_updates)


def test_background_failure_marks_run_failed_without_raising_to_caller():
    with TemporaryDirectory() as temp_dir:
        service, _, repository, _ = build_service(
            Path(temp_dir),
            failure=RuntimeError("review model unavailable"),
        )
        run = service.create_run("c1", "审查规范")

        completed = service.execute_run(run.run_id)
        loaded = repository.get_run(run.run_id)
        payload = service.get_run_payload("c1", run.run_id)

    assert completed.status == "failed"
    assert loaded.error_message == "review model unavailable"
    assert payload["error_message"] == "风险审查执行失败，请查看服务日志。"
    assert loaded.result == {}


def test_create_run_rejects_contract_that_is_not_ready_and_blank_rule():
    with TemporaryDirectory() as temp_dir:
        service, _, repository, _ = build_service(Path(temp_dir), status="processing")

        with pytest.raises(ContractNotReadyError):
            service.create_run("c1", "审查规范")

        assert repository.get_run("missing") is None

    with TemporaryDirectory() as temp_dir:
        ready_service, _, _, _ = build_service(Path(temp_dir))
        with pytest.raises(ValueError, match="review rule text must not be empty"):
            ready_service.create_run("c1", "   ")


def test_run_payload_requires_matching_contract():
    with TemporaryDirectory() as temp_dir:
        service, _, _, _ = build_service(Path(temp_dir))
        run = service.create_run("c1", "审查规范")

        with pytest.raises(KeyError):
            service.get_run_payload("another-contract", run.run_id)
