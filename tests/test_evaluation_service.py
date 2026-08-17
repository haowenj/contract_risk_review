from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.db import ContractRepository
from app.evaluation_db import EvaluationRepository
from app.evaluation_service import EvaluationService, EvaluationStaleError


def result_for(source_object_index: int, text: str, score: float = 0.9):
    node = SimpleNamespace(
        node_id=f"node-{source_object_index}",
        text=text,
        metadata={
            "source_object_index": source_object_index,
            "page_idx": 4,
            "retrieval_score": 0.6,
        },
    )
    return SimpleNamespace(node=node, score=score)


def pipeline_result(question: str, source_object_index: int):
    result = result_for(source_object_index, f"{question}-证据")
    return {
        "query": question,
        "vector_results": [result],
        "reranked_results": [result],
        "selected_indices": [source_object_index],
        "selected_nodes": [result],
        "llm_summary": {
            "answer": f"{question}-答案",
            "evidence_indices": [source_object_index],
        },
    }


def build_service(root: Path, *, contract_index_version: str = "index-v2"):
    database_path = root / "contracts.db"
    contracts = ContractRepository(database_path)
    contract = contracts.create("contract.pdf", root / "contract", contract_id="c1")
    contracts.update_status(
        contract.contract_id,
        "ready",
        index_version=contract_index_version,
    )
    evaluation_repository = EvaluationRepository(database_path)
    index_manager = Mock()
    index_manager.get.return_value = object()
    pipeline = Mock()
    service = EvaluationService(
        contracts,
        evaluation_repository,
        index_manager,
        pipeline,
    )
    return service, contracts, evaluation_repository, index_manager, pipeline


def test_service_rejects_case_bound_to_old_index_version():
    with TemporaryDirectory() as temp_dir:
        service, _, repository, _, _ = build_service(Path(temp_dir))
        old_case = repository.replace_cases("c1", "index-v1", [("问题", [7])])[0]

        with pytest.raises(EvaluationStaleError):
            service.create_single_run("c1", old_case.case_id)


def test_execute_all_run_reuses_one_persisted_index_and_saves_full_results():
    with TemporaryDirectory() as temp_dir:
        service, _, repository, index_manager, pipeline = build_service(Path(temp_dir))
        repository.replace_cases(
            "c1",
            "index-v2",
            [("问题一", [1]), ("问题二", [2])],
        )
        pipeline.run.side_effect = [pipeline_result("问题一", 1), pipeline_result("问题二", 2)]

        run = service.create_all_run("c1")
        result = service.execute_run(run.run_id)
        items = repository.list_run_items(run.run_id)

    assert result.status == "ready"
    assert run.config_snapshot["vector_top_k"] == 10
    index_manager.get.assert_called_once()
    assert pipeline.run.call_count == 2
    assert len(items) == 2
    assert items[0].result["vector_results"][0]["source_object_index"] == 1
    assert "vector_recall_at_10" in items[0].result


def test_execute_run_marks_failed_when_index_load_fails():
    with TemporaryDirectory() as temp_dir:
        service, _, repository, index_manager, _ = build_service(Path(temp_dir))
        case = repository.replace_cases("c1", "index-v2", [("问题", [1])])[0]
        index_manager.get.side_effect = FileNotFoundError("missing index")

        run = service.create_single_run("c1", case.case_id)
        result = service.execute_run(run.run_id)

    assert result.status == "failed"
    assert result.error_message == "missing index"
