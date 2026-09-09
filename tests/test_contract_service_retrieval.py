from types import SimpleNamespace

import pytest

from app.service import ContractService


class FakeIndexManager:
    def __init__(self, *, bm25_error=None):
        self.vector_index = object()
        self.bm25_index = object()
        self.bm25_error = bm25_error
        self.get_calls = []
        self.get_bm25_calls = []

    def get(self, contract):
        self.get_calls.append(contract)
        return self.vector_index

    def get_bm25(self, contract):
        self.get_bm25_calls.append(contract)
        if self.bm25_error is not None:
            raise self.bm25_error
        return self.bm25_index


class RecordingRAGPipeline:
    def __init__(self):
        self.calls = []

    def retrieve_evidence(self, index, query, **kwargs):
        self.calls.append(
            {"index": index, "query": query, **kwargs}
        )
        node = SimpleNamespace(
            node_id="node-1",
            text="付款期限：30日",
            metadata={
                "source_object_index": 1,
                "node_type": "text",
            },
        )
        return {"selected_nodes": [SimpleNamespace(node=node, score=0.8)]}


def build_service(manager, pipeline):
    contract = SimpleNamespace(
        contract_id="c1",
        status="ready",
        index_version="v1",
    )
    repository = SimpleNamespace(get=lambda contract_id: contract)
    service = ContractService(
        repository,
        SimpleNamespace(),
        processor=object(),
        index_manager=manager,
        rag_pipeline=pipeline,
        evaluation_service=object(),
    )
    return service, contract


def test_search_contract_loads_both_persisted_indexes_for_one_hybrid_query():
    manager = FakeIndexManager()
    pipeline = RecordingRAGPipeline()
    service, contract = build_service(manager, pipeline)

    results = service.search_contract("c1", "付款期限")

    assert results[0]["node_id"] == "node-1"
    assert manager.get_calls == [contract]
    assert manager.get_bm25_calls == [contract]
    assert pipeline.calls == [
        {
            "index": manager.vector_index,
            "query": "付款期限",
            "bm25_index": manager.bm25_index,
            "fallback_on_empty_selection": False,
        }
    ]


def test_search_contract_does_not_fall_back_to_vector_when_bm25_is_missing():
    manager = FakeIndexManager(
        bm25_error=FileNotFoundError("persisted BM25 index not found")
    )
    pipeline = RecordingRAGPipeline()
    service, _ = build_service(manager, pipeline)

    with pytest.raises(RuntimeError, match="BM25.*重新解析"):
        service.search_contract("c1", "付款期限")

    assert pipeline.calls == []
