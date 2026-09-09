import json
from types import SimpleNamespace

import pytest

from app.rag_pipeline import RAGPipeline


def result_for(
    source_object_index: int,
    text: str,
    score: float = 0.9,
    *,
    node_id: str | None = None,
):
    node = SimpleNamespace(
        metadata={"source_object_index": source_object_index},
        node_id=node_id or f"node-{source_object_index}",
        text=text,
    )
    return SimpleNamespace(node=node, score=score)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.results


class FakeIndex:
    def __init__(self, results):
        self.retriever = FakeRetriever(results)
        self.similarity_top_k = None

    def as_retriever(self, *, similarity_top_k):
        self.similarity_top_k = similarity_top_k
        return self.retriever


class FakeBM25Index:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query, *, top_k):
        self.queries.append((query, top_k))
        return self.results


class FakeReranker:
    def postprocess_nodes(self, results, *, query_str):
        assert query_str == "问题"
        return list(reversed(results))


class RecordingReranker:
    def __init__(self):
        self.node_ids = []

    def postprocess_nodes(self, results, *, query_str):
        assert query_str == "问题"
        self.node_ids = [result.node.node_id for result in results]
        return list(results)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt):
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def test_pipeline_runs_retrieve_rerank_selector_and_answer_without_gold_semantics():
    index = FakeIndex([result_for(10, "证据")])
    pipeline = RAGPipeline()

    result = pipeline.run(
        index,
        "问题",
        reranker=FakeReranker(),
        selector_llm=FakeLLM({"evidence_indices": [10]}),
        answer_llm=FakeLLM({"answer": "答案"}),
    )

    assert result["query"] == "问题"
    assert result["selected_indices"] == [10]
    assert result["llm_summary"] == {"answer": "答案", "evidence_indices": [10]}
    assert "expected_source_object_indices" not in result
    assert index.similarity_top_k == 10


def test_pipeline_retrieves_reranks_and_selects_evidence_without_answer_generation():
    index = FakeIndex([result_for(10, "证据")])

    result = RAGPipeline().retrieve_evidence(
        index,
        "问题",
        reranker=FakeReranker(),
        selector_llm=FakeLLM({"evidence_indices": [10]}),
    )

    assert result["query"] == "问题"
    assert result["selected_indices"] == [10]
    assert [item.node.text for item in result["selected_nodes"]] == ["证据"]
    assert "llm_summary" not in result


def test_pipeline_can_preserve_explicit_empty_evidence_selection():
    index = FakeIndex([result_for(10, "仅有章节标题")])

    result = RAGPipeline().retrieve_evidence(
        index,
        "问题",
        reranker=FakeReranker(),
        selector_llm=FakeLLM({"evidence_indices": []}),
        fallback_on_empty_selection=False,
    )

    assert result["selected_indices"] == []
    assert result["selected_nodes"] == []


def test_pipeline_hybrid_merges_vector_and_bm25_candidates_before_rerank(caplog):
    vector_index = FakeIndex(
        [
            result_for(10, "向量证据", 0.9),
            result_for(11, "重复证据-向量", 0.8),
        ]
    )
    bm25_index = FakeBM25Index(
        [
            result_for(11, "重复证据-BM25", 2.0),
            result_for(12, "BM25关键词证据", 1.5),
        ]
    )
    reranker = RecordingReranker()
    caplog.set_level("INFO", logger="app.rag_pipeline")

    result = RAGPipeline().retrieve_evidence(
        vector_index,
        "问题",
        bm25_index=bm25_index,
        reranker=reranker,
        selector_llm=FakeLLM({"evidence_indices": [10, 11, 12]}),
        fallback_on_empty_selection=False,
    )

    assert reranker.node_ids == ["node-10", "node-11", "node-12"]
    assert bm25_index.queries == [("问题", 10)]
    assert result["retrieval_diagnostics"]["vector_count"] == 2
    assert result["retrieval_diagnostics"]["bm25_count"] == 2
    assert result["retrieval_diagnostics"]["merged_candidate_count"] == 3
    assert result["retrieval_diagnostics"]["reranked_count"] == 3
    assert result["selected_nodes"][1].node.node_id == "node-11"
    assert result["selected_nodes"][1].node.text == "重复证据-向量"
    assert result["selected_nodes"][2].node.metadata["retrieval_score"] == 1.5
    assert any(
        "vector_count=2" in record.getMessage()
        and "bm25_count=2" in record.getMessage()
        and "merged_candidate_count=3" in record.getMessage()
        for record in caplog.records
    )


def test_pipeline_hybrid_refreshes_raw_scores_for_each_query():
    vector_result = result_for(10, "向量证据", 0.7)
    vector_result.node.metadata["retrieval_score"] = 9.0
    bm25_result = result_for(11, "BM25证据", 1.5)
    bm25_result.node.metadata["retrieval_score"] = 8.0

    result = RAGPipeline().retrieve_evidence(
        FakeIndex([vector_result]),
        "问题",
        bm25_index=FakeBM25Index([bm25_result]),
        reranker=RecordingReranker(),
        selector_llm=FakeLLM({"evidence_indices": [10, 11]}),
    )

    assert result["selected_nodes"][0].node.metadata["retrieval_score"] == 0.7
    assert result["selected_nodes"][1].node.metadata["retrieval_score"] == 1.5


def test_pipeline_rejects_blank_questions():
    with pytest.raises(ValueError, match="question must not be empty"):
        RAGPipeline().run(FakeIndex([]), "  ")
