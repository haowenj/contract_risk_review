import json
from types import SimpleNamespace

import pytest

from app.rag_pipeline import RAGPipeline


def result_for(source_object_index: int, text: str, score: float = 0.9):
    node = SimpleNamespace(
        metadata={"source_object_index": source_object_index},
        node_id=f"node-{source_object_index}",
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


class FakeReranker:
    def postprocess_nodes(self, results, *, query_str):
        assert query_str == "问题"
        return list(reversed(results))


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


def test_pipeline_rejects_blank_questions():
    with pytest.raises(ValueError, match="question must not be empty"):
        RAGPipeline().run(FakeIndex([]), "  ")
