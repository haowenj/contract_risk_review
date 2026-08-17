from types import SimpleNamespace

from app.evaluation_metrics import build_evaluation_result, serialize_pipeline_result


def result_for(source_object_index: int, text: str, score: float = 0.9):
    node = SimpleNamespace(
        node_id=f"node-{source_object_index}",
        text=text,
        metadata={
            "source_object_index": source_object_index,
            "page_idx": 4,
            "retrieval_context": f"context-{source_object_index}",
            "retrieval_score": 0.6,
        },
    )
    return SimpleNamespace(node=node, score=score)


def test_evaluation_metrics_compare_gold_only_after_pipeline_runs():
    result = {
        "query": "问题",
        "vector_results": [
            result_for(1, "一"),
            result_for(7, "七"),
            result_for(8, "八"),
        ],
        "reranked_results": [
            result_for(7, "七"),
            result_for(1, "一"),
            result_for(8, "八"),
        ],
        "selected_indices": [7],
        "selected_nodes": [result_for(7, "七")],
        "llm_summary": {"answer": "答案", "evidence_indices": [7]},
    }

    evaluated = build_evaluation_result(result, [7])

    assert evaluated["vector_recall_at_5"] == 1.0
    assert evaluated["rerank_recall_at_5"] == 1.0
    assert evaluated["vector_ranks"][7] == 2
    assert evaluated["rerank_ranks"][7] == 1
    assert evaluated["expected_source_object_indices"] == [7]


def test_empty_gold_recall_is_none():
    result = {
        "query": "问题",
        "vector_results": [result_for(1, "一")],
        "reranked_results": [result_for(1, "一")],
        "selected_indices": [],
        "selected_nodes": [],
        "llm_summary": {"answer": "证据不足", "evidence_indices": []},
    }

    evaluated = build_evaluation_result(result, [])

    assert evaluated["vector_recall_at_5"] is None
    assert evaluated["rerank_recall_at_10"] is None


def test_serialize_pipeline_result_keeps_node_and_score_metadata():
    result = {
        "query": "问题",
        "vector_results": [result_for(7, "证据", score=0.8)],
        "reranked_results": [result_for(7, "证据", score=0.95)],
        "selected_indices": [7],
        "selected_nodes": [result_for(7, "证据", score=0.95)],
        "llm_summary": {"answer": "答案", "evidence_indices": [7]},
    }

    serialized = serialize_pipeline_result(result)

    assert serialized["vector_results"][0]["source_object_index"] == 7
    assert serialized["vector_results"][0]["node_id"] == "node-7"
    assert serialized["reranked_results"][0]["score"] == 0.95
    assert serialized["selected_nodes"][0]["page_idx"] == 4
