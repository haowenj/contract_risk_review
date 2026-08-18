from __future__ import annotations

from typing import Any

import retrieval_evaluation

from app.rag_pipeline import PIPELINE_VERSION


_SERIALIZED_METADATA_KEYS = (
    "page_idx",
    "start_page_idx",
    "end_page_idx",
    "source_page_indices",
    "source_bboxes",
    "merged_cross_page",
    "retrieval_context",
)


def _source_object_index(result: Any) -> Any:
    return (getattr(result.node, "metadata", {}) or {}).get(
        "source_object_index"
    )


def _rank_by_source_object_index(results: list[Any]) -> dict[Any, int]:
    ranks: dict[Any, int] = {}
    for rank, result in enumerate(results, start=1):
        source_index = _source_object_index(result)
        ranks.setdefault(source_index, rank)
    return ranks


def _vector_scores(results: list[Any]) -> dict[Any, Any]:
    scores: dict[Any, Any] = {}
    for result in results:
        metadata = getattr(result.node, "metadata", {}) or {}
        score = metadata.get("retrieval_score")
        if score is None:
            score = getattr(result, "score", None)
        scores[_source_object_index(result)] = score
    return scores


def build_config_snapshot() -> dict[str, Any]:
    return {
        "vector_top_k": retrieval_evaluation.TOP_K,
        "rerank_top_k": retrieval_evaluation.RERANK_TOP_N,
        "rerank_model": retrieval_evaluation.RERANK_MODEL,
        "selector_model": retrieval_evaluation.SUMMARY_LLM_MODEL,
        "answer_model": retrieval_evaluation.SUMMARY_LLM_MODEL,
        "pipeline_version": PIPELINE_VERSION,
    }


def build_evaluation_result(
    pipeline_result: dict[str, Any],
    expected_source_object_indices: list[int],
) -> dict[str, Any]:
    vector_results = list(pipeline_result.get("vector_results", []))
    reranked_results = list(pipeline_result.get("reranked_results", []))
    expected = list(expected_source_object_indices)
    return {
        **pipeline_result,
        "expected_source_object_indices": expected,
        "vector_scores": _vector_scores(vector_results),
        "vector_ranks": _rank_by_source_object_index(vector_results),
        "rerank_ranks": _rank_by_source_object_index(reranked_results),
        "vector_source_object_indices": [
            _source_object_index(result) for result in vector_results
        ],
        "rerank_source_object_indices": [
            _source_object_index(result) for result in reranked_results
        ],
        "vector_recall_at_5": retrieval_evaluation.recall_at_k(
            vector_results,
            expected,
            5,
        ),
        "vector_recall_at_10": retrieval_evaluation.recall_at_k(
            vector_results,
            expected,
            10,
        ),
        "rerank_recall_at_5": retrieval_evaluation.recall_at_k(
            reranked_results,
            expected,
            5,
        ),
        "rerank_recall_at_10": retrieval_evaluation.recall_at_k(
            reranked_results,
            expected,
            10,
        ),
    }


def _serialize_result(result: Any) -> dict[str, Any]:
    node = result.node
    metadata = getattr(node, "metadata", {}) or {}
    serialized: dict[str, Any] = {
        "source_object_index": metadata.get("source_object_index"),
        "node_id": getattr(node, "node_id", None),
        "text": getattr(node, "text", ""),
    }
    for key in _SERIALIZED_METADATA_KEYS:
        if key in metadata and metadata[key] is not None:
            serialized[key] = metadata[key]

    if metadata.get("node_type") == "table":
        for key in (
            "node_type",
            "bbox",
            "table_body",
            "table_caption",
            "table_footnote",
        ):
            if key in metadata and metadata[key] is not None:
                serialized[key] = metadata[key]

    score = getattr(result, "score", None)
    if score is not None:
        serialized["score"] = score
    retrieval_score = metadata.get("retrieval_score")
    if retrieval_score is not None:
        serialized["retrieval_score"] = retrieval_score
    return serialized


def serialize_pipeline_result(result: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(result)
    for key in ("vector_results", "reranked_results", "selected_nodes"):
        serialized[key] = [
            _serialize_result(item) for item in result.get(key, [])
        ]
    return serialized
