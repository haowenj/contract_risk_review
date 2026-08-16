from __future__ import annotations

from typing import Any

import retrieval_evaluation


def _serialize_result(result: Any) -> dict[str, Any]:
    node = result.node
    metadata = getattr(node, "metadata", {}) or {}
    serialized: dict[str, Any] = {
        "source_object_index": metadata.get("source_object_index"),
        "node_id": getattr(node, "node_id", None),
        "text": getattr(node, "text", ""),
    }
    for key in (
        "page_idx",
        "start_page_idx",
        "end_page_idx",
        "source_page_indices",
        "source_bboxes",
        "merged_cross_page",
        "retrieval_context",
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


def answer_question(
    index: Any,
    question: str,
    *,
    debug: bool = False,
    reranker: Any | None = None,
    selector_llm: Any | None = None,
    answer_llm: Any | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("question must not be empty")

    evaluations = retrieval_evaluation.run_evaluation(
        index,
        [{"query": question, "expected_source_object_indices": []}],
        reranker=reranker,
    )
    evaluation = retrieval_evaluation.generate_summaries(
        evaluations,
        selector_llm=selector_llm,
        answer_llm=answer_llm,
    )[0]
    answer = evaluation["llm_summary"]["answer"]
    evidence = [
        _serialize_result(result)
        for result in evaluation.get("selected_nodes", [])
    ]

    result: dict[str, Any] = {
        "answer": answer,
        "evidence": evidence,
        "debug": None,
    }
    if debug:
        result["debug"] = {
            "rerank_top10": [
                _serialize_result(result_item)
                for result_item in evaluation.get("reranked_results", [])[:10]
            ],
            "selected_evidence": evidence,
            "final_answer": answer,
        }
    return result
