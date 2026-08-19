from __future__ import annotations

from typing import Any

import retrieval_evaluation
from app.evidence_serialization import serialize_node_result
from app.rag_pipeline import RAGPipeline


def _serialize_result(result: Any) -> dict[str, Any]:
    return serialize_node_result(result)


def answer_question(
    index: Any,
    question: str,
    *,
    debug: bool = False,
    reranker: Any | None = None,
    selector_llm: Any | None = None,
    answer_llm: Any | None = None,
    pipeline: RAGPipeline | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("question must not be empty")

    evaluation = (pipeline or RAGPipeline()).run(
        index,
        question,
        reranker=reranker,
        selector_llm=selector_llm,
        answer_llm=answer_llm,
    )
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
