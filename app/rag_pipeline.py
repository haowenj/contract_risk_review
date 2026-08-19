from __future__ import annotations

from typing import Any

import retrieval_evaluation


PIPELINE_VERSION = "rag-v1"


class RAGPipeline:
    """Shared single-question retrieve, rerank, select, and answer pipeline."""

    def retrieve_evidence(
        self,
        index: Any,
        question: str,
        *,
        reranker: Any | None = None,
        selector_llm: Any | None = None,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be empty")

        vector_results, reranked_results = retrieval_evaluation.retrieve_and_rerank(
            index,
            question,
            reranker=reranker,
        )
        selected_indices = retrieval_evaluation.select_evidence(
            question,
            reranked_results,
            llm=selector_llm,
        )
        selected_nodes = retrieval_evaluation.filter_nodes_by_indices(
            reranked_results,
            selected_indices,
        )
        return {
            "query": question,
            "vector_results": vector_results,
            "reranked_results": reranked_results,
            "selected_indices": selected_indices,
            "selected_nodes": selected_nodes,
        }

    def run(
        self,
        index: Any,
        question: str,
        *,
        reranker: Any | None = None,
        selector_llm: Any | None = None,
        answer_llm: Any | None = None,
    ) -> dict[str, Any]:
        result = self.retrieve_evidence(
            index,
            question,
            reranker=reranker,
            selector_llm=selector_llm,
        )
        answer = retrieval_evaluation.generate_answer(
            question,
            result["selected_nodes"],
            llm=answer_llm,
        )
        return {
            **result,
            "llm_summary": {
                "answer": answer,
                "evidence_indices": result["selected_indices"],
            },
        }
