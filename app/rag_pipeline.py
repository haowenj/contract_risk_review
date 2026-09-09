from __future__ import annotations

from typing import Any

import retrieval_evaluation


PIPELINE_VERSION = "rag-v1"
RETRIEVAL_MODES = frozenset({"vector", "bm25", "hybrid"})


class RAGPipeline:
    """Shared single-question retrieve, rerank, select, and answer pipeline."""

    def retrieve_evidence(
        self,
        index: Any,
        question: str,
        *,
        reranker: Any | None = None,
        selector_llm: Any | None = None,
        fallback_on_empty_selection: bool = True,
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
            fallback_on_empty=fallback_on_empty_selection,
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

    def retrieve_raw(
        self,
        index: Any,
        question: str,
        *,
        retrieval_mode: str = "vector",
        bm25_index: Any | None = None,
    ) -> dict[str, Any]:
        """Return raw Top K results without rerank, selection, or generation."""

        if not question.strip():
            raise ValueError("question must not be empty")
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"unsupported retrieval mode: {retrieval_mode}")

        vector_results: list[Any] = []
        bm25_results: list[Any] = []
        if retrieval_mode in {"vector", "hybrid"}:
            vector_retriever = index.as_retriever(
                similarity_top_k=retrieval_evaluation.TOP_K
            )
            vector_results = list(
                vector_retriever.retrieve(question)
            )[: retrieval_evaluation.TOP_K]
            retrieval_evaluation._record_vector_scores(vector_results)

        if retrieval_mode in {"bm25", "hybrid"}:
            if bm25_index is None:
                raise ValueError("BM25 index is required for this retrieval mode")
            bm25_results = list(
                bm25_index.retrieve(
                    question,
                    top_k=retrieval_evaluation.TOP_K,
                )
            )[: retrieval_evaluation.TOP_K]

        if retrieval_mode == "vector":
            results = vector_results
        elif retrieval_mode == "bm25":
            results = bm25_results
        else:
            results = []
            seen_node_ids: set[str] = set()
            for result in [*vector_results, *bm25_results]:
                node_id = getattr(result.node, "node_id", None)
                if node_id is not None and node_id in seen_node_ids:
                    continue
                if node_id is not None:
                    seen_node_ids.add(node_id)
                results.append(result)

        return {
            "query": question,
            "retrieval_mode": retrieval_mode,
            "results": results,
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
