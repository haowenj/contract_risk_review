from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import retrieval_evaluation


PIPELINE_VERSION = "rag-v1"
RETRIEVAL_MODES = frozenset({"vector", "bm25", "hybrid"})
logger = logging.getLogger(__name__)


def _record_raw_scores(results: list[Any]) -> None:
    for result in results:
        node = result.node
        metadata = getattr(node, "metadata", {})
        score = getattr(result, "score", None)
        if score is not None:
            metadata["retrieval_score"] = score


def _merge_candidates(
    vector_results: list[Any],
    bm25_results: list[Any],
) -> list[Any]:
    merged: list[Any] = []
    seen_node_ids: set[str] = set()
    for result in [*vector_results, *bm25_results]:
        node_id = getattr(result.node, "node_id", None)
        if node_id is not None:
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
        merged.append(result)
    return merged


class RAGPipeline:
    """Shared single-question retrieve, rerank, select, and answer pipeline."""

    def retrieve_evidence(
        self,
        index: Any,
        question: str,
        *,
        bm25_index: Any | None = None,
        reranker: Any | None = None,
        selector_llm: Any | None = None,
        fallback_on_empty_selection: bool = True,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be empty")

        operation_started = perf_counter()
        vector_started = perf_counter()
        vector_retriever = index.as_retriever(
            similarity_top_k=retrieval_evaluation.TOP_K
        )
        vector_results = list(
            vector_retriever.retrieve(question)
        )[: retrieval_evaluation.TOP_K]
        _record_raw_scores(vector_results)
        vector_ms = (perf_counter() - vector_started) * 1000

        bm25_results: list[Any] = []
        bm25_ms = 0.0
        if bm25_index is not None:
            bm25_started = perf_counter()
            bm25_results = list(
                bm25_index.retrieve(
                    question,
                    top_k=retrieval_evaluation.TOP_K,
                )
            )[: retrieval_evaluation.TOP_K]
            _record_raw_scores(bm25_results)
            bm25_ms = (perf_counter() - bm25_started) * 1000

        merge_started = perf_counter()
        merged_results = _merge_candidates(vector_results, bm25_results)
        merge_ms = (perf_counter() - merge_started) * 1000

        rerank_started = perf_counter()
        if merged_results:
            active_reranker = (
                retrieval_evaluation.build_reranker()
                if reranker is None
                else reranker
            )
            reranked_results = list(
                active_reranker.postprocess_nodes(
                    merged_results,
                    query_str=question,
                )
            )[: retrieval_evaluation.RERANK_TOP_N]
        else:
            reranked_results = []
        rerank_ms = (perf_counter() - rerank_started) * 1000

        evidence_started = perf_counter()
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
        evidence_ms = (perf_counter() - evidence_started) * 1000
        timings_ms = {
            "vector": round(vector_ms, 3),
            "bm25": round(bm25_ms, 3),
            "merge": round(merge_ms, 3),
            "rerank": round(rerank_ms, 3),
            "evidence": round(evidence_ms, 3),
            "total": round((perf_counter() - operation_started) * 1000, 3),
        }
        retrieval_diagnostics = {
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
            "merged_candidate_count": len(merged_results),
            "reranked_count": len(reranked_results),
            "evidence_count": len(selected_nodes),
            "evidence_node_ids": [
                getattr(result.node, "node_id", None)
                for result in selected_nodes
            ],
            "timings_ms": timings_ms,
        }
        logger.info(
            "hybrid retrieval completed query=%r mode=%s "
            "vector_count=%d bm25_count=%d merged_candidate_count=%d "
            "reranked_count=%d evidence_count=%d evidence_node_ids=%s "
            "vector_ms=%.3f bm25_ms=%.3f merge_ms=%.3f rerank_ms=%.3f "
            "evidence_ms=%.3f total_ms=%.3f",
            question,
            "hybrid" if bm25_index is not None else "vector",
            retrieval_diagnostics["vector_count"],
            retrieval_diagnostics["bm25_count"],
            retrieval_diagnostics["merged_candidate_count"],
            retrieval_diagnostics["reranked_count"],
            retrieval_diagnostics["evidence_count"],
            retrieval_diagnostics["evidence_node_ids"],
            timings_ms["vector"],
            timings_ms["bm25"],
            timings_ms["merge"],
            timings_ms["rerank"],
            timings_ms["evidence"],
            timings_ms["total"],
        )
        return {
            "query": question,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "merged_results": merged_results,
            "reranked_results": reranked_results,
            "selected_indices": selected_indices,
            "selected_nodes": selected_nodes,
            "retrieval_diagnostics": retrieval_diagnostics,
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
