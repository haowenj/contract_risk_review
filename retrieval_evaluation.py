import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode

from mineru_to_nodes import (
    INPUT_PATH,
    RETRIEVAL_CONTEXT_PATH,
    build_nodes,
    embedding_model,
    load_retrieval_contexts,
)

TOP_K = 10
RERANK_TOP_N = 10
SELECTOR_FALLBACK_TOP_K = 3
SUMMARY_TIMEOUT_SECONDS = 120.0

load_dotenv()
RERANK_MODEL = os.getenv("LLM_RERANK_MODEL", "qwen3-rerank")
SUMMARY_LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-plus")
logger = logging.getLogger(__name__)

SELECTOR_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "evidence_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["evidence_indices"],
            "additionalProperties": False,
        },
    },
}

ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "contract_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

EVALUATION_QUERIES = [
    {
        "query": "合同的付款方式是什么？",
        "expected_source_object_indices": [],
    },

    {
        "query": "合同分几期付款，每期比例是多少？",
        "expected_source_object_indices": [111, 112, 113, 114],
    },

    {
        "query": "乙方未经甲方同意能否进行分包？",
        "expected_source_object_indices": [174],
    },

    {
        "query": "保密义务什么时候终止？",
        "expected_source_object_indices": [148],
    },

    {
        "query": "合同解除或终止后保密条款是否继续有效？",
        "expected_source_object_indices": [150],
    },

    {
        "query": "乙方收到索赔要求后多久需要答复？",
        "expected_source_object_indices": [204],
    },

    {
        "query": "技术服务的质保期是多久？",
        "expected_source_object_indices": [136],
    },
    {
        "query": "乙方延期履约需要承担什么违约责任？",
        "expected_source_object_indices": [209],
    },
    {
        "query": "本合同产生的新知识产权归谁所有？",
        "expected_source_object_indices": [143],
    },
    {
        "query": "违反网络和数据安全义务需要承担什么责任？",
        "expected_source_object_indices": [223],
    },
]


def _source_object_index(result: Any) -> Any:
    return result.node.metadata.get("source_object_index")


def _build_structured_llm(response_format: dict[str, Any]) -> Any:
    return ChatOpenAI(
        model=SUMMARY_LLM_MODEL,
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0,
        timeout=SUMMARY_TIMEOUT_SECONDS,
        max_retries=0,
        extra_body={"enable_thinking": False},
    ).bind(response_format=response_format)


def build_selector_llm() -> Any:
    return _build_structured_llm(SELECTOR_RESPONSE_FORMAT)


def build_answer_llm() -> Any:
    return _build_structured_llm(ANSWER_RESPONSE_FORMAT)


def build_summary_llm() -> Any:
    """Backward-compatible alias for the Evidence Selector LLM."""
    return build_selector_llm()


def _build_evidence_text(results: list[Any]) -> str:
    return "\n\n".join(
        f"[source_object_index={_source_object_index(result)}]\n{result.node.text}"
        for result in results
    )


def _selector_prompt(query: str, evidence_text: str) -> str:
    return f"""你需要从候选合同证据中，选择回答用户问题所需的“最小充分证据集”。

要求：
1. 只选择能够直接回答问题，或构成答案必要条件、例外、限制的证据。
2. 不要因为某条证据只是与主题相关就选择它。
3. 如果一条证据已经足够回答问题，只选择这一条。
4. 如果多个证据共同构成完整答案，则选择所有不可缺少的证据。
5. 不回答用户问题，不总结证据，只进行证据选择。
6. 与问题关系不明确、文本存在明显异常或无法可靠理解的证据不要选择。
7. 不得选择候选列表中不存在的 source_object_index。

协议标识：json
协议字段名：evidence_indices；只填写候选列表中实际选中的 source_object_index 数组。

用户问题：
{query}

候选证据：
{evidence_text}
"""


def _answer_prompt(query: str, selected_evidence: str) -> str:
    return f"""你需要仅依据下面已经筛选出的合同证据回答用户问题。

要求：
1. 直接回答用户问题。
2. 不得加入所提供证据之外的合同事实。
3. 如果多个证据共同组成完整答案，应完整综合。
4. 不扩展用户没有询问的其他合同事项。
5. 如果证据不足以可靠回答，应明确说明证据不足，不得猜测。

协议标识：json
协议字段名：answer；只生成 answer。

用户问题：
{query}

合同证据：
{selected_evidence}
"""


def _response_content(response: Any) -> str:
    content = response if isinstance(response, str) else getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    raise ValueError("Summary LLM response does not contain text content")


def _parse_json_value(response: Any) -> Any:
    additional_kwargs = getattr(response, "additional_kwargs", {})
    payload = additional_kwargs.get("parsed")
    if payload is None:
        payload = json.loads(_response_content(response))
        if isinstance(payload, str):
            payload = json.loads(payload)
    return payload


def _parse_json_object(response: Any) -> dict[str, Any]:
    payload = _parse_json_value(response)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def _normalize_selected_indices(
    reranked_nodes: list[Any],
    selected_indices: Any,
) -> list[int]:
    if not isinstance(selected_indices, list) or not selected_indices:
        raise ValueError("Selector must return a non-empty evidence_indices list")

    candidate_indices = {
        _source_object_index(result)
        for result in reranked_nodes
        if type(_source_object_index(result)) is int
    }
    selected_set = set()
    for source_index in selected_indices:
        if type(source_index) is not int or source_index not in candidate_indices:
            raise ValueError(
                "Selector returned source_object_index outside rerank candidates"
            )
        selected_set.add(source_index)

    if not selected_set:
        raise ValueError("Selector returned no usable evidence index")

    ordered_indices = []
    for result in reranked_nodes:
        source_index = _source_object_index(result)
        if source_index in selected_set and source_index not in ordered_indices:
            ordered_indices.append(source_index)
    return ordered_indices


def _fallback_selected_indices(reranked_nodes: list[Any]) -> list[int]:
    selected_indices = []
    for result in reranked_nodes[:SELECTOR_FALLBACK_TOP_K]:
        source_index = _source_object_index(result)
        if type(source_index) is int and source_index not in selected_indices:
            selected_indices.append(source_index)
    return selected_indices


def select_evidence(
    query: str,
    reranked_nodes: list[Any],
    *,
    llm: Any | None = None,
) -> list[int]:
    if not reranked_nodes:
        return []

    active_llm = build_selector_llm() if llm is None else llm
    try:
        response = active_llm.invoke(
            _selector_prompt(query, _build_evidence_text(reranked_nodes))
        )
        payload = _parse_json_value(response)
        if isinstance(payload, list):
            raw_selected_indices = payload
        elif isinstance(payload, dict):
            raw_selected_indices = payload.get("evidence_indices")
        else:
            raise ValueError("Selector response must be a JSON object or array")
        return _normalize_selected_indices(
            reranked_nodes,
            raw_selected_indices,
        )
    except Exception as exc:
        logger.warning(
            "Evidence selector failed for query=%r; falling back to rerank Top%d: %s",
            query,
            SELECTOR_FALLBACK_TOP_K,
            exc,
        )
        return _fallback_selected_indices(reranked_nodes)


def _filter_nodes_by_indices(
    reranked_nodes: list[Any],
    selected_indices: list[int],
) -> list[Any]:
    selected_set = set(selected_indices)
    return [
        result
        for result in reranked_nodes
        if _source_object_index(result) in selected_set
    ]


def generate_answer(
    query: str,
    selected_nodes: list[Any],
    *,
    llm: Any | None = None,
) -> str:
    if not selected_nodes:
        return "证据不足，无法根据当前筛选证据生成答案。"

    active_llm = build_answer_llm() if llm is None else llm
    try:
        response = active_llm.invoke(
            _answer_prompt(query, _build_evidence_text(selected_nodes))
        )
        payload = _parse_json_object(response)
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer Generator must return a non-empty answer")
        return answer.strip()
    except Exception as exc:
        logger.warning(
            "Answer generation failed for query=%r: %s",
            query,
            exc,
        )
        return "证据不足，无法根据当前筛选证据生成答案。"


def generate_summaries(
    evaluations: list[dict[str, Any]],
    *,
    selector_llm: Any | None = None,
    answer_llm: Any | None = None,
) -> list[dict[str, Any]]:
    active_selector_llm = selector_llm or build_selector_llm()
    active_answer_llm = answer_llm or build_answer_llm()

    for evaluation in evaluations:
        reranked_nodes = evaluation.get("reranked_results", [])
        selected_indices = select_evidence(
            evaluation["query"],
            reranked_nodes,
            llm=active_selector_llm,
        )
        selected_nodes = _filter_nodes_by_indices(
            reranked_nodes,
            selected_indices,
        )
        selected_indices = [
            source_index
            for result in selected_nodes
            for source_index in [_source_object_index(result)]
            if type(source_index) is int
        ]
        answer = generate_answer(
            evaluation["query"],
            selected_nodes,
            llm=active_answer_llm,
        )

        evaluation["selected_indices"] = selected_indices
        evaluation["selected_nodes"] = selected_nodes
        evaluation["llm_summary"] = {
            "answer": answer,
            "evidence_indices": selected_indices,
        }

    return evaluations


def _build_rerank_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/reranks"):
        return base
    if base.endswith("/compatible-api/v1"):
        return f"{base}/reranks"
    if base.endswith("/compatible-mode/v1"):
        base = base[: -len("/compatible-mode/v1")]
        return f"{base}/compatible-api/v1/reranks"
    return f"{base}/compatible-api/v1/reranks"


class DashScopeReranker:
    def __init__(self, *, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.endpoint = _build_rerank_endpoint(base_url)

    def postprocess_nodes(
        self,
        nodes: list[Any],
        *,
        query_str: str,
    ) -> list[Any]:
        documents = []
        for result in nodes:
            node = result.node
            if node.metadata.get("retrieval_score") is None:
                node.metadata["retrieval_score"] = result.score
            documents.append(node.get_content(metadata_mode=MetadataMode.EMBED))

        response = httpx.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query_str,
                "documents": documents,
                "top_n": RERANK_TOP_N,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()
        ranked_items = payload.get("results")
        if not isinstance(ranked_items, list):
            raise ValueError("Rerank response must contain a results list")

        reranked_nodes = []
        seen_indices = set()
        for item in ranked_items:
            if not isinstance(item, dict):
                raise ValueError("Each rerank result must be an object")

            document_index = item.get("index")
            relevance_score = item.get("relevance_score")
            if (
                type(document_index) is not int
                or document_index < 0
                or document_index >= len(nodes)
                or document_index in seen_indices
            ):
                raise ValueError("Rerank result contains an invalid document index")
            if not isinstance(relevance_score, (int, float)):
                raise ValueError("Rerank result must contain a numeric relevance_score")

            seen_indices.add(document_index)
            result = nodes[document_index]
            result.score = float(relevance_score)
            reranked_nodes.append(result)

        return reranked_nodes


def build_reranker() -> DashScopeReranker:
    return DashScopeReranker(
        model=RERANK_MODEL,
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )


def _record_vector_scores(results: list[Any]) -> dict[Any, Any]:
    scores: dict[Any, Any] = {}
    for result in results:
        node = result.node
        score = node.metadata.get("retrieval_score")
        if score is None:
            score = result.score
            if score is not None:
                node.metadata["retrieval_score"] = score
        scores[_source_object_index(result)] = score
    return scores


def _rank_by_source_object_index(results: list[Any]) -> dict[Any, int]:
    ranks: dict[Any, int] = {}
    for rank, result in enumerate(results, start=1):
        source_index = _source_object_index(result)
        ranks.setdefault(source_index, rank)
    return ranks


def recall_at_k(
    results: list[Any],
    expected_source_object_indices: list[int],
    k: int,
) -> float | None:
    expected = set(expected_source_object_indices)
    if not expected:
        return None

    retrieved = {
        _source_object_index(result)
        for result in results[:k]
    }
    return len(expected & retrieved) / len(expected)


def evaluate_query(
    retriever: Any,
    query_item: dict[str, Any],
    *,
    reranker: Any,
) -> dict[str, Any]:
    expected = query_item["expected_source_object_indices"]
    query = query_item["query"]
    vector_results = list(retriever.retrieve(query))[:TOP_K]
    vector_scores = _record_vector_scores(vector_results)
    reranked_results = list(
        reranker.postprocess_nodes(vector_results, query_str=query)
    )[:RERANK_TOP_N]

    return {
        "query": query,
        "expected_source_object_indices": expected,
        "vector_results": vector_results,
        "reranked_results": reranked_results,
        "vector_scores": vector_scores,
        "vector_ranks": _rank_by_source_object_index(vector_results),
        "rerank_ranks": _rank_by_source_object_index(reranked_results),
        "vector_recall_at_5": recall_at_k(vector_results, expected, 5),
        "vector_recall_at_10": recall_at_k(vector_results, expected, 10),
        "rerank_recall_at_5": recall_at_k(reranked_results, expected, 5),
        "rerank_recall_at_10": recall_at_k(reranked_results, expected, 10),
    }


def run_evaluation(
    index: Any,
    queries: list[dict[str, Any]] = EVALUATION_QUERIES,
    *,
    reranker: Any | None = None,
) -> list[dict[str, Any]]:
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    active_reranker = build_reranker() if reranker is None else reranker
    return [
        evaluate_query(
            retriever,
            query_item,
            reranker=active_reranker,
        )
        for query_item in queries
    ]


def _summary_is_insufficient(summary: dict[str, Any]) -> bool:
    answer = summary.get("answer", "")
    evidence_indices = summary.get("evidence_indices")
    return (
        not evidence_indices
        or not isinstance(answer, str)
        or "证据不足" in answer
    )


def print_evaluation(
    evaluations: list[dict[str, Any]],
    *,
    file: Any | None = None,
) -> None:
    stream = sys.stdout if file is None else file
    summary_count = 0

    for evaluation in evaluations:
        summary = evaluation.get("llm_summary")
        if summary is None:
            continue

        summary_count += 1
        print(f"Query: {evaluation['query']}", file=stream)

        print("=== Rerank Top10 ===", file=stream)
        for rank, result in enumerate(
            evaluation.get("reranked_results", []),
            start=1,
        ):
            print(f"rank: {rank}", file=stream)
            print(f"rerank_score: {result.score}", file=stream)
            print(
                "source_object_index: "
                f"{_source_object_index(result)}",
                file=stream,
            )
            print(f"text: {result.node.text}", file=stream)

        print("=== Selected Evidence ===", file=stream)
        for result in evaluation.get("selected_nodes", []):
            print(
                "source_object_index: "
                f"{_source_object_index(result)}",
                file=stream,
            )
            print(f"text: {result.node.text}", file=stream)

        print("=== Final Answer ===", file=stream)
        if _summary_is_insufficient(summary):
            print("Evidence insufficient:", file=stream)
        print(f"answer: {summary.get('answer', '')}", file=stream)
        print(
            "evidence_indices: "
            f"{summary.get('evidence_indices', [])}",
            file=stream,
        )
        print(file=stream)

    if summary_count == 0:
        print("未生成 LLM 总结。", file=stream)


def build_index(
    input_path: Path = INPUT_PATH,
    context_path: Path = RETRIEVAL_CONTEXT_PATH,
) -> VectorStoreIndex:
    with input_path.open("r", encoding="utf-8") as file:
        objects = json.load(file)

    retrieval_contexts = load_retrieval_contexts(context_path)
    nodes = build_nodes(objects, retrieval_contexts=retrieval_contexts)
    return VectorStoreIndex(nodes, embed_model=embedding_model)


def main() -> None:
    index = build_index()
    evaluations = run_evaluation(index)
    generate_summaries(evaluations)
    print_evaluation(evaluations)


if __name__ == "__main__":
    main()
