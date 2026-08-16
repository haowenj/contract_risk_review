import json
import os
from types import SimpleNamespace
from unittest import TestCase

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")
os.environ.setdefault("LLM_MODEL", "test-answer-model")
os.environ.setdefault("LLM_RERANK_MODEL", "qwen3-rerank")

from app.qa import answer_question


def result_for(index: int, text: str, score: float):
    node = SimpleNamespace(
        node_id=f"node-{index}",
        text=text,
        metadata={
            "source_object_index": index,
            "page_idx": 4,
            "start_page_idx": 4,
            "end_page_idx": 5,
            "retrieval_score": 0.6,
        },
    )
    return SimpleNamespace(node=node, score=score)


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, question):
        return self.results


class FakeIndex:
    def __init__(self, results):
        self.retriever = FakeRetriever(results)

    def as_retriever(self, *, similarity_top_k):
        return self.retriever


class FakeReranker:
    def postprocess_nodes(self, results, *, query_str):
        return list(reversed(results))


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt):
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


class AppQATest(TestCase):
    def test_answer_question_returns_selected_evidence_and_optional_debug(self):
        index = FakeIndex([result_for(1, "付款条款", 0.8), result_for(2, "付款期限", 0.7)])
        result = answer_question(
            index,
            "付款多久？",
            debug=True,
            reranker=FakeReranker(),
            selector_llm=FakeLLM({"evidence_indices": [2]}),
            answer_llm=FakeLLM({"answer": "180日内付款。"}),
        )

        self.assertEqual(result["answer"], "180日内付款。")
        self.assertEqual([item["source_object_index"] for item in result["evidence"]], [2])
        self.assertEqual(result["debug"]["final_answer"], "180日内付款。")
        self.assertEqual(len(result["debug"]["rerank_top10"]), 2)

    def test_answer_question_hides_debug_but_always_returns_evidence(self):
        result = answer_question(
            FakeIndex([result_for(1, "证据", 0.8)]),
            "问题",
            debug=False,
            reranker=FakeReranker(),
            selector_llm=FakeLLM({"evidence_indices": [1]}),
            answer_llm=FakeLLM({"answer": "答案"}),
        )

        self.assertEqual(result["answer"], "答案")
        self.assertEqual(result["debug"], None)
        self.assertEqual(result["evidence"][0]["text"], "证据")
