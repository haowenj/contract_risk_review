import os
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_MODEL", "test-context-model")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")
os.environ.setdefault("LLM_RERANK_MODEL", "qwen3-rerank")

import retrieval_evaluation


def result_for(source_object_index, text, score=0.9, retrieval_score=None):
    metadata = {
        "source_object_index": source_object_index,
        "retrieval_context": f"context-{source_object_index}",
    }
    if retrieval_score is not None:
        metadata["retrieval_score"] = retrieval_score

    def embedding_content(metadata_mode):
        if metadata_mode.name != "EMBED":
            return f"wrong-content-{source_object_index}"
        return f"embedding-content-{source_object_index}"

    node = SimpleNamespace(
        metadata=metadata,
        node_id=f"node-{source_object_index}",
        text=text,
        get_content=embedding_content,
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
    def __init__(self, retriever):
        self.retriever = retriever
        self.similarity_top_k = None

    def as_retriever(self, *, similarity_top_k):
        self.similarity_top_k = similarity_top_k
        return self.retriever


class RecordingReranker:
    def __init__(self, reranked_results):
        self.reranked_results = reranked_results
        self.calls = []

    def postprocess_nodes(self, results, *, query_str):
        self.calls.append((list(results), query_str))
        return self.reranked_results


class RecordingSummaryLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


class QueueSummaryLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.responses.pop(0))


class RetrievalEvaluationTest(TestCase):
    def test_select_evidence_uses_candidates_and_returns_indices_only(self):
        llm = RecordingSummaryLLM(
            json.dumps(
                {
                    "evidence_indices": [111, 112],
                },
                ensure_ascii=False,
            )
        )
        reranked_nodes = [
            result_for(111, "第一期支付合同款的50%。"),
            result_for(112, "第二期支付合同款的40%。"),
            result_for(113, "第三期支付合同款的10%。"),
            result_for(114, "付款账户信息。"),
        ]

        selected_indices = retrieval_evaluation.select_evidence(
            "合同分几期付款，每期比例是多少？",
            reranked_nodes,
            llm=llm,
        )

        self.assertEqual(selected_indices, [111, 112])
        self.assertIn("最小充分证据集", llm.prompts[0])
        self.assertIn("不回答用户问题，不总结证据，只进行证据选择", llm.prompts[0])
        self.assertIn("不得选择候选列表中不存在的 source_object_index", llm.prompts[0])
        self.assertIn("第一期支付合同款的50%", llm.prompts[0])
        self.assertIn("第二期支付合同款的40%", llm.prompts[0])
        self.assertNotIn("answer", llm.prompts[0])

    def test_select_evidence_normalizes_json_array_response(self):
        llm = RecordingSummaryLLM(json.dumps([112, 111], ensure_ascii=False))
        reranked_nodes = [
            result_for(111, "第一期支付合同款的50%。"),
            result_for(112, "第二期支付合同款的40%。"),
            result_for(113, "第三期支付合同款的10%。"),
            result_for(114, "付款账户信息。"),
        ]

        selected_indices = retrieval_evaluation.select_evidence(
            "合同分几期付款，每期比例是多少？",
            reranked_nodes,
            llm=llm,
        )

        self.assertEqual(selected_indices, [111, 112])

    def test_generate_answer_only_receives_selected_evidence(self):
        llm = RecordingSummaryLLM(
            json.dumps(
                {
                    "answer": "合同分两期支付。",
                },
                ensure_ascii=False,
            )
        )
        selected_nodes = [
            result_for(111, "第一期支付合同款的50%。"),
            result_for(112, "第二期支付合同款的40%。"),
        ]

        answer = retrieval_evaluation.generate_answer(
            "合同分几期付款，每期比例是多少？",
            selected_nodes,
            llm=llm,
        )

        self.assertEqual(answer, "合同分两期支付。")
        self.assertIn("第一期支付合同款的50%", llm.prompts[0])
        self.assertIn("第二期支付合同款的40%", llm.prompts[0])
        self.assertIn("直接回答用户问题", llm.prompts[0])
        self.assertNotIn("evidence_indices", llm.prompts[0])

    def test_generate_summaries_assembles_answer_from_python_selected_nodes(self):
        selector_llm = RecordingSummaryLLM(
            json.dumps(
                {
                    "evidence_indices": [112, 111, 112],
                },
                ensure_ascii=False,
            )
        )
        answer_llm = RecordingSummaryLLM(
            json.dumps(
                {
                    "answer": "合同分两期支付：第一期50%，第二期40%。",
                },
                ensure_ascii=False,
            )
        )
        evaluation = {
            "query": "合同分几期付款，每期比例是多少？",
            "reranked_results": [
                result_for(111, "第一期支付合同款的50%。"),
                result_for(112, "第二期支付合同款的40%。"),
                result_for(113, "付款账户信息。"),
            ],
        }

        result = retrieval_evaluation.generate_summaries(
            [evaluation],
            selector_llm=selector_llm,
            answer_llm=answer_llm,
        )[0]

        self.assertEqual(
            result["llm_summary"],
            {
                "answer": "合同分两期支付：第一期50%，第二期40%。",
                "evidence_indices": [111, 112],
            },
        )
        self.assertEqual(
            [
                item.node.metadata["source_object_index"]
                for item in result["selected_nodes"]
            ],
            [111, 112],
        )
        self.assertIn("第一期支付合同款的50%", answer_llm.prompts[0])
        self.assertIn("第二期支付合同款的40%", answer_llm.prompts[0])
        self.assertNotIn("付款账户信息", answer_llm.prompts[0])

    def test_selector_failure_falls_back_to_rerank_top_three(self):
        class FailingSelectorLLM:
            def invoke(self, prompt):
                raise RuntimeError("selector failed")

        reranked_nodes = [
            result_for(111, "证据一"),
            result_for(112, "证据二"),
            result_for(113, "证据三"),
            result_for(114, "证据四"),
        ]

        selected_indices = retrieval_evaluation.select_evidence(
            "问题",
            reranked_nodes,
            llm=FailingSelectorLLM(),
        )

        self.assertEqual(selected_indices, [111, 112, 113])

    def test_selector_empty_or_invalid_indices_fall_back_to_rerank_top_three(self):
        reranked_nodes = [
            result_for(111, "证据一"),
            result_for(112, "证据二"),
            result_for(113, "证据三"),
            result_for(114, "证据四"),
        ]
        for selector_response in [
            {"evidence_indices": []},
            {"evidence_indices": [999]},
        ]:
            with self.subTest(selector_response=selector_response):
                selected_indices = retrieval_evaluation.select_evidence(
                    "问题",
                    reranked_nodes,
                    llm=RecordingSummaryLLM(
                        json.dumps(selector_response, ensure_ascii=False)
                    ),
                )
                self.assertEqual(selected_indices, [111, 112, 113])

    def test_answer_failure_returns_explicit_insufficient_answer(self):
        class FailingAnswerLLM:
            def invoke(self, prompt):
                raise RuntimeError("answer generator failed")

        answer = retrieval_evaluation.generate_answer(
            "问题",
            [result_for(111, "证据")],
            llm=FailingAnswerLLM(),
        )

        self.assertEqual(answer, "证据不足，无法根据当前筛选证据生成答案。")

    def test_summary_prompt_keeps_json_protocol_for_new_stages(self):
        selector_llm = RecordingSummaryLLM(
            json.dumps({"evidence_indices": [111]}, ensure_ascii=False)
        )
        answer_llm = RecordingSummaryLLM(
            json.dumps({"answer": "答案"}, ensure_ascii=False)
        )
        nodes = [result_for(111, "证据")]

        retrieval_evaluation.select_evidence("问题", nodes, llm=selector_llm)
        retrieval_evaluation.generate_answer("问题", nodes, llm=answer_llm)

        self.assertIn("协议标识：json", selector_llm.prompts[0])
        self.assertIn("协议字段名：evidence_indices", selector_llm.prompts[0])
        self.assertIn("协议标识：json", answer_llm.prompts[0])
        self.assertIn("协议字段名：answer", answer_llm.prompts[0])

    def test_build_summary_llm_uses_json_response_protocol(self):
        with mock.patch.object(retrieval_evaluation, "ChatOpenAI") as factory:
            bound_llm = mock.Mock()
            factory.return_value.bind.return_value = bound_llm

            result = retrieval_evaluation.build_summary_llm()

        factory.assert_called_once_with(
            model=os.environ["LLM_MODEL"],
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            temperature=0,
            timeout=120.0,
            max_retries=0,
            extra_body={"enable_thinking": False},
        )
        factory.return_value.bind.assert_called_once_with(
            response_format=retrieval_evaluation.SELECTOR_RESPONSE_FORMAT,
        )
        self.assertIs(result, bound_llm)

    def test_build_answer_llm_uses_answer_json_schema(self):
        with mock.patch.object(retrieval_evaluation, "ChatOpenAI") as factory:
            bound_llm = mock.Mock()
            factory.return_value.bind.return_value = bound_llm

            result = retrieval_evaluation.build_answer_llm()

        factory.return_value.bind.assert_called_once_with(
            response_format=retrieval_evaluation.ANSWER_RESPONSE_FORMAT,
        )
        self.assertIs(result, bound_llm)

    def test_summary_failure_degrades_to_explicit_insufficient_evidence(self):
        class FailingSummaryLLM:
            def invoke(self, prompt):
                raise RuntimeError("remote summary failed")

        answer = retrieval_evaluation.generate_answer(
            "问题",
            [],
            llm=FailingSummaryLLM(),
        )

        self.assertEqual(answer, "证据不足，无法根据当前筛选证据生成答案。")

    def test_run_evaluation_reranks_all_vector_top_ten_and_keeps_scores(self):
        vector_results = [
            result_for(index, f"证据-{index}", score=1.0 - index / 10)
            for index in range(10)
        ]
        vector_results[9].node.metadata["retrieval_score"] = 0.1234
        reranked_results = [
            SimpleNamespace(node=vector_results[9].node, score=0.9876),
            *[
                SimpleNamespace(node=result.node, score=0.5 - index / 100)
                for index, result in enumerate(vector_results[:9])
            ],
        ]
        retriever = FakeRetriever(vector_results)
        reranker = RecordingReranker(reranked_results)

        evaluations = retrieval_evaluation.run_evaluation(
            FakeIndex(retriever),
            [{"query": "问题", "expected_source_object_indices": [9]}],
            reranker=reranker,
        )

        self.assertEqual(len(reranker.calls), 1)
        self.assertEqual(len(reranker.calls[0][0]), 10)
        self.assertEqual(reranker.calls[0][1], "问题")
        evaluation = evaluations[0]
        self.assertEqual(
            [result.node.metadata["source_object_index"] for result in evaluation["vector_results"]],
            list(range(10)),
        )
        self.assertEqual(
            evaluation["reranked_results"][0].node.metadata["source_object_index"],
            9,
        )
        self.assertEqual(evaluation["vector_recall_at_5"], 0.0)
        self.assertEqual(evaluation["vector_recall_at_10"], 1.0)
        self.assertEqual(evaluation["rerank_recall_at_5"], 1.0)
        self.assertEqual(evaluation["rerank_recall_at_10"], 1.0)
        self.assertEqual(evaluation["vector_ranks"][9], 10)
        self.assertEqual(evaluation["rerank_ranks"][9], 1)
        self.assertEqual(evaluation["vector_scores"][9], 0.1234)

    def test_remote_reranker_posts_embedding_content_and_maps_response_indices(self):
        results = [
            result_for(10, "证据一", score=0.8),
            result_for(11, "证据二", score=0.7, retrieval_score=0.123),
        ]
        response = mock.Mock()
        response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.12},
            ]
        }
        reranker = retrieval_evaluation.DashScopeReranker(
            model="qwen3-rerank",
            api_key="test-key",
            base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
        )

        with mock.patch.object(
            retrieval_evaluation.httpx,
            "post",
            return_value=response,
        ) as post:
            reranked = reranker.postprocess_nodes(results, query_str="问题")

        post.assert_called_once_with(
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen3-rerank",
                "query": "问题",
                "documents": ["embedding-content-10", "embedding-content-11"],
                "top_n": 10,
            },
            timeout=120.0,
        )
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(
            [result.node.metadata["source_object_index"] for result in reranked],
            [11, 10],
        )
        self.assertEqual(reranked[0].score, 0.95)
        self.assertEqual(reranked[1].score, 0.12)
        self.assertEqual(results[0].node.metadata["retrieval_score"], 0.8)
        self.assertEqual(results[1].node.metadata["retrieval_score"], 0.123)

    def test_build_reranker_uses_existing_bailian_configuration(self):
        with mock.patch.object(
            retrieval_evaluation,
            "DashScopeReranker",
        ) as factory:
            retrieval_evaluation.build_reranker()

        factory.assert_called_once_with(
            model="qwen3-rerank",
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
        )

    def test_evaluation_queries_keep_current_questions_and_gold_evidence(self):
        self.assertEqual(len(retrieval_evaluation.EVALUATION_QUERIES), 10)
        self.assertEqual(
            [item["query"] for item in retrieval_evaluation.EVALUATION_QUERIES],
            [
                "合同的付款方式是什么？",
                "合同分几期付款，每期比例是多少？",
                "乙方未经甲方同意能否进行分包？",
                "保密义务什么时候终止？",
                "合同解除或终止后保密条款是否继续有效？",
                "乙方收到索赔要求后多久需要答复？",
                "技术服务的质保期是多久？",
                "乙方延期履约需要承担什么违约责任？",
                "本合同产生的新知识产权归谁所有？",
                "违反网络和数据安全义务需要承担什么责任？",
            ],
        )
        self.assertEqual(
            [
                item["expected_source_object_indices"]
                for item in retrieval_evaluation.EVALUATION_QUERIES
            ],
            [
                [],
                [111, 112, 113, 114],
                [174],
                [148],
                [150],
                [204],
                [136],
                [209],
                [143],
                [223],
            ],
        )

    def test_run_evaluation_uses_top_k_ten_and_keeps_query_results(self):
        retriever = FakeRetriever([result_for(10, "证据")])
        index = FakeIndex(retriever)
        reranker = RecordingReranker([result_for(10, "证据", score=0.88)])
        queries = [
            {"query": "问题一", "expected_source_object_indices": []},
            {"query": "问题二", "expected_source_object_indices": [10]},
        ]

        evaluations = retrieval_evaluation.run_evaluation(
            index,
            queries,
            reranker=reranker,
        )

        self.assertEqual(index.similarity_top_k, 10)
        self.assertEqual(retriever.queries, ["问题一", "问题二"])
        self.assertEqual(len(evaluations), 2)
        self.assertEqual(evaluations[0]["vector_results"][0].node.text, "证据")
        self.assertEqual(evaluations[0]["reranked_results"][0].node.text, "证据")
        self.assertIsNone(evaluations[0]["vector_recall_at_5"])
        self.assertEqual(evaluations[1]["rerank_recall_at_10"], 1.0)

    def test_recall_counts_unique_expected_source_indices_in_top_k(self):
        results = [
            result_for(1, "a"),
            result_for(7, "b"),
            result_for(8, "c"),
            result_for(10, "d"),
            result_for(11, "e"),
            result_for(99, "f"),
        ]

        self.assertEqual(
            retrieval_evaluation.recall_at_k(results, [7, 10, 99], 5),
            2 / 3,
        )
        self.assertEqual(
            retrieval_evaluation.recall_at_k(results, [7, 10, 99], 10),
            1.0,
        )
        self.assertIsNone(retrieval_evaluation.recall_at_k(results, [], 5))

    def test_print_output_includes_result_fields_and_ignores_empty_gold_in_mean(self):
        retriever = FakeRetriever(
            [
                result_for(10, "付款证据", score=0.8765),
                result_for(11, "其他证据", score=0.5),
            ]
        )
        evaluations = retrieval_evaluation.run_evaluation(
            FakeIndex(retriever),
            [
                {"query": "未标注问题", "expected_source_object_indices": []},
                {"query": "已标注问题", "expected_source_object_indices": [10]},
            ],
            reranker=RecordingReranker(
                [
                    result_for(10, "付款证据", score=0.99),
                    result_for(11, "其他证据", score=0.4),
                ]
            ),
        )
        evaluations[0]["llm_summary"] = {
            "answer": "证据不足。",
            "evidence_indices": [],
        }
        evaluations[1]["llm_summary"] = {
            "answer": "付款证据支持该结论。",
            "evidence_indices": [10],
        }
        evaluations[0]["selected_nodes"] = evaluations[0]["reranked_results"][:1]
        evaluations[1]["selected_nodes"] = evaluations[1]["reranked_results"][:1]
        output = StringIO()

        retrieval_evaluation.print_evaluation(evaluations, file=output)

        text = output.getvalue()
        self.assertIn("Evidence insufficient:", text)
        self.assertIn("Query: 未标注问题", text)
        self.assertIn("answer: 证据不足。", text)
        self.assertIn("evidence_indices: []", text)
        self.assertIn("Query: 已标注问题", text)
        self.assertIn("answer: 付款证据支持该结论。", text)
        self.assertIn("evidence_indices: [10]", text)
        self.assertIn("=== Rerank Top10 ===", text)
        self.assertIn("=== Selected Evidence ===", text)
        self.assertIn("=== Final Answer ===", text)
        self.assertIn("source_object_index: 10", text)
        self.assertIn("text: 付款证据", text)

    def test_print_evaluation_prints_summaries_when_all_evidence_is_sufficient(self):
        output = StringIO()
        retrieval_evaluation.print_evaluation(
            [{
                "query": "问题",
                "expected_source_object_indices": [1],
                "reranked_results": [result_for(1, "证据")],
                "selected_nodes": [result_for(1, "证据")],
                "llm_summary": {
                    "answer": "证据支持。",
                    "evidence_indices": [1],
                },
            }],
            file=output,
        )

        self.assertIn("Query: 问题", output.getvalue())
        self.assertIn("answer: 证据支持。", output.getvalue())
        self.assertIn("evidence_indices: [1]", output.getvalue())
        self.assertIn("=== Rerank Top10 ===", output.getvalue())
        self.assertIn("=== Selected Evidence ===", output.getvalue())
        self.assertIn("=== Final Answer ===", output.getvalue())
        self.assertNotIn("Evidence insufficient:", output.getvalue())

    def test_build_index_loads_persisted_context_before_index_construction(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "objects.json"
            context_path = root / "contexts.json"
            input_path.write_text(
                json.dumps([{"type": "text", "text": "付款正文"}], ensure_ascii=False),
                encoding="utf-8",
            )
            context_path.write_text(
                json.dumps(
                    [{
                        "source_object_index": 0,
                        "retrieval_context": "位于付款条款相关章节",
                    }],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                retrieval_evaluation,
                "VectorStoreIndex",
                side_effect=lambda nodes, embed_model: SimpleNamespace(nodes=nodes),
            ):
                index = retrieval_evaluation.build_index(input_path, context_path)

        self.assertEqual(
            index.nodes[0].metadata["retrieval_context"],
            "位于付款条款相关章节",
        )
