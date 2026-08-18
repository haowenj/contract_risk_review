import os
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_MODEL", "test-context-model")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")

import retrieval_context_preprocess
from mineru_to_nodes import load_retrieval_contexts


class RecordingLLM:
    def __init__(self, response="位于支付条款相关章节"):
        self.prompts = []
        self.response = response

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


class RetrievalContextPreprocessTest(TestCase):
    def test_context_generation_retries_until_success(self):
        attempts = []

        def generate_context(_chunk_text, _section_path):
            attempts.append(True)
            if len(attempts) < 3:
                raise TimeoutError("Request timed out")
            return "位于支付条款相关章节"

        contexts = retrieval_context_preprocess.generate_contexts(
            [{"type": "text", "text": "付款条款"}],
            context_generator=generate_context,
            concurrency=1,
        )

        self.assertEqual(len(attempts), 3)
        self.assertEqual(contexts[0], "位于支付条款相关章节")

    def test_context_generation_stops_after_three_attempts_and_uses_fallback(self):
        attempts = []

        def generate_context(_chunk_text, _section_path):
            attempts.append(True)
            raise TimeoutError("Request timed out")

        contexts = retrieval_context_preprocess.generate_contexts(
            [
                {"type": "text", "text": "付款条款", "text_level": 1},
                {"type": "text", "text": "付款失败"},
            ],
            context_generator=generate_context,
            concurrency=1,
        )

        self.assertEqual(len(attempts), 3 * 2)
        self.assertEqual(contexts[1], "文档章节：付款条款")

    def test_context_llm_uses_fast_non_thinking_configuration(self):
        self.assertEqual(retrieval_context_preprocess.context_llm.model_name, "qwen3.7-plus")
        self.assertEqual(retrieval_context_preprocess.context_llm.temperature, 0)
        self.assertEqual(
            retrieval_context_preprocess.context_llm.request_timeout,
            retrieval_context_preprocess.CONTEXT_LLM_TIMEOUT_SECONDS,
        )
        self.assertGreaterEqual(
            retrieval_context_preprocess.CONTEXT_LLM_TIMEOUT_SECONDS,
            120,
        )
        self.assertEqual(
            retrieval_context_preprocess.context_llm.extra_body.get("enable_thinking"),
            False,
        )

    def test_prompt_requires_explicit_evidence_and_avoids_chunk_fact_rewrite(self):
        llm = RecordingLLM()
        objects = [
            {"type": "text", "text": "8.2. 支付", "text_level": 2},
            {"type": "text", "text": "分（三）期支付："},
            {"type": "text", "text": "首期支付全部合同款的 50%。"},
        ]

        contexts = retrieval_context_preprocess.generate_contexts(
            objects,
            llm=llm,
            concurrency=1,
        )

        prompt_for_lead_in = llm.prompts[1]
        self.assertIn("8.2. 支付", prompt_for_lead_in)
        self.assertIn("分（三）期支付：", prompt_for_lead_in)
        self.assertNotIn("首期支付全部合同款的 50%", prompt_for_lead_in)
        self.assertIn("不得猜测", prompt_for_lead_in)
        self.assertIn("不要重复", prompt_for_lead_in)
        self.assertEqual(contexts[1], "位于支付条款相关章节")

    def test_context_generation_includes_table_searchable_text_and_section_path(self):
        llm = RecordingLLM(response="第四条付款表格")
        objects = [
            {"type": "text", "text": "第四条 付款方式", "text_level": 2},
            {
                "type": "table",
                "table_caption": ["付款计划"],
                "table_body": (
                    "<table><tr><th>比例</th></tr>"
                    "<tr><td>30%</td></tr></table>"
                ),
                "table_footnote": ["以到账为准"],
            },
        ]

        contexts = retrieval_context_preprocess.generate_contexts(
            objects,
            llm=llm,
            concurrency=1,
        )

        self.assertEqual(contexts[1], "第四条付款表格")
        table_prompt = llm.prompts[-1]
        self.assertIn("30%", table_prompt)
        self.assertIn("付款计划", table_prompt)
        self.assertIn("以到账为准", table_prompt)
        self.assertIn("第四条 付款方式", table_prompt)

    def test_persists_and_loads_context_by_source_object_index(self):
        contexts = {
            111: "位于付款条款相关章节",
            112: "位于付款条款相关章节",
        }

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "contexts.json"
            retrieval_context_preprocess.save_retrieval_contexts(
                contexts,
                output_path,
            )

            self.assertEqual(load_retrieval_contexts(output_path), contexts)

    def test_debug_output_focuses_on_requested_source_indices(self):
        objects = [
            {"type": "text", "text": "其他内容"},
            {"type": "text", "text": "付款条款原文"},
        ]
        contexts = {111: "付款章节 context"}
        source_objects = [(110, objects[0]), (111, objects[1])]
        output = StringIO()

        retrieval_context_preprocess.print_context_debug(
            source_objects,
            contexts,
            source_object_indices={111},
            file=output,
        )

        debug_text = output.getvalue()
        self.assertIn("source_object_index: 111", debug_text)
        self.assertIn("付款条款原文", debug_text)
        self.assertIn("付款章节 context", debug_text)
        self.assertNotIn("其他内容", debug_text)
