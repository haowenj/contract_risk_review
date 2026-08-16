import os
from io import StringIO
from unittest import TestCase

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")

import mineru_to_nodes
from llama_index.core.schema import MetadataMode


class MineruToNodesTest(TestCase):
    def test_default_paths_do_not_reference_deleted_contract_fixture(self):
        self.assertNotIn("test_hetong", str(mineru_to_nodes.INPUT_PATH))
        self.assertNotIn("test_hetong", str(mineru_to_nodes.RETRIEVAL_CONTEXT_PATH))

    def test_embedding_batch_size_matches_dashscope_limit(self):
        self.assertEqual(mineru_to_nodes.embedding_model.embed_batch_size, 10)

    def test_build_nodes_consumes_persisted_context_and_preserves_original_text(self):
        raw_text = "  (1) 首期付款为全部合同款的 50%。  \n"
        objects = [
            {
                "type": "text",
                "text": "8.2. 支付",
                "text_level": 2,
            },
            {
                "type": "text",
                "text": raw_text,
                "page_idx": 15,
                "start_page_idx": 14,
                "end_page_idx": 15,
                "source_page_indices": [14, 15],
                "source_bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
                "merged_cross_page": True,
            },
        ]
        contexts = {1: "位于支付条款相关章节"}

        nodes = mineru_to_nodes.build_nodes(
            objects,
            retrieval_contexts=contexts,
        )

        node = nodes[1]
        self.assertEqual(node.text, raw_text)
        self.assertEqual(
            node.metadata["retrieval_context"],
            "位于支付条款相关章节",
        )
        self.assertEqual(node.metadata["source_object_index"], 1)
        self.assertEqual(node.metadata["start_page_idx"], 14)
        self.assertEqual(node.metadata["end_page_idx"], 15)
        self.assertEqual(node.metadata["source_page_indices"], [14, 15])
        self.assertTrue(node.metadata["merged_cross_page"])

        embedding_text = node.get_content(metadata_mode=MetadataMode.EMBED)
        self.assertIn("位于支付条款相关章节", embedding_text)
        self.assertIn(raw_text.strip(), embedding_text)
        self.assertNotIn("page_idx", embedding_text)
        self.assertNotIn("bbox", embedding_text)
        self.assertNotIn("text_level", embedding_text)

    def test_missing_persisted_context_does_not_trigger_llm_generation(self):
        objects = [{"type": "text", "text": "正文"}]

        nodes = mineru_to_nodes.build_nodes(objects)

        self.assertEqual(len(nodes), 1)
        self.assertNotIn("retrieval_context", nodes[0].metadata)

    def test_debug_output_can_focus_on_source_indices(self):
        nodes = mineru_to_nodes.build_nodes(
            [
                {"type": "text", "text": "其他内容"},
                {"type": "text", "text": "付款条款原文"},
            ],
            retrieval_contexts={1: "付款章节 context"},
        )
        output = StringIO()

        mineru_to_nodes.print_embedding_debug(
            nodes,
            source_object_indices={1},
            file=output,
        )

        debug_text = output.getvalue()
        self.assertIn("付款条款原文", debug_text)
        self.assertIn("付款章节 context", debug_text)
        self.assertNotIn("其他内容", debug_text)
