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

    def test_build_nodes_creates_one_table_node_with_raw_metadata_and_embedding_text(self):
        objects = [
            {
                "type": "table",
                "table_body": (
                    "<table><tr><td>付款比例</td><td>30%</td></tr></table>"
                ),
                "table_caption": ["付款计划"],
                "table_footnote": ["以到账为准"],
                "img_path": "images/payment-table.jpg",
                "page_idx": 2,
                "bbox": [1, 2, 3, 4],
            }
        ]

        node = mineru_to_nodes.build_nodes(
            objects,
            retrieval_contexts={0: "付款条款表格"},
        )[0]

        self.assertEqual(node.metadata["node_type"], "table")
        self.assertEqual(node.metadata["source_object_index"], 0)
        self.assertEqual(node.metadata["page_idx"], 2)
        self.assertEqual(node.metadata["bbox"], [1, 2, 3, 4])
        self.assertEqual(
            node.metadata["table_body"],
            objects[0]["table_body"],
        )
        self.assertEqual(node.metadata["table_caption"], ["付款计划"])
        self.assertEqual(node.metadata["table_footnote"], ["以到账为准"])
        self.assertEqual(node.metadata["img_path"], "images/payment-table.jpg")
        self.assertIn("付款比例", node.text)
        self.assertIn("30%", node.get_content(metadata_mode=MetadataMode.EMBED))
        self.assertIn(
            "付款条款表格",
            node.get_content(metadata_mode=MetadataMode.EMBED),
        )
        self.assertNotIn("table_body", node.get_content(metadata_mode=MetadataMode.EMBED))
        self.assertNotIn(
            "images/payment-table.jpg",
            node.get_content(metadata_mode=MetadataMode.EMBED),
        )

    def test_build_nodes_creates_image_node_with_reference_and_searchable_text(self):
        objects = [
            {
                "type": "image",
                "img_path": "images/account.jpg",
                "image_type": "bank_account",
                "structured_data": {
                    "account_name": "甲公司",
                    "account_number": "110914414810101",
                    "bank_name": "甲银行",
                    "bank_branch": None,
                },
                "ocr_text": "账号：110914414810101",
                "ocr_status": "ready",
                "verification_status": "verified",
                "verification_details": {
                    "account_number": {"status": "verified"}
                },
                "image_processing_status": "ready",
                "page_idx": 4,
                "bbox": [1, 2, 3, 4],
            }
        ]

        node = mineru_to_nodes.build_nodes(
            objects,
            retrieval_contexts={0: "位于开户资料章节"},
        )[0]

        self.assertEqual(node.metadata["node_type"], "image")
        self.assertEqual(node.metadata["source_object_index"], 0)
        self.assertEqual(node.metadata["img_path"], "images/account.jpg")
        self.assertEqual(node.metadata["image_type"], "bank_account")
        self.assertEqual(node.metadata["verification_status"], "verified")
        self.assertIn("110914414810101", node.text)
        embedding_text = node.get_content(metadata_mode=MetadataMode.EMBED)
        self.assertIn("位于开户资料章节", embedding_text)
        self.assertIn("110914414810101", embedding_text)
        self.assertNotIn("images/account.jpg", embedding_text)

    def test_build_nodes_skips_unusable_image_results(self):
        nodes = mineru_to_nodes.build_nodes(
            [
                {
                    "type": "image",
                    "image_processing_status": "vl_failed",
                    "structured_data": None,
                },
                {
                    "type": "image",
                    "image_type": "general",
                    "structured_data": {
                        "visible_text": None,
                        "content_description": "",
                    },
                },
            ]
        )

        self.assertEqual(nodes, [])

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
