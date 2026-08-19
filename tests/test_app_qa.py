import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_EMBEDDING_MODEL", "test-embedding-model")
os.environ.setdefault("LLM_MODEL", "test-answer-model")
os.environ.setdefault("LLM_RERANK_MODEL", "qwen3-rerank")

from app.qa import answer_question
from app.config import Settings
from app.db import ContractRepository
from app.service import (
    ContractNotFoundError,
    ContractNotReadyError,
    ContractService,
)


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


def table_result_for(index: int, text: str, score: float = 0.8):
    node = SimpleNamespace(
        node_id=f"table-node-{index}",
        text=text,
        metadata={
            "node_type": "table",
            "source_object_index": index,
            "page_idx": 2,
            "bbox": [1, 2, 3, 4],
            "table_body": "<table><tr><td>30%</td></tr></table>",
            "table_caption": ["付款计划"],
            "table_footnote": ["以到账为准"],
            "img_path": "images/payment-table.jpg",
        },
    )
    return SimpleNamespace(node=node, score=score)


def image_result_for(index: int, text: str, score: float = 0.9):
    node = SimpleNamespace(
        node_id=f"image-node-{index}",
        text=text,
        metadata={
            "node_type": "image",
            "source_object_index": index,
            "page_idx": 4,
            "bbox": [1, 2, 3, 4],
            "img_path": "images/account.jpg",
            "image_type": "bank_account",
            "structured_data": {
                "account_name": "甲公司",
                "account_number": "110914414810101",
                "bank_name": "甲银行",
                "bank_branch": None,
            },
            "ocr_text": "账号 110914414810101",
            "ocr_status": "ready",
            "verification_status": "verified",
            "verification_details": {
                "account_number": {"status": "verified"}
            },
            "image_processing_status": "ready",
            "retrieval_context": "开户资料章节",
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


class EvidenceOnlyPipeline:
    def __init__(self, selected_nodes, reranked_results=None):
        self.selected_nodes = selected_nodes
        self.reranked_results = (
            selected_nodes if reranked_results is None else reranked_results
        )
        self.calls = []

    def retrieve_evidence(
        self,
        index,
        query,
        *,
        fallback_on_empty_selection=True,
    ):
        self.calls.append((index, query, fallback_on_empty_selection))
        return {
            "selected_nodes": self.selected_nodes,
            "reranked_results": self.reranked_results,
        }

    def run(self, *args, **kwargs):
        raise AssertionError("answer pipeline must not be called")


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

    def test_answer_question_serializes_table_original_information_in_evidence(self):
        result = answer_question(
            FakeIndex([table_result_for(7, "第1行：付款比例 | 30%")]),
            "付款比例是多少？",
            reranker=FakeReranker(),
            selector_llm=FakeLLM({"evidence_indices": [7]}),
            answer_llm=FakeLLM({"answer": "30%"}),
        )

        evidence = result["evidence"][0]
        self.assertEqual(evidence["node_type"], "table")
        self.assertEqual(
            evidence["table_body"],
            "<table><tr><td>30%</td></tr></table>",
        )
        self.assertEqual(evidence["table_caption"], ["付款计划"])
        self.assertEqual(evidence["table_footnote"], ["以到账为准"])
        self.assertEqual(evidence["bbox"], [1, 2, 3, 4])
        self.assertEqual(evidence["img_path"], "images/payment-table.jpg")

    def test_answer_question_serializes_image_reference_and_verification(self):
        result = answer_question(
            FakeIndex([image_result_for(12, "银行账号：110914414810101")]),
            "账号是什么？",
            debug=True,
            reranker=FakeReranker(),
            selector_llm=FakeLLM({"evidence_indices": [12]}),
            answer_llm=FakeLLM({"answer": "账号为110914414810101。"}),
        )

        evidence = result["evidence"][0]
        self.assertEqual(evidence["node_type"], "image")
        self.assertEqual(evidence["source_object_index"], 12)
        self.assertEqual(evidence["img_path"], "images/account.jpg")
        self.assertEqual(evidence["image_type"], "bank_account")
        self.assertEqual(evidence["structured_data"]["account_number"], "110914414810101")
        self.assertEqual(evidence["verification_status"], "verified")
        self.assertEqual(evidence["evidence_text"], "银行账号：110914414810101")
        self.assertEqual(result["debug"]["rerank_top10"][0]["img_path"], "images/account.jpg")


class ContractSearchTest(TestCase):
    def _build_service(self, root: Path, selected_nodes):
        settings = Settings(
            project_dir=root,
            data_dir=root / "data",
            database_path=root / "data" / "contracts.db",
            contracts_dir=root / "data" / "contracts",
            mineru_url="http://mineru.test",
            mineru_backend="hybrid-engine",
            mineru_server_url=None,
        )
        repository = ContractRepository(settings.database_path)
        index_manager = Mock()
        index_manager.get.return_value = object()
        pipeline = EvidenceOnlyPipeline(selected_nodes)
        service = ContractService(
            repository,
            settings,
            Mock(),
            index_manager,
            rag_pipeline=pipeline,
        )
        return service, pipeline, index_manager

    def test_search_contract_returns_serialized_rag_evidence_without_answer(self):
        with TemporaryDirectory() as temp_dir:
            service, pipeline, index_manager = self._build_service(
                Path(temp_dir),
                [image_result_for(12, "银行账号：110914414810101")],
            )
            contract = service.repository.create(
                "contract.pdf",
                Path(temp_dir) / "contract",
            )
            ready = service.repository.update_status(contract.contract_id, "ready")

            evidence = service.search_contract(contract.contract_id, "收款账号")

        self.assertEqual(evidence[0]["source_object_index"], 12)
        self.assertEqual(evidence[0]["page_idx"], 4)
        self.assertEqual(evidence[0]["node_type"], "image")
        index_manager.get.assert_called_once_with(ready)
        self.assertEqual(
            pipeline.calls,
            [(index_manager.get.return_value, "收款账号", False)],
        )

    def test_search_contract_reports_rerank_top_three_only_when_evidence_is_empty(self):
        with TemporaryDirectory() as temp_dir:
            reranked_results = [
                result_for(1, "候选一", 0.9),
                result_for(2, "候选二", 0.8),
                result_for(3, "候选三", 0.7),
                result_for(4, "候选四", 0.6),
            ]
            service, _, _ = self._build_service(Path(temp_dir), [])
            service.rag_pipeline.reranked_results = reranked_results
            contract = service.repository.create(
                "contract.pdf",
                Path(temp_dir) / "contract",
            )
            service.repository.update_status(contract.contract_id, "ready")
            debug_payloads = []

            evidence = service.search_contract(
                contract.contract_id,
                "分包约定",
                debug_callback=debug_payloads.append,
            )

        self.assertEqual(evidence, [])
        self.assertEqual(
            debug_payloads,
            [[
                {"source_object_index": 1, "text": "候选一"},
                {"source_object_index": 2, "text": "候选二"},
                {"source_object_index": 3, "text": "候选三"},
            ]],
        )

    def test_search_contract_rejects_blank_query(self):
        with TemporaryDirectory() as temp_dir:
            service, _, index_manager = self._build_service(Path(temp_dir), [])

            with self.assertRaisesRegex(ValueError, "query must not be empty"):
                service.search_contract("contract-id", "  ")

        index_manager.get.assert_not_called()

    def test_search_contract_requires_existing_ready_contract(self):
        with TemporaryDirectory() as temp_dir:
            service, _, index_manager = self._build_service(Path(temp_dir), [])

            with self.assertRaises(ContractNotFoundError):
                service.search_contract("missing", "问题")

            contract = service.repository.create(
                "contract.pdf",
                Path(temp_dir) / "contract",
            )
            with self.assertRaises(ContractNotReadyError):
                service.search_contract(contract.contract_id, "问题")

        index_manager.get.assert_not_called()

    def test_load_contract_content_objects_reads_ready_merged_content(self):
        with TemporaryDirectory() as temp_dir:
            service, _, index_manager = self._build_service(Path(temp_dir), [])
            storage_dir = Path(temp_dir) / "contract"
            storage_dir.mkdir()
            expected = [{"type": "text", "text": "条款", "page_idx": 0}]
            (storage_dir / "merged_content_list.json").write_text(
                json.dumps(expected, ensure_ascii=False),
                encoding="utf-8",
            )
            contract = service.repository.create("contract.pdf", storage_dir)
            service.repository.update_status(contract.contract_id, "ready")

            actual = service.load_contract_content_objects(contract.contract_id)

        self.assertEqual(actual, expected)
        index_manager.get.assert_not_called()

    def test_load_contract_content_objects_rejects_missing_or_non_ready_contract(self):
        with TemporaryDirectory() as temp_dir:
            service, _, index_manager = self._build_service(Path(temp_dir), [])
            with self.assertRaises(ContractNotFoundError):
                service.load_contract_content_objects("missing")

            contract = service.repository.create(
                "contract.pdf",
                Path(temp_dir) / "queued-contract",
            )
            with self.assertRaises(ContractNotReadyError):
                service.load_contract_content_objects(contract.contract_id)

        index_manager.get.assert_not_called()

    def test_load_contract_content_objects_rejects_missing_or_invalid_content_file(self):
        invalid_payloads = [
            "not-json",
            json.dumps({"type": "text"}),
            json.dumps([{"type": "text", "text": "条款"}, "invalid-member"]),
        ]
        with TemporaryDirectory() as temp_dir:
            service, _, index_manager = self._build_service(Path(temp_dir), [])
            missing_dir = Path(temp_dir) / "missing-file"
            missing_dir.mkdir()
            missing = service.repository.create("missing.pdf", missing_dir)
            service.repository.update_status(missing.contract_id, "ready")
            with self.assertRaises(FileNotFoundError):
                service.load_contract_content_objects(missing.contract_id)

            for index, payload in enumerate(invalid_payloads):
                storage_dir = Path(temp_dir) / f"invalid-{index}"
                storage_dir.mkdir()
                (storage_dir / "merged_content_list.json").write_text(
                    payload,
                    encoding="utf-8",
                )
                contract = service.repository.create(
                    f"invalid-{index}.pdf",
                    storage_dir,
                )
                service.repository.update_status(contract.contract_id, "ready")
                with self.subTest(payload=payload), self.assertRaises(
                    (json.JSONDecodeError, ValueError)
                ):
                    service.load_contract_content_objects(contract.contract_id)

        index_manager.get.assert_not_called()
