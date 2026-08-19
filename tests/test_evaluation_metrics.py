from types import SimpleNamespace

from app.evaluation_metrics import build_evaluation_result, serialize_pipeline_result


def result_for(source_object_index: int, text: str, score: float = 0.9):
    node = SimpleNamespace(
        node_id=f"node-{source_object_index}",
        text=text,
        metadata={
            "source_object_index": source_object_index,
            "page_idx": 4,
            "retrieval_context": f"context-{source_object_index}",
            "retrieval_score": 0.6,
        },
    )
    return SimpleNamespace(node=node, score=score)


def table_result_for(source_object_index: int, text: str, score: float = 0.9):
    node = SimpleNamespace(
        node_id=f"table-node-{source_object_index}",
        text=text,
        metadata={
            "node_type": "table",
            "source_object_index": source_object_index,
            "page_idx": 2,
            "bbox": [1, 2, 3, 4],
            "table_body": "<table><tr><td>30%</td></tr></table>",
            "table_caption": ["付款计划"],
            "table_footnote": ["以到账为准"],
            "img_path": "images/payment-table.jpg",
        },
    )
    return SimpleNamespace(node=node, score=score)


def image_result_for(source_object_index: int, text: str, score: float = 0.9):
    node = SimpleNamespace(
        node_id=f"image-node-{source_object_index}",
        text=text,
        metadata={
            "node_type": "image",
            "source_object_index": source_object_index,
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


def test_evaluation_metrics_compare_gold_only_after_pipeline_runs():
    result = {
        "query": "问题",
        "vector_results": [
            result_for(1, "一"),
            result_for(7, "七"),
            result_for(8, "八"),
        ],
        "reranked_results": [
            result_for(7, "七"),
            result_for(1, "一"),
            result_for(8, "八"),
        ],
        "selected_indices": [7],
        "selected_nodes": [result_for(7, "七")],
        "llm_summary": {"answer": "答案", "evidence_indices": [7]},
    }

    evaluated = build_evaluation_result(result, [7])

    assert evaluated["vector_recall_at_5"] == 1.0
    assert evaluated["rerank_recall_at_5"] == 1.0
    assert evaluated["vector_ranks"][7] == 2
    assert evaluated["rerank_ranks"][7] == 1
    assert evaluated["expected_source_object_indices"] == [7]


def test_empty_gold_recall_is_none():
    result = {
        "query": "问题",
        "vector_results": [result_for(1, "一")],
        "reranked_results": [result_for(1, "一")],
        "selected_indices": [],
        "selected_nodes": [],
        "llm_summary": {"answer": "证据不足", "evidence_indices": []},
    }

    evaluated = build_evaluation_result(result, [])

    assert evaluated["vector_recall_at_5"] is None
    assert evaluated["rerank_recall_at_10"] is None


def test_serialize_pipeline_result_keeps_node_and_score_metadata():
    result = {
        "query": "问题",
        "vector_results": [result_for(7, "证据", score=0.8)],
        "reranked_results": [result_for(7, "证据", score=0.95)],
        "selected_indices": [7],
        "selected_nodes": [result_for(7, "证据", score=0.95)],
        "llm_summary": {"answer": "答案", "evidence_indices": [7]},
    }

    serialized = serialize_pipeline_result(result)

    assert serialized["vector_results"][0]["source_object_index"] == 7
    assert serialized["vector_results"][0]["node_id"] == "node-7"
    assert serialized["reranked_results"][0]["score"] == 0.95
    assert serialized["selected_nodes"][0]["page_idx"] == 4


def test_serialize_pipeline_result_keeps_original_table_information():
    result = {
        "query": "问题",
        "vector_results": [table_result_for(7, "第1行：付款比例 | 30%")],
        "reranked_results": [table_result_for(7, "第1行：付款比例 | 30%")],
        "selected_indices": [7],
        "selected_nodes": [table_result_for(7, "第1行：付款比例 | 30%")],
        "llm_summary": {"answer": "答案", "evidence_indices": [7]},
    }

    serialized = serialize_pipeline_result(result)

    evidence = serialized["selected_nodes"][0]
    assert evidence["node_type"] == "table"
    assert evidence["bbox"] == [1, 2, 3, 4]
    assert evidence["table_body"].startswith("<table>")
    assert evidence["table_caption"] == ["付款计划"]
    assert evidence["table_footnote"] == ["以到账为准"]
    assert evidence["img_path"] == "images/payment-table.jpg"


def test_serialize_pipeline_result_keeps_image_reference_and_verification():
    image_result = image_result_for(12, "银行账号：110914414810101")
    result = {
        "query": "账号是什么？",
        "vector_results": [image_result],
        "reranked_results": [image_result],
        "selected_indices": [12],
        "selected_nodes": [image_result],
        "llm_summary": {"answer": "账号为110914414810101。", "evidence_indices": [12]},
    }

    serialized = serialize_pipeline_result(result)

    for stage in ("vector_results", "reranked_results", "selected_nodes"):
        evidence = serialized[stage][0]
        assert evidence["node_type"] == "image"
        assert evidence["source_object_index"] == 12
        assert evidence["page_idx"] == 4
        assert evidence["img_path"] == "images/account.jpg"
        assert evidence["image_type"] == "bank_account"
        assert evidence["structured_data"]["account_number"] == "110914414810101"
        assert evidence["verification_status"] == "verified"
        assert evidence["text"] == "银行账号：110914414810101"
