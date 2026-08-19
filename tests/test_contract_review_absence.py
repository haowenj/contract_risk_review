import pytest

from app.contract_review.absence import scan_source_objects


def test_scan_source_objects_covers_text_table_and_image_with_real_indices():
    source_objects = [
        {"type": "text", "text": "普通标题", "page_idx": 0},
        {"type": "text", "text": "未经甲方书面同意不得转委托", "page_idx": 1},
        {
            "type": "table",
            "table_caption": ["分包审批"],
            "table_body": "<table><tr><td>第三方履行须书面批准</td></tr></table>",
            "table_footnote": ["禁止转包"],
            "page_idx": 2,
        },
        {
            "type": "image",
            "image_type": "general",
            "structured_data": {
                "content_description": "外包限制告知书",
                "visible_text": "委托第三方履行",
            },
            "page_idx": 3,
        },
        {"type": "aside_text", "text": "分包", "page_idx": 4},
    ]

    result = scan_source_objects(
        source_objects,
        primary_keywords=["转委托", "分包审批", "外包限制"],
        secondary_keywords=["第三方履行", "禁止转包", "书面批准"],
    )

    assert result.candidate_count == 3
    assert [item["source_object_index"] for item in result.candidates] == [2, 3, 1]
    assert result.candidates[0]["node_type"] == "table"
    assert result.candidates[0]["matched_primary_keywords"] == ["分包审批"]
    assert result.candidates[0]["matched_secondary_keywords"] == [
        "第三方履行",
        "禁止转包",
        "书面批准",
    ]
    assert result.candidates[0]["matched_keywords"] == [
        "分包审批",
        "第三方履行",
        "禁止转包",
        "书面批准",
    ]
    assert result.candidates[1]["node_type"] == "image"


def test_scan_normalizes_nfkc_case_and_whitespace_without_fuzzy_matching():
    objects = [
        {"type": "text", "text": "ＡＢＣ\n第三方   履行", "page_idx": 0},
        {"type": "text", "text": "转委拖", "page_idx": 1},
    ]

    result = scan_source_objects(
        objects,
        primary_keywords=["abc", "转委托"],
        secondary_keywords=["第三方履行"],
    )

    assert result.candidate_count == 1
    assert result.candidates[0]["source_object_index"] == 0
    assert result.candidates[0]["matched_keywords"] == ["abc", "第三方履行"]


def test_secondary_keyword_cannot_create_candidate_without_primary_match():
    objects = [
        {"type": "text", "text": "未经甲方书面同意向第三方转让", "page_idx": 0},
        {"type": "text", "text": "乙方分包须经甲方书面同意", "page_idx": 1},
    ]

    result = scan_source_objects(
        objects,
        primary_keywords=["分包"],
        secondary_keywords=["第三方", "转让", "书面同意"],
    )

    assert result.candidate_count == 1
    assert result.candidates[0]["source_object_index"] == 1
    assert result.candidates[0]["matched_primary_keywords"] == ["分包"]
    assert result.candidates[0]["matched_secondary_keywords"] == ["书面同意"]


def test_scan_reports_total_count_before_twenty_candidate_limit():
    objects = [
        {"type": "text", "text": f"分包限制 {index}", "page_idx": index}
        for index in range(25)
    ]

    result = scan_source_objects(
        objects,
        primary_keywords=["分包限制"],
        secondary_keywords=[],
        limit=20,
    )

    assert result.candidate_count == 25
    assert len(result.candidates) == 20
    assert [item["source_object_index"] for item in result.candidates] == list(range(20))


def test_scan_rejects_invalid_limit_and_non_object_member():
    with pytest.raises(ValueError, match="limit"):
        scan_source_objects([], ["分包"], [], limit=0)
    with pytest.raises(ValueError, match="primary_keywords"):
        scan_source_objects([], [], ["书面同意"])
    with pytest.raises(ValueError, match="JSON objects"):
        scan_source_objects(["not-an-object"], ["分包"], [])
