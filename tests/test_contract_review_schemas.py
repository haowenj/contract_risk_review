import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.contract_review.schemas import (
    Evidence,
    ReviewItem,
    ReviewItemList,
    ReviewResult,
    RetrievalQueryRewrite,
    RiskDecision,
    parse_llm_response,
)


ITEM = {
    "id": "item_1",
    "name": "付款期限",
    "rule_basis": "付款期限不得超过90日",
    "review_goal": "判断合同付款期限是否超过90日",
    "retrieval_query": "合同约定的付款期限是多久",
}

DECISION = {
    "risk_status": "risk",
    "risk_level": "high",
    "evidence_status": "found",
    "finding": "合同约定付款期限为180日。",
    "risk_description": "付款期限超过规范上限。",
    "suggestion": "将付款期限调整为90日以内。",
}

EVIDENCE = {
    "source_object_index": 12,
    "page_idx": 4,
    "node_type": "text",
    "evidence_text": "未经甲方书面同意，乙方不得将合同义务转委托给第三方。",
}


def test_review_item_strips_non_empty_fields():
    item = ReviewItem.model_validate(
        {
            key: f"  {value}  "
            for key, value in ITEM.items()
        }
    )

    assert item.model_dump() == ITEM


def test_review_item_rejects_extra_llm_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReviewItem.model_validate({**ITEM, "invented_rule": "行业惯例"})


def test_review_item_list_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="review item ids must be unique"):
        ReviewItemList.model_validate({"review_items": [ITEM, ITEM]})


def test_review_item_list_rejects_empty_items():
    with pytest.raises(ValidationError):
        ReviewItemList.model_validate({"review_items": []})


def test_risk_decision_allows_null_level_for_risk_status():
    decision = RiskDecision.model_validate({**DECISION, "risk_level": None})

    assert decision.risk_status == "risk"
    assert decision.risk_level is None


def test_non_risk_decision_rejects_risk_level():
    with pytest.raises(ValidationError, match="risk_level"):
        RiskDecision.model_validate(
            {
                **DECISION,
                "risk_status": "no_obvious_risk",
                "risk_level": "low",
            }
        )


def test_insufficient_evidence_requires_needs_review():
    with pytest.raises(ValidationError, match="insufficient"):
        RiskDecision.model_validate(
            {
                **DECISION,
                "risk_status": "no_obvious_risk",
                "risk_level": None,
                "evidence_status": "insufficient",
            }
        )


def test_needs_review_with_insufficient_evidence_is_valid():
    decision = RiskDecision.model_validate(
        {
            **DECISION,
            "risk_status": "needs_review",
            "risk_level": None,
            "evidence_status": "insufficient",
        }
    )

    assert decision.risk_level is None


def test_evidence_preserves_rag_owned_table_and_image_metadata():
    evidence = Evidence.model_validate(
        {
            "source_object_index": 12,
            "page_idx": 4,
            "node_type": "image",
            "evidence_text": "银行账号：110914414810101",
            "text": "银行账号：110914414810101",
            "img_path": "images/account.jpg",
            "structured_data": {"account_number": "110914414810101"},
        }
    )

    assert evidence.model_dump()["img_path"] == "images/account.jpg"
    assert evidence.model_dump()["structured_data"]["account_number"] == "110914414810101"


def test_review_result_rechecks_cross_field_rules():
    with pytest.raises(ValidationError, match="insufficient"):
        ReviewResult.model_validate(
            {
                "item_id": "item_1",
                "item_name": "付款期限",
                **DECISION,
                "risk_status": "no_obvious_risk",
                "risk_level": None,
                "evidence_status": "insufficient",
                "evidence": [],
            }
        )


def test_parse_llm_response_accepts_content_json_and_parsed_payload():
    content_response = SimpleNamespace(
        content=json.dumps({"review_items": [ITEM]}, ensure_ascii=False),
        additional_kwargs={},
    )
    parsed_response = SimpleNamespace(
        content="ignored",
        additional_kwargs={"parsed": {"review_items": [ITEM]}},
    )

    assert parse_llm_response(content_response, ReviewItemList).review_items[0].id == "item_1"
    assert parse_llm_response(parsed_response, ReviewItemList).review_items[0].id == "item_1"


def test_retrieval_query_rewrite_strips_fields_and_rejects_extras():
    rewrite = RetrievalQueryRewrite.model_validate(
        {
            "retrieval_query": "  乙方权利义务及第三方履约约定  ",
            "reason": "  扩展分包和转委托的近义表达  ",
            "primary_keywords": ["  分包  ", "转包"],
            "secondary_keywords": ["  第三方  ", "书面同意"],
        }
    )

    assert rewrite.model_dump() == {
        "retrieval_query": "乙方权利义务及第三方履约约定",
        "reason": "扩展分包和转委托的近义表达",
        "primary_keywords": ["分包", "转包"],
        "secondary_keywords": ["第三方", "书面同意"],
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetrievalQueryRewrite.model_validate(
            {
                **rewrite.model_dump(),
                "new_rule": "未经同意一律视为高风险",
            }
        )


def test_retrieval_query_rewrite_normalizes_and_deduplicates_keyword_tiers():
    rewrite = RetrievalQueryRewrite.model_validate(
        {
            "retrieval_query": "检索乙方分包、转包和第三方履约限制",
            "reason": "覆盖同义表达",
            "primary_keywords": [
                " 分包 ",
                "转包",
                "ＦＥＮＢＡＯ",
                "fenbao",
                "",
                "转委托",
            ],
            "secondary_keywords": [" 第三方 ", "第三方", "", "书面同意"],
        }
    )

    assert rewrite.primary_keywords == ["分包", "转包", "ＦＥＮＢＡＯ", "转委托"]
    assert rewrite.secondary_keywords == ["第三方", "书面同意"]


def test_retrieval_query_rewrite_drops_generic_primary_but_allows_secondary():
    rewrite = RetrievalQueryRewrite.model_validate(
        {
            "retrieval_query": "检索乙方分包审批限制",
            "reason": "覆盖审批表达",
            "primary_keywords": [
                "分包",
                "第三方",
                "转让",
                "书面同意",
                "书面批准",
                "书面授权",
                "甲方书面同意",
                "须经甲方书面批准",
                "未经甲方书面授权",
                "事先取得书面许可",
                "经甲方书面同意后",
                "经甲方批准后方可",
                "合同",
                "分包须书面同意",
            ],
            "secondary_keywords": ["第三方", "转让", "书面同意", "批准", "许可"],
        }
    )

    assert rewrite.primary_keywords == ["分包", "分包须书面同意"]
    assert rewrite.secondary_keywords == ["第三方", "转让", "书面同意", "批准", "许可"]


def test_retrieval_query_rewrite_drops_punctuated_generic_primary_keywords():
    rewrite = RetrievalQueryRewrite.model_validate(
        {
            "retrieval_query": "检索乙方分包审批限制",
            "reason": "覆盖审批表达",
            "primary_keywords": [
                "同意。",
                "（批准）",
                "“合同”：",
                "乙方分包",
                "违约金5%",
                "违约金5％",
            ],
            "secondary_keywords": [],
        }
    )

    assert rewrite.primary_keywords == ["乙方分包", "违约金5%"]
    assert rewrite.secondary_keywords == []


@pytest.mark.parametrize(
    "primary_keywords",
    [[], ["", "   "], ["书面同意", "批准", "合同"]],
)
def test_retrieval_query_rewrite_rejects_unusable_primary_keywords(primary_keywords):
    with pytest.raises(ValidationError, match="primary_keywords"):
        RetrievalQueryRewrite.model_validate(
            {
                "retrieval_query": "检索乙方分包限制",
                "reason": "覆盖同义表达",
                "primary_keywords": primary_keywords,
                "secondary_keywords": [],
            }
        )


def test_absence_verified_review_result_preserves_scan_audit():
    result = ReviewResult.model_validate(
        {
            "item_id": "item_1",
            "item_name": "分包转包限制",
            "risk_status": "risk",
            "risk_level": "medium",
            "evidence_status": "absence_verified",
            "finding": "基于当前合同全文解析结果，未发现明确的分包转包限制条款。",
            "risk_description": "审查规范要求合同包含相关限制。",
            "suggestion": "建议补充明确限制条款。",
            "evidence": [],
            "absence_check": {
                "primary_keywords": [" 分包 ", "转包", "分包"],
                "secondary_keywords": ["第三方", "书面同意", "第三方"],
                "candidate_count": 0,
            },
        }
    )

    assert result.absence_check is not None
    assert result.absence_check.primary_keywords == ["分包", "转包"]
    assert result.absence_check.secondary_keywords == ["第三方", "书面同意"]
    assert result.absence_check.candidate_count == 0


@pytest.mark.parametrize(
    ("evidence", "absence_check"),
    [
        (
            [EVIDENCE],
            {
                "primary_keywords": ["分包"],
                "secondary_keywords": [],
                "candidate_count": 0,
            },
        ),
        ([], None),
        (
            [],
            {
                "primary_keywords": ["分包"],
                "secondary_keywords": [],
                "candidate_count": 1,
            },
        ),
    ],
)
def test_absence_verified_review_result_rejects_inconsistent_audit(
    evidence,
    absence_check,
):
    with pytest.raises(ValidationError, match="absence_verified"):
        ReviewResult.model_validate(
            {
                "item_id": "item_1",
                "item_name": "分包转包限制",
                "risk_status": "risk",
                "risk_level": "medium",
                "evidence_status": "absence_verified",
                "finding": "基于当前合同全文解析结果，未发现相关条款。",
                "risk_description": "规范要求合同包含相关限制。",
                "suggestion": "建议补充限制条款。",
                "evidence": evidence,
                "absence_check": absence_check,
            }
        )
