import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.contract_review.schemas import (
    Evidence,
    ReviewItem,
    ReviewItemList,
    ReviewResult,
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


def test_risk_decision_requires_level_for_risk():
    with pytest.raises(ValidationError, match="risk_level"):
        RiskDecision.model_validate({**DECISION, "risk_level": None})


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
