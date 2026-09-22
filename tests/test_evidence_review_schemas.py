from __future__ import annotations

from pydantic import ValidationError
import pytest

from app.evidence_review.schemas import (
    EvidencePackage,
    HumanDecision,
    RuleParseResult,
)


def rule_payload() -> dict:
    return {
        "document_title": "供应合同审核规则",
        "sections": [
            {
                "section_id": "s1",
                "source_number": "第一章",
                "title": "付款条件",
                "parent_section_id": None,
                "level": 1,
                "source_pages": [1],
            }
        ],
        "review_items": [
            {
                "item_id": "item_1",
                "source_number": "（一）",
                "section_path": ["付款条件"],
                "name": "付款周期",
                "rule_text": "付款周期过长",
                "source_pages": [1],
                "item_kind": "review_check",
                "evidence_scope": "contract",
                "decision_mode": "threshold_required",
                "retrieval_queries": ["合同约定的付款周期是多久"],
                "fact_requirements": [
                    {
                        "fact_key": "payment_days",
                        "label": "付款周期",
                        "value_type": "integer",
                        "required": True,
                    }
                ],
                "research_requirements": [],
            }
        ],
    }


def test_unknown_title_and_numbering_are_valid():
    result = RuleParseResult.model_validate(rule_payload())

    assert result.document_title == "供应合同审核规则"
    assert result.review_items[0].source_number == "（一）"


def test_duplicate_item_ids_are_invalid():
    payload = rule_payload()
    payload["review_items"].append(dict(payload["review_items"][0]))

    with pytest.raises(ValidationError, match="unique"):
        RuleParseResult.model_validate(payload)


def test_extra_rule_fields_are_rejected():
    payload = rule_payload()
    payload["review_items"][0]["risk_status"] = "risk"

    with pytest.raises(ValidationError, match="Extra inputs"):
        RuleParseResult.model_validate(payload)


def test_fact_cannot_reference_missing_evidence():
    with pytest.raises(ValidationError, match="evidence_indices"):
        EvidencePackage.model_validate(
            {
                "rule_item_id": "item_1",
                "evidence_status": "found",
                "evidence": [],
                "extracted_facts": [
                    {
                        "fact_key": "payment_days",
                        "label": "付款周期",
                        "value": "180",
                        "unit": "日",
                        "evidence_indices": [0],
                    }
                ],
                "missing_sources": [],
                "research_package": None,
                "item_error": None,
            }
        )


def test_pending_research_disallows_risk():
    with pytest.raises(ValidationError, match="pending research"):
        HumanDecision(
            human_review_status="completed",
            decision="risk",
            risk_level="high",
            opinion="存在风险",
            research_status="pending",
            research_notes="",
            updated_at="2026-09-22T08:00:00+00:00",
        )


def test_completed_decision_requires_opinion():
    with pytest.raises(ValidationError, match="opinion"):
        HumanDecision(
            human_review_status="completed",
            decision="cannot_determine",
            risk_level=None,
            opinion="",
            research_status="pending",
            research_notes="等待查询",
            updated_at="2026-09-22T08:00:00+00:00",
        )


def test_non_risk_decision_rejects_risk_level():
    with pytest.raises(ValidationError, match="risk_level"):
        HumanDecision(
            human_review_status="completed",
            decision="no_obvious_risk",
            risk_level="low",
            opinion="未发现明显风险",
            research_status="not_required",
            research_notes="",
            updated_at="2026-09-22T08:00:00+00:00",
        )
