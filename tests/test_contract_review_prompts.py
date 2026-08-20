from app.contract_review.prompts import (
    build_absence_result_prompt,
    build_review_item_prompt,
    build_retrieval_query_rewrite_prompt,
)
from app.contract_review.schemas import ReviewItem


ITEM = ReviewItem.model_validate(
    {
        "id": "item_1",
        "name": "分包转包限制",
        "rule_basis": "乙方未经甲方书面同意不得分包或转包。",
        "review_goal": "判断合同是否明确限制乙方分包或转包。",
        "retrieval_query": "合同如何约定乙方分包或转包？",
    }
)


def test_rewrite_prompt_requires_discriminating_absence_keywords():
    prompt = build_retrieval_query_rewrite_prompt(
        ITEM,
        attempted_queries=[ITEM.retrieval_query],
        evidence=[],
        decision=None,
    )

    assert '"primary_keywords"' in prompt
    assert '"secondary_keywords"' in prompt
    assert "secondary_keywords 不能独立形成候选" in prompt
    assert "第三方、同意、批准、许可、转让" in prompt
    assert "核心术语或具有业务区分度的短语" in prompt
    assert "第三方履行、权利义务转让、分包须书面同意" in prompt
    assert "不得增加、修改或放宽 rule_basis" in prompt


def test_rewrite_prompt_prefers_concise_primary_topic_terms():
    prompt = build_retrieval_query_rewrite_prompt(
        ITEM,
        attempted_queries=[ITEM.retrieval_query],
        evidence=[],
        decision=None,
    )

    assert "分包、转包、转委托本身可以作为 primary_keywords" in prompt
    assert "不要为了增加长度而过度短语化" in prompt


def test_review_prompt_does_not_invent_risk_level_or_consequences():
    prompt = build_review_item_prompt(ITEM, evidence=[])

    assert "审查规范没有提供风险等级或分级依据" in prompt
    assert "risk_level=null" in prompt
    assert "只解释合同事实与 rule_basis 的偏离" in prompt
    assert "不得自行扩展法律后果、商业后果" in prompt


def test_absence_result_prompt_requires_bounded_absence_wording():
    prompt = build_absence_result_prompt(
        ITEM,
        primary_keywords=["分包", "转包", "转委托"],
        secondary_keywords=["第三方", "书面同意"],
    )

    assert "基于当前合同全文解析结果" in prompt
    assert "合同肯定没有" in prompt
    assert "绝对不存在、确认没有、完全不存在" in prompt
    assert "根本不存在、断定合同没有" in prompt
    assert "不得使用" in prompt
    assert "absence_verified" in prompt
    assert "两次语义检索" in prompt
    assert '"evidence"' not in prompt
    assert "审查规范没有提供风险等级或分级依据" in prompt
    assert "risk_level=null" in prompt
    assert "只解释核验事实与 rule_basis 的偏离" in prompt
    assert "不得自行扩展法律后果、商业后果" in prompt
