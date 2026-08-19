from app.contract_review.prompts import (
    build_absence_result_prompt,
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

    assert '"keywords"' in prompt
    assert "核心术语或具有业务区分度的短语" in prompt
    assert "不要单独输出“同意、批准、许可、责任、合同”" in prompt
    assert "书面同意、书面批准、书面许可、书面授权" in prompt
    assert "必须与当前审查主题组合" in prompt
    assert "不得增加、修改或放宽 rule_basis" in prompt


def test_absence_result_prompt_requires_bounded_absence_wording():
    prompt = build_absence_result_prompt(
        ITEM,
        keywords=["分包", "转包", "转委托", "委托第三方"],
    )

    assert "基于当前合同全文解析结果" in prompt
    assert "合同肯定没有" in prompt
    assert "不得使用" in prompt
    assert "absence_verified" in prompt
    assert "两次语义检索" in prompt
    assert '"evidence"' not in prompt
