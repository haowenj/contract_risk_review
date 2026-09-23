import json

from app.evidence_review.prompts import build_rule_parse_prompt
from app.evidence_review.rule_import import ExtractedRuleDocument


def test_rule_parse_prompt_requires_generic_structure_discovery():
    document = ExtractedRuleDocument(
        title_hint="任意业务规则",
        blocks=[
            {"page_number": 1, "text": "一、商务条件\n① 付款条件"},
            {"page_number": 2, "text": "未编号的风险描述"},
        ],
        page_count=2,
    )

    prompt = build_rule_parse_prompt(document)

    assert "不得假定固定章节、编号格式、层级或项目数量" in prompt
    assert "不得补充法律知识、行业阈值或公司制度" in prompt
    assert "父级目录" in prompt
    assert "叶子" in prompt
    assert "process_control" in prompt
    assert "review_items 必须是顶层数组" in prompt
    assert "不得嵌套在 sections" in prompt
    assert "完整编号路径" in prompt
    assert "查询标识" in prompt
    assert "企业全称" in prompt
    assert "一、商务条件" in prompt
    assert "未编号的风险描述" in prompt


def test_rule_parse_prompt_embeds_pages_as_machine_readable_json():
    document = ExtractedRuleDocument(
        title_hint='包含"引号"的标题',
        blocks=[{"page_number": 3, "text": "1.2 争议解决\n内容"}],
        page_count=3,
    )

    prompt = build_rule_parse_prompt(document)
    embedded = prompt.split("<rule_document>\n", 1)[1].split(
        "\n</rule_document>", 1
    )[0]

    assert json.loads(embedded) == {
        "title_hint": '包含"引号"的标题',
        "page_count": 3,
        "blocks": [{"page_number": 3, "text": "1.2 争议解决\n内容"}],
    }


def test_rule_parse_prompt_does_not_contain_sample_specific_assumptions():
    prompt = build_rule_parse_prompt(
        ExtractedRuleDocument(
            title_hint="通用规则",
            blocks=[{"page_number": 1, "text": "自定义检查项"}],
            page_count=1,
        )
    )

    forbidden = ["42项", "四十二项", "中建", "负面清单必须", "投标阶段必须"]
    assert all(value not in prompt for value in forbidden)


def test_rule_parse_prompt_spells_out_nested_response_contract():
    prompt = build_rule_parse_prompt(
        ExtractedRuleDocument(
            title_hint="规则",
            blocks=[{"page_number": 1, "text": "核验合同相对方"}],
            page_count=1,
        )
    )

    assert "fact_requirements 必须是对象数组" in prompt
    assert "research_requirements 必须是对象数组" in prompt
    assert "review_items 中禁止 section_id" in prompt
    assert "retrieval_queries 至少包含一个检索问题" in prompt
