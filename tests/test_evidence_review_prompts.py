import json

from app.evidence_review.prompts import build_fact_extraction_prompt
from app.evidence_review.schemas import Evidence, RuleItem


def item() -> RuleItem:
    return RuleItem.model_validate(
        {
            "item_id": "company-check",
            "source_number": "1.1",
            "section_path": ["主体资格"],
            "name": "相对方主体信息",
            "rule_text": "结合合同和公开信息核验相对方主体状态",
            "source_pages": [1],
            "item_kind": "review_check",
            "evidence_scope": "hybrid",
            "decision_mode": "query_and_compare",
            "retrieval_queries": ["合同相对方的名称和统一社会信用代码"],
            "fact_requirements": [
                {
                    "fact_key": "counterparty_name",
                    "label": "相对方公司名称",
                    "value_type": "string",
                    "required": True,
                }
            ],
            "research_requirements": [],
        }
    )


def test_fact_prompt_requires_evidence_only_and_forbids_conclusions_and_metadata():
    prompt = build_fact_extraction_prompt(
        item(),
        [
            Evidence(
                source_object_index=8,
                page_idx=2,
                node_type="text",
                evidence_text="甲方：示例工程有限公司",
            )
        ],
    )

    assert "只能提取 Evidence 文本直接支持的事实" in prompt
    assert "不得输出风险、无风险、风险等级、判断、建议或法律结论" in prompt
    assert "不得输出或改写 source_object_index、page_idx、node_type" in prompt
    assert "evidence_indices" in prompt
    assert "甲方：示例工程有限公司" in prompt
    assert '"source_object_index": 8' not in prompt
    assert '"page_idx": 2' not in prompt


def test_fact_prompt_embeds_only_requested_fact_schema():
    prompt = build_fact_extraction_prompt(
        item(),
        [
            Evidence(
                source_object_index=8,
                page_idx=2,
                node_type="text",
                evidence_text="甲方：示例工程有限公司",
            )
        ],
    )
    requirements_text = prompt.split("<fact_requirements>\n", 1)[1].split(
        "\n</fact_requirements>", 1
    )[0]

    assert json.loads(requirements_text) == [
        {
            "fact_key": "counterparty_name",
            "label": "相对方公司名称",
            "value_type": "string",
            "required": True,
        }
    ]
