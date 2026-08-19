from __future__ import annotations

import json
from typing import Any

from app.contract_review.schemas import Evidence, ReviewItem


def build_parse_review_rules_prompt(review_rule_text: str) -> str:
    return f"""你需要把用户提供的合同审查规范拆分为独立、可执行的审查项。

要求：
1. 只能拆分、整理输入规范中明确存在的标准，不得增加任何法律知识、行业惯例、阈值或其他审查标准。
2. 每个 rule_basis 必须能直接追溯到输入规范的原意。
3. review_goal 说明需要对合同事实作出的判断。
4. retrieval_query 只用于从合同中检索相关事实，不要预设风险结论。
5. 不合并相互独立的标准，也不要把同一标准重复拆分。
6. 只返回严格 JSON Schema 要求的字段。

审查规范原文：
<review_rule_text>
{review_rule_text}
</review_rule_text>
"""


def build_review_item_prompt(
    item: ReviewItem,
    evidence: list[Evidence],
) -> str:
    evidence_payload: list[dict[str, Any]] = [
        item.model_dump(mode="json") for item in evidence
    ]
    return f"""你需要依据当前风险规范和真实合同证据判断风险。

要求：
1. 只使用当前 rule_basis、review_goal 和合同证据，不得增加输入中不存在的审查标准。
2. 不得生成、猜测、复制或改写 source_object_index、page_idx、node_type 等证据引用字段；证据由程序附加。
3. 如果证据不足以支持风险或无风险结论，返回 risk_status=needs_review、evidence_status=insufficient、risk_level=null。
4. 没有检索到证据不等于合同没有约定，不得作此推断。
5. risk_status=risk 时必须给出 high、medium 或 low；其他状态的 risk_level 必须为 null。
6. 只返回严格 JSON Schema 要求的判断字段。

rule_basis：
{item.rule_basis}

review_goal：
{item.review_goal}

真实合同 Evidence：
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}
"""
