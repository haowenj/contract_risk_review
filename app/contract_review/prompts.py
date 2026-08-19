from __future__ import annotations

import json
from typing import Any

from app.contract_review.schemas import Evidence, ReviewItem, RiskDecision


def build_parse_review_rules_prompt(review_rule_text: str) -> str:
    return f"""你需要把用户提供的合同审查规范拆分为独立、可执行的审查项。

要求：
1. 只能拆分、整理输入规范中明确存在的标准，不得增加任何法律知识、行业惯例、阈值或其他审查标准。
2. 每个 rule_basis 必须能直接追溯到输入规范的原意。
3. review_goal 说明需要对合同事实作出的判断。
4. retrieval_query 只用于从合同中检索相关事实，不要预设风险结论。
5. 不合并相互独立的标准，也不要把同一标准重复拆分。
6. 每个审查项必须包含且只能包含 id、name、rule_basis、review_goal、retrieval_query 五个字段。
7. id 按 item_1、item_2、item_3 的格式依次生成；name 是输入规范中该项标准的简短名称。
8. 顶层必须是 JSON 对象，且只能包含 review_items 数组；不得直接返回数组。

JSON 协议：
{{
  "review_items": [
    {{
      "id": "item_1",
      "name": "审查项名称",
      "rule_basis": "输入规范中明确存在的标准",
      "review_goal": "需要根据合同事实判断的目标",
      "retrieval_query": "用于检索合同事实的问题"
    }}
  ]
}}

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
6. 顶层必须是 JSON 对象，只能包含 risk_status、risk_level、evidence_status、finding、risk_description、suggestion 六个字段。
7. risk_status 只能是 risk、no_obvious_risk、needs_review；evidence_status 只能是 found、insufficient。
8. 不得输出 evidence 数组或任何证据引用字段。

JSON 协议：
{{
  "risk_status": "risk | no_obvious_risk | needs_review",
  "risk_level": "high | medium | low | null",
  "evidence_status": "found | insufficient",
  "finding": "基于证据的审查发现",
  "risk_description": "风险说明",
  "suggestion": "修改或人工核对建议"
}}

rule_basis：
{item.rule_basis}

review_goal：
{item.review_goal}

真实合同 Evidence：
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}
"""


def build_retrieval_query_rewrite_prompt(
    item: ReviewItem,
    *,
    attempted_queries: list[str],
    evidence: list[Evidence],
    decision: RiskDecision | None,
) -> str:
    evidence_payload = [value.model_dump(mode="json") for value in evidence]
    insufficient_context = (
        decision.model_dump(mode="json")
        if decision is not None
        else {"reason": "当前检索未返回合同证据"}
    )
    return f"""你需要为同一个合同审查项改写检索问题，以寻找第一轮遗漏的合同表达。

要求：
1. 只能改写检索方式，不得增加、修改或放宽 rule_basis 中的审查标准。
2. 使用合同中可能出现的近义词、章节名称、主体称谓和履约表达扩展查询。
3. 结合已尝试查询、已有 Evidence 和证据不足原因，避免重复原查询。
4. 生成一个适合 Vector → Rerank → Evidence Selector 链路的自然语言检索问题。
5. 顶层必须是 JSON 对象，只能包含 retrieval_query 和 reason 两个字段。

JSON 协议：
{{
  "retrieval_query": "改写后的单个合同检索问题",
  "reason": "本次改写覆盖了哪些遗漏表达"
}}

rule_basis：
{item.rule_basis}

review_goal：
{item.review_goal}

已尝试查询：
{json.dumps(attempted_queries, ensure_ascii=False, indent=2)}

已有合同 Evidence：
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}

证据不足上下文：
{json.dumps(insufficient_context, ensure_ascii=False, indent=2)}
"""
