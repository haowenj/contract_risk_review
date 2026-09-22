from __future__ import annotations

import json

from app.evidence_review.rule_import import ExtractedRuleDocument


def build_rule_parse_prompt(document: ExtractedRuleDocument) -> str:
    """Build the generic, evidence-preserving rule-document parse prompt."""
    payload = json.dumps(
        {
            "title_hint": document.title_hint,
            "page_count": document.page_count,
            "blocks": document.blocks,
        },
        ensure_ascii=False,
    )
    return f"""你是合同风险审查规则结构化助手。只整理输入文件明确写出的内容，
输出必须完全符合给定 JSON Schema。

解析原则：
1. 不得假定固定章节、编号格式、层级或项目数量。应从当前文件动态识别结构，兼容中文序号、
   阿拉伯数字、多级编号、带圈数字、项目符号和无编号条目。
2. source_number 必须原样保留条目前的编号或项目符号；原文无编号时填 null。
3. 父级目录、阶段名和分类标题只放入 sections。只有表达完整检查要求的叶子条目才放入
   review_items，不要把父级标题重复生成为检查项。
4. 要求检查合同或材料事实的叶子项标为 review_check；仅要求完成审批、报备、会签、
   前置程序或其他流程动作的叶子项标为 process_control。
5. rule_text 应忠实保留原规则含义；不得补充法律知识、行业阈值或公司制度，也不得根据常识
   发明文件中不存在的判断标准。
6. evidence_scope 表示判断所需证据来自合同、内部材料、外部查询或组合来源；无法仅凭文件
   确认外部结论时，只生成所需查询资料，不提前给出风险结论。
7. retrieval_queries 应是后续从合同或材料中查找原文证据的问题。fact_requirements 描述需要
   提取的结构化事实；research_requirements 描述后续人工或联网查询所需的主体、字段、主题
   和比对点。只根据规则原文生成。
8. section_id 和 item_id 在本次结果内必须唯一且稳定；source_pages 使用从 1 开始的页码。

<rule_document>
{payload}
</rule_document>"""
