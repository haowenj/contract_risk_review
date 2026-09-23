from __future__ import annotations

from collections.abc import Iterable, Mapping


EVIDENCE_STATUSES = (
    "found",
    "not_found",
    "source_missing",
    "extraction_failed",
)

IDENTIFIER_LABELS = {
    "credit_code": "统一社会信用代码",
    "owner_credit_code": "业主统一社会信用代码",
    "counterparty_credit_code": "相对方统一社会信用代码",
    "company_name": "企业名称",
    "owner_name": "业主名称",
    "parent_company_name": "母公司名称",
    "counterparty_name": "相对方名称",
    "project_name": "项目名称",
    "project_location": "项目所在地",
    "legal_representative": "法定代表人",
}


def summarize_evidence_statuses(entries: Iterable[Mapping]) -> dict[str, int]:
    counts = {status: 0 for status in EVIDENCE_STATUSES}
    for entry in entries:
        status = entry["evidence_package"]["evidence_status"]
        if status in counts:
            counts[status] += 1
    return counts


def identifier_label(key: str) -> str:
    return IDENTIFIER_LABELS.get(key, "其他查询信息")


def present_result_item(entry: Mapping) -> dict:
    rule = entry["rule_snapshot"]
    package = entry["evidence_package"]
    human = entry["human_decision"]
    research = package.get("research_package") or {}
    research_pending = human.get("research_status") == "pending"
    source_types = set(research.get("source_types") or [])
    online_queries = (
        list(research.get("query_topics") or [])
        if research_pending
        and source_types.intersection({"public_query", "professional_database"})
        else []
    )
    query_prerequisites = (
        [identifier_label(key) for key in research.get("missing_identifiers") or []]
        if online_queries
        else []
    )
    missing_materials = list(package.get("missing_sources") or [])
    if research_pending and "internal_material" in source_types:
        missing_materials.extend(research.get("query_topics") or [])
    missing_materials = list(dict.fromkeys(missing_materials))
    if len(missing_materials) > 1:
        missing_materials = [
            value for value in missing_materials
            if value != "规则要求的内部材料尚未提供"
        ]

    suggestion = package.get("system_suggestion") or {}
    evidence = package.get("evidence") or []
    first_evidence = evidence[0] if evidence else {}
    contract_page = first_evidence.get("page_idx")
    if contract_page is not None:
        contract_page += 1

    if suggestion.get("finding"):
        contract_label = "合同情况"
        contract_text = suggestion["finding"]
    elif evidence:
        contract_label = "合同摘录"
        contract_text = first_evidence["evidence_text"]
        if len(contract_text) > 200:
            contract_text = contract_text[:200].rstrip() + "…"
    else:
        contract_label = "合同情况"
        if package["evidence_status"] == "extraction_failed":
            contract_text = "系统未完成这项取证，请人工核对或重新校验。"
        elif rule.get("evidence_scope") in {"external_query", "internal_material"}:
            contract_text = "这项需要合同外信息，当前合同本身不足以判断。"
        else:
            contract_text = "当前未找到足以判断的合同条款。这不代表合同没有约定，请人工查阅全文和附件。"
            retrieval_queries = rule.get("retrieval_queries") or []
            if retrieval_queries:
                contract_text += f"重点核对：{retrieval_queries[0]}"

    decision = human.get("decision") if human.get("human_review_status") == "completed" else None
    if decision == "risk":
        verdict_key, verdict_label, verdict_source = "risk", "有风险", "人工结论"
    elif decision == "no_obvious_risk":
        verdict_key, verdict_label, verdict_source = (
            "no_obvious_risk", "无明显风险", "人工结论"
        )
    elif decision == "cannot_determine":
        verdict_key, verdict_label, verdict_source = (
            "needs_review", "暂无法判断", "人工结论"
        )
    elif online_queries:
        verdict_key, verdict_label, verdict_source = (
            "needs_review", "待联网查询", ""
        )
    elif missing_materials:
        verdict_key, verdict_label, verdict_source = (
            "needs_review", "待补资料", ""
        )
    elif package["evidence_status"] == "extraction_failed":
        verdict_key, verdict_label, verdict_source = (
            "needs_review", "处理失败", ""
        )
    elif research_pending:
        verdict_key, verdict_label, verdict_source = (
            "needs_review", "待查询核实", ""
        )
    elif suggestion.get("risk_status") == "risk":
        verdict_key, verdict_label, verdict_source = (
            "risk", "疑似有风险", "系统建议"
        )
    elif suggestion.get("risk_status") == "no_obvious_risk":
        verdict_key, verdict_label, verdict_source = (
            "no_obvious_risk", "暂未见明显风险", "系统建议"
        )
    elif package["evidence_status"] == "not_found":
        verdict_key, verdict_label, verdict_source = (
            "needs_review", "待核对合同", ""
        )
    else:
        verdict_key, verdict_label, verdict_source = (
            "needs_review", "待人工核实", ""
        )

    if online_queries:
        filter_status = "query"
    elif missing_materials:
        filter_status = "materials"
    elif verdict_key == "risk":
        filter_status = "risk"
    elif verdict_key == "no_obvious_risk":
        filter_status = "no_obvious_risk"
    else:
        filter_status = "review"

    return {
        "rule_item_id": rule["item_id"],
        "verdict_key": verdict_key,
        "verdict_label": verdict_label,
        "verdict_source": verdict_source,
        "contract_label": contract_label,
        "contract_text": contract_text,
        "contract_page": contract_page,
        "online_queries": online_queries,
        "query_prerequisites": query_prerequisites,
        "missing_materials": missing_materials,
        "filter_status": filter_status,
    }


def summarize_result_items(items: Iterable[Mapping]) -> dict[str, int]:
    result = {
        "total": 0,
        "risk": 0,
        "no_obvious_risk": 0,
        "needs_review": 0,
        "online_query": 0,
    }
    for item in items:
        result["total"] += 1
        result[item["verdict_key"]] += 1
        if item["online_queries"]:
            result["online_query"] += 1
    return result
