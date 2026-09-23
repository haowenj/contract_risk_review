from app.evidence_review.presentation import present_result_item, summarize_result_items


def make_entry(*, evidence_status="found", suggestion=None, decision=None, research=None):
    return {
        "rule_snapshot": {
            "item_id": "rule-1",
            "name": "付款责任",
            "evidence_scope": "contract",
        },
        "evidence_package": {
            "evidence_status": evidence_status,
            "evidence": [
                {"page_idx": 3, "evidence_text": "乙方承担全部延期责任"}
            ] if evidence_status == "found" else [],
            "missing_sources": [],
            "research_package": research,
            "system_suggestion": suggestion,
        },
        "human_decision": decision or {
            "human_review_status": "pending",
            "decision": None,
            "opinion": "",
            "research_status": "pending" if research else "not_required",
        },
    }


def test_saved_legacy_suggestion_does_not_replace_a_human_verdict():
    entry = make_entry(suggestion={
        "risk_status": "risk",
        "finding": "延期责任全部由乙方承担",
        "suggestion": "核对责任范围",
    })

    proposed = present_result_item(entry)

    assert proposed["verdict_key"] == "needs_review"
    assert proposed["verdict_label"] == "待人工核实"
    assert proposed["contract_text"] == "乙方承担全部延期责任"
    assert proposed["contract_page"] == 4

    entry["human_decision"] = {
        "human_review_status": "completed",
        "decision": "no_obvious_risk",
        "opinion": "核对原文后未发现明显风险",
        "research_status": "not_required",
    }
    confirmed = present_result_item(entry)

    assert confirmed["verdict_key"] == "no_obvious_risk"
    assert confirmed["verdict_label"] == "无明显风险"
    assert confirmed["verdict_source"] == "人工结论"


def test_extracted_contract_facts_are_shown_without_a_new_model_judgment():
    entry = make_entry()
    entry["evidence_package"]["extracted_facts"] = [
        {"label": "付款期限", "value": "30", "unit": "天"}
    ]

    result = present_result_item(entry)

    assert result["verdict_key"] == "needs_review"
    assert result["contract_facts"] == ["付款期限：30 天"]


def test_external_query_is_a_todo_not_a_contract_no_risk_result():
    entry = make_entry(
        evidence_status="not_found",
        research={
            "source_types": ["public_query"],
            "query_topics": ["企业登记状态", "诉讼记录"],
            "missing_identifiers": ["credit_code"],
        },
    )
    entry["rule_snapshot"]["evidence_scope"] = "external_query"

    result = present_result_item(entry)

    assert result["verdict_key"] == "needs_review"
    assert result["verdict_label"] == "待联网查询"
    assert result["online_queries"] == ["企业登记状态", "诉讼记录"]
    assert result["query_prerequisites"] == ["统一社会信用代码"]
    assert "未检到" not in result["contract_text"]


def test_missing_contract_evidence_names_the_clause_to_check():
    entry = make_entry(evidence_status="not_found")
    entry["rule_snapshot"]["retrieval_queries"] = ["付款责任如何约定？"]

    result = present_result_item(entry)

    assert result["verdict_label"] == "待核对合同"
    assert "这不代表合同没有约定" in result["contract_text"]
    assert "重点核对：付款责任如何约定？" in result["contract_text"]


def test_specific_materials_replace_the_generic_missing_material_hint():
    entry = make_entry(
        evidence_status="source_missing",
        research={
            "source_types": ["internal_material"],
            "query_topics": ["项目审批记录"],
            "missing_identifiers": [],
        },
    )
    entry["evidence_package"]["missing_sources"] = ["规则要求的内部材料尚未提供"]

    result = present_result_item(entry)

    assert result["missing_materials"] == ["项目审批记录"]


def test_confirmed_risk_remains_in_risk_filter_when_source_was_missing():
    entry = make_entry(
        evidence_status="source_missing",
        decision={
            "human_review_status": "completed",
            "decision": "risk",
            "opinion": "补充材料后确认有风险",
            "research_status": "completed",
        },
    )
    entry["evidence_package"]["missing_sources"] = ["项目审批记录"]

    result = present_result_item(entry)

    assert result["verdict_key"] == "risk"
    assert result["filter_status"] == "risk"


def test_result_summary_keeps_query_count_separate_from_verdict_totals():
    risk = make_entry(decision={
        "human_review_status": "completed",
        "decision": "risk",
        "opinion": "人工确认付款责任有风险",
        "research_status": "not_required",
    })
    no_risk = make_entry(decision={
        "human_review_status": "completed",
        "decision": "no_obvious_risk",
        "opinion": "人工确认无明显风险",
        "research_status": "not_required",
    })
    query = make_entry(evidence_status="not_found", research={
        "source_types": ["public_query"],
        "query_topics": ["企业登记状态"],
        "missing_identifiers": [],
    })

    summary = summarize_result_items([
        present_result_item(risk),
        present_result_item(no_risk),
        present_result_item(query),
    ])

    assert summary == {
        "total": 3,
        "risk": 1,
        "no_obvious_risk": 1,
        "needs_review": 1,
        "online_query": 1,
    }
