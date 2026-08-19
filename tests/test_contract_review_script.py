from scripts.test_contract_review import print_progress


ABSENCE_DECISION = {
    "risk_status": "risk",
    "risk_level": "medium",
    "evidence_status": "absence_verified",
    "finding": "基于当前合同全文解析结果，未发现分包转包限制条款。",
    "risk_description": "未发现对应约定。",
    "suggestion": "建议补充明确限制条款。",
}


def test_print_progress_labels_absence_events(capsys):
    print_progress("absence_check_started", {"item_id": "item_2"})
    print_progress(
        "absence_keywords_generated",
        {
            "primary_keywords": ["分包"],
            "secondary_keywords": ["第三方"],
        },
    )
    print_progress(
        "absence_candidates_found",
        {"candidate_count": 0, "candidates": []},
    )
    print_progress(
        "absence_confirmed",
        {"candidate_count": 0, "decision": ABSENCE_DECISION},
    )

    output = capsys.readouterr().out
    assert "=== 开始全文缺失核验 ===" in output
    assert "=== 全文扫描关键词 ===" in output
    assert "=== 全文扫描候选 ===" in output
    assert "=== 缺失核验结果 ===" in output
