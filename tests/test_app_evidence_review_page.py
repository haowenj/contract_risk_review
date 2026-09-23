from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

from app.api import create_app
from app.config import Settings
from app.db import ContractRepository
from app.evidence_review.presentation import summarize_evidence_statuses
from app.evidence_review.repository import EvidenceReviewRepository
from app.evidence_review.schemas import EvidencePackage, ResearchPackage
from app.evidence_review.web_service import EvidenceReviewWebService


class RecoveryStub:
    def recover_interrupted_runs(self):
        return 0


class ActiveContractService:
    def __init__(self, repository):
        self.repository = repository
        self.evaluation_service = RecoveryStub()

    def get_contract(self, contract_id):
        return self.repository.get(contract_id)

    def list_contracts(self):
        return self.repository.list()


class FixedEvidenceService:
    def run(self, contract_id, items, progress_callback=None):
        outputs = [
            EvidencePackage(
                rule_item_id="found",
                evidence_status="found",
                evidence=[
                    {
                        "source_object_index": 12,
                        "page_idx": 4,
                        "node_type": "text",
                        "evidence_text": "乙方：示例工程有限公司",
                    }
                ],
                extracted_facts=[
                    {
                        "fact_key": "counterparty_name",
                        "label": "相对方公司名称",
                        "value": "示例工程有限公司",
                        "unit": None,
                        "evidence_indices": [0],
                    }
                ],
                missing_sources=[],
                research_package=ResearchPackage(
                    required=True,
                    research_status="pending",
                    source_types=["public_query"],
                    reason="需要查询企业公开信息",
                    query_targets=[
                        {
                            "target_type": "company",
                            "fields": {
                                "counterparty_name": "示例工程有限公司"
                            },
                        }
                    ],
                    query_topics=["企业登记状态", "经营异常"],
                    comparison_points=["主体名称及存续状态"],
                    missing_identifiers=["credit_code"],
                ),
                item_error=None,
            ),
            EvidencePackage(
                rule_item_id="not-found",
                evidence_status="not_found",
                evidence=[],
                extracted_facts=[],
                missing_sources=[],
                research_package=None,
                item_error=None,
            ),
            EvidencePackage(
                rule_item_id="source-missing",
                evidence_status="source_missing",
                evidence=[],
                extracted_facts=[],
                missing_sources=["项目内部立项审批记录"],
                research_package=ResearchPackage(
                    required=True,
                    research_status="pending",
                    source_types=["internal_material"],
                    reason="需要补充内部材料",
                    query_targets=[],
                    query_topics=["立项审批状态"],
                    comparison_points=["是否已完成审批"],
                    missing_identifiers=[],
                ),
                item_error=None,
            ),
            EvidencePackage(
                rule_item_id="assessed",
                evidence_status="found",
                evidence=[
                    {
                        "source_object_index": 19,
                        "page_idx": 5,
                        "node_type": "text",
                        "evidence_text": "乙方承担全部延期责任",
                    }
                ],
                extracted_facts=[],
                missing_sources=[],
                research_package=None,
                item_error=None,
            ),
        ]
        for index, item in enumerate(items, start=1):
            if progress_callback:
                progress_callback(
                    "evidence_item_completed",
                    {
                        "completed": index,
                        "total": len(items),
                        "rule_item_id": item.item_id,
                    },
                )
        return outputs

def parsed_rules() -> dict:
    items = []
    for item_id, number, text, scope in [
        ("found", "一-1", "核验相对方企业状态", "hybrid"),
        ("not-found", "一-2", "核验付款保障条款", "contract"),
        ("source-missing", "二-1", "核验内部立项审批", "internal_material"),
        ("assessed", "三-1", "核验延期责任", "contract"),
    ]:
        items.append(
            {
                "item_id": item_id,
                "source_number": number,
                "section_path": ["项目审核"],
                "name": text,
                "rule_text": text,
                "source_pages": [1],
                "item_kind": "review_check",
                "evidence_scope": scope,
                "decision_mode": "expert_review",
                "retrieval_queries": [text],
                "fact_requirements": [],
                "research_requirements": [],
            }
        )
    return {
        "document_title": "项目审核规则",
        "sections": [],
        "review_items": items,
        "summary": {
            "section_count": 0,
            "review_item_count": 4,
            "review_check_count": 4,
            "process_control_count": 0,
        },
    }


def settings_for(root: Path) -> Settings:
    return Settings(
        project_dir=root,
        data_dir=root / "data",
        database_path=root / "data" / "contracts.db",
        contracts_dir=root / "data" / "contracts",
        rule_sets_dir=root / "data" / "rule_sets",
        mineru_url="http://mineru.test",
        mineru_backend="hybrid-engine",
        mineru_server_url=None,
    )


def build_client(tmp_path: Path, *, contract_status="ready"):
    settings = settings_for(tmp_path)
    contracts = ContractRepository(settings.database_path)
    contract = contracts.create(
        "contract.pdf", tmp_path / "contract", contract_id="c1"
    )
    contracts.update_status(
        contract.contract_id,
        contract_status,
        index_version="v1" if contract_status == "ready" else None,
    )
    contract_service = ActiveContractService(contracts)
    repository = EvidenceReviewRepository(settings.database_path)
    source = tmp_path / "rules.md"
    source.write_text("规则", encoding="utf-8")
    rule_set = repository.create_rule_set(
        name="项目审核规则",
        source_filename="rules.md",
        source_path=source,
        source_sha256="abc",
    )
    repository.mark_rule_set_processing(rule_set.rule_set_id)
    repository.mark_rule_set_draft(rule_set.rule_set_id, parsed_rules())
    rule_set = repository.activate_rule_set(rule_set.rule_set_id)
    evidence_web = EvidenceReviewWebService(
        contract_service=contract_service,
        repository=repository,
        evidence_service_factory=lambda **kwargs: FixedEvidenceService(),
    )
    application = create_app(
        settings=settings,
        service=contract_service,
        contract_review_web_service=RecoveryStub(),
        rule_set_repository=repository,
        evidence_review_web_service=evidence_web,
    )
    return TestClient(application), repository, rule_set


def test_page_lists_active_rule_sets_and_dashboard_has_one_review_entry(
    tmp_path: Path,
):
    client, _, rule_set = build_client(tmp_path)

    page = client.get("/contracts/c1/evidence-review")
    dashboard = client.get("/")

    assert page.status_code == 200
    assert rule_set.name in page.text
    assert f'value="{rule_set.rule_set_id}"' in page.text
    assert 'href="/contracts/c1/evidence-review"' in dashboard.text
    assert "选择规则并校验" in dashboard.text
    assert 'href="/contracts/c1/review"' not in dashboard.text
    assert 'href="/contracts/c1/evaluation"' not in dashboard.text
    assert "实验性自动判断" not in dashboard.text
    assert "召回测试" not in dashboard.text
    assert "reviewLink" not in dashboard.text
    assert "evaluationLink" not in dashboard.text


def test_run_creation_executes_in_background_and_redirects(tmp_path: Path):
    client, repository, rule_set = build_client(tmp_path)

    response = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(response.headers["location"]).query)["run_id"][0]

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/contracts/c1/evidence-review?run_id={run_id}"
    )
    assert repository.get_evidence_run(run_id).status == "ready"


def test_contract_rule_selection_page_links_to_previous_result(tmp_path: Path):
    client, _, rule_set = build_client(tmp_path)
    created = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(created.headers["location"]).query)["run_id"][0]

    page = client.get("/contracts/c1/evidence-review")

    assert page.status_code == 200
    assert "历史校验结果" in page.text
    assert rule_set.name in page.text
    assert f'/contracts/c1/evidence-review?run_id={run_id}' in page.text


def test_ready_page_displays_evidence_facts_missing_sources_and_research(
    tmp_path: Path,
):
    client, _, rule_set = build_client(tmp_path)
    created = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(created.headers["location"]).query)["run_id"][0]

    response = client.get(f"/contracts/c1/evidence-review?run_id={run_id}")

    assert response.status_code == 200
    for text in [
        "一-1",
        "核验相对方企业状态",
        "乙方：示例工程有限公司",
        "第 5 页",
        "相对方公司名称",
        "示例工程有限公司",
        "企业登记状态",
        "主体名称及存续状态",
        "查询前还缺",
        "统一社会信用代码",
        "项目内部立项审批记录",
        "这不代表合同没有约定，请人工查阅全文和附件",
    ]:
        assert text in response.text
    for forbidden in ["risk_status", "风险说明", "修改建议"]:
        assert forbidden not in response.text
    assert 'id="result-filter"' in response.text


def test_result_page_leads_with_contract_situation_and_next_queries(tmp_path: Path):
    client, _, rule_set = build_client(tmp_path)
    created = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(created.headers["location"]).query)["run_id"][0]

    page = client.get(f"/contracts/c1/evidence-review?run_id={run_id}")

    assert page.status_code == 200
    assert "合同检查结果" in page.text
    assert "待联网查询" in page.text
    assert "待补资料" in page.text
    assert "合同情况" in page.text
    assert "需要联网查询" in page.text
    assert "企业登记状态" in page.text
    assert "规则检查项" not in page.text
    assert "取证状态汇总" not in page.text
    assert "待人工判断" not in page.text
    assert 'id="result-filter"' in page.text
    assert 'id="research-status-filter"' not in page.text
    assert '<details class="source-details"' in page.text
    assert '<details class="decision-details"' in page.text
    assert "统一社会信用代码" in page.text
    assert "credit_code" not in page.text
    assert "来源对象 12" not in page.text


def test_existing_result_has_no_second_model_judgment_action(
    tmp_path: Path,
):
    client, _, rule_set = build_client(tmp_path)
    created = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(created.headers["location"]).query)["run_id"][0]
    result_url = f"/contracts/c1/evidence-review?run_id={run_id}"

    before = client.get(result_url)
    assert "为已有结果生成系统建议" not in before.text

    submitted = client.post(
        f"/contracts/c1/evidence-review/runs/{run_id}/suggestions",
        follow_redirects=False,
    )
    after = client.get(result_url)

    assert submitted.status_code == 404
    assert "疑似有风险" not in after.text
    assert "乙方承担全部延期责任" in after.text
    assert "系统建议" not in after.text
    assert "为已有结果生成系统建议" not in after.text
    payload = client.get(f"/api/contracts/c1/evidence-review/runs/{run_id}").json()
    assert payload["items"][3]["evidence_package"]["system_suggestion"] is None
    assert payload["items"][3]["human_decision"]["human_review_status"] == "pending"


def test_status_summary_includes_failed_items():
    entries = [
        {"evidence_package": {"evidence_status": status}}
        for status in ["found", "not_found", "source_missing", "extraction_failed"]
    ]

    summary = summarize_evidence_statuses(entries)

    assert summary == {
        "found": 1,
        "not_found": 1,
        "source_missing": 1,
        "extraction_failed": 1,
    }


def test_status_api_is_pollable_and_cross_contract_is_hidden(tmp_path: Path):
    client, _, rule_set = build_client(tmp_path)
    created = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(created.headers["location"]).query)["run_id"][0]

    response = client.get(f"/api/contracts/c1/evidence-review/runs/{run_id}")
    hidden = client.get(f"/api/contracts/other/evidence-review/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["progress"]["completed"] == 4
    assert response.json()["items"][0]["evidence_package"]["evidence"][0] == {
        "source_object_index": 12,
        "page_idx": 4,
        "node_type": "text",
        "evidence_text": "乙方：示例工程有限公司",
    }
    assert hidden.status_code == 404


def test_non_ready_contract_cannot_create_evidence_run(tmp_path: Path):
    client, repository, rule_set = build_client(
        tmp_path, contract_status="processing"
    )

    response = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
    )

    assert response.status_code == 409
    assert "完成入库后才能进行人工取证" in response.text
    assert repository.get_evidence_run("missing") is None


def test_processing_page_contains_polling_endpoint(tmp_path: Path):
    client, repository, rule_set = build_client(tmp_path)
    run = repository.create_evidence_run(
        contract_id="c1",
        rule_set_id=rule_set.rule_set_id,
        rule_snapshot=rule_set.parsed_rules,
    )
    repository.mark_evidence_run_processing(run.run_id)

    response = client.get(
        f"/contracts/c1/evidence-review?run_id={run.run_id}"
    )

    assert response.status_code == 200
    assert f"/api/contracts/c1/evidence-review/runs/{run.run_id}" in response.text
    assert "正在提取合同证据" in response.text


def test_decision_form_saves_and_redirects_to_item_anchor(tmp_path: Path):
    client, _, rule_set = build_client(tmp_path)
    created = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(created.headers["location"]).query)["run_id"][0]
    payload = client.get(
        f"/api/contracts/c1/evidence-review/runs/{run_id}"
    ).json()
    decision = payload["items"][0]["human_decision"]

    response = client.post(
        f"/contracts/c1/evidence-review/runs/{run_id}/items/found/decision",
        data={
            "decision": "cannot_determine",
            "risk_level": "",
            "opinion": "等待企业查询结果后再判断",
            "research_status": "pending",
            "research_notes": "尚未完成公开查询",
            "expected_updated_at": decision["updated_at"],
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/contracts/c1/evidence-review?run_id={run_id}#item-found"
    )
    saved = client.get(
        f"/api/contracts/c1/evidence-review/runs/{run_id}"
    ).json()["items"][0]["human_decision"]
    assert saved["decision"] == "cannot_determine"
    assert saved["opinion"] == "等待企业查询结果后再判断"


def test_invalid_decision_preserves_input_and_stale_route_returns_409(
    tmp_path: Path,
):
    client, _, rule_set = build_client(tmp_path)
    created = client.post(
        "/contracts/c1/evidence-review/runs",
        data={"rule_set_id": rule_set.rule_set_id},
        follow_redirects=False,
    )
    run_id = parse_qs(urlparse(created.headers["location"]).query)["run_id"][0]
    endpoint = (
        f"/contracts/c1/evidence-review/runs/{run_id}"
        "/items/found/decision"
    )
    initial = client.get(
        f"/api/contracts/c1/evidence-review/runs/{run_id}"
    ).json()["items"][0]["human_decision"]

    invalid = client.post(
        endpoint,
        data={
            "decision": "risk",
            "risk_level": "high",
            "opinion": "这段输入必须保留",
            "research_status": "pending",
            "research_notes": "尚未查询",
            "expected_updated_at": initial["updated_at"],
        },
    )
    assert invalid.status_code == 400
    assert "这段输入必须保留" in invalid.text
    assert "外部或内部查询尚未完成" in invalid.text
    assert '<details class="decision-details" open>' in invalid.text

    valid_data = {
        "decision": "cannot_determine",
        "risk_level": "",
        "opinion": "暂无法判断",
        "research_status": "pending",
        "research_notes": "等待查询",
        "expected_updated_at": initial["updated_at"],
    }
    assert client.post(endpoint, data=valid_data).status_code == 200
    stale = client.post(endpoint, data=valid_data)
    assert stale.status_code == 409
    assert "页面数据已更新" in stale.text
