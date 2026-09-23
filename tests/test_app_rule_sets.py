from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://llm.test/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

from app.api import create_app
from app.config import Settings
from app.evidence_review.repository import EvidenceReviewRepository


class RecoveryStub:
    def recover_interrupted_runs(self):
        return None


class ContractServiceStub:
    def __init__(self):
        self.evaluation_service = RecoveryStub()

    def list_contracts(self):
        return []


class ReviewServiceStub(RecoveryStub):
    pass


class RecordingImporter:
    def __init__(self):
        self.rule_set_ids: list[str] = []

    def import_rule_set(self, rule_set_id: str):
        self.rule_set_ids.append(rule_set_id)


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


def build_client(tmp_path: Path):
    settings = settings_for(tmp_path)
    repository = EvidenceReviewRepository(settings.database_path)
    importer = RecordingImporter()
    app = create_app(
        settings=settings,
        service=ContractServiceStub(),
        contract_review_web_service=ReviewServiceStub(),
        rule_set_repository=repository,
        rule_set_import_service=importer,
    )
    return TestClient(app), repository, importer, settings


def draft_payload() -> dict:
    return {
        "schema_version": "1.0",
        "document_title": "项目风险审查规则",
        "sections": [
            {
                "section_id": "s1",
                "source_number": "一、",
                "title": "合同条件",
                "parent_section_id": None,
                "level": 1,
                "source_pages": [1],
            }
        ],
        "review_items": [
            {
                "item_id": "i1",
                "source_number": "1.2",
                "section_path": ["合同条件"],
                "name": "付款条件",
                "rule_text": "付款条件应结合合同约定核验",
                "source_pages": [2],
                "item_kind": "review_check",
                "evidence_scope": "hybrid",
                "decision_mode": "query_and_compare",
                "retrieval_queries": ["合同约定的付款条件是什么"],
                "fact_requirements": [
                    {
                        "fact_key": "counterparty_name",
                        "label": "相对方公司名称",
                        "value_type": "string",
                        "required": True,
                    }
                ],
                "research_requirements": [
                    {
                        "source_type": "public_query",
                        "target_type": "company",
                        "required_fact_keys": ["counterparty_name"],
                        "query_topics": ["企业登记状态"],
                        "comparison_points": ["名称及存续状态"],
                    }
                ],
            }
        ],
        "summary": {
            "section_count": 1,
            "review_item_count": 1,
            "review_check_count": 1,
            "process_control_count": 0,
        },
        "warnings": ["第 3 页图片文字请人工确认"],
    }


def create_record(repository: EvidenceReviewRepository, tmp_path: Path):
    source = tmp_path / "source.md"
    source.write_text("规则", encoding="utf-8")
    return repository.create_rule_set(
        name="项目规则",
        source_filename="项目规则.md",
        source_path=source,
        source_sha256="abc",
    )


def test_upload_schedules_rule_import_and_redirects_to_detail(tmp_path: Path):
    client, repository, importer, settings = build_client(tmp_path)

    response = client.post(
        "/rule-sets",
        data={"name": " 商务审核规则 "},
        files={"file": ("审核规则.md", "一、付款条件".encode(), "text/markdown")},
        follow_redirects=False,
    )

    records = repository.list_rule_sets()
    assert response.status_code == 303
    assert len(records) == 1
    assert records[0].name == "商务审核规则"
    assert records[0].status == "queued"
    assert importer.rule_set_ids == [records[0].rule_set_id]
    assert response.headers["location"] == f"/rule-sets/{records[0].rule_set_id}"
    assert Path(records[0].source_path).is_file()
    assert Path(records[0].source_path).is_relative_to(settings.rule_sets_dir)


def test_invalid_upload_returns_safe_form_error(tmp_path: Path):
    client, repository, importer, _ = build_client(tmp_path)

    response = client.post(
        "/rule-sets",
        data={"name": "规则"},
        files={"file": ("规则.docx", b"invalid", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "仅支持 .pdf、.txt 或 .md" in response.text
    assert repository.list_rule_sets() == []
    assert importer.rule_set_ids == []


def test_draft_detail_shows_only_extracted_rules_and_source_pages(
    tmp_path: Path,
):
    client, repository, _, _ = build_client(tmp_path)
    record = create_record(repository, tmp_path)
    repository.mark_rule_set_processing(record.rule_set_id)
    payload = draft_payload()
    payload["review_items"][0]["name"] = "机器生成的摘要"
    repository.mark_rule_set_draft(record.rule_set_id, payload)

    response = client.get(f"/rule-sets/{record.rule_set_id}")

    assert response.status_code == 200
    assert "共 1 条规则" in response.text
    assert "付款条件" in response.text
    assert "付款条件应结合合同约定核验" in response.text
    assert "第 2 页" in response.text
    assert "确认并启用此版本" in response.text
    assert "机器生成的摘要" not in response.text
    for text in [
        "项目风险审查规则",
        "合同条件",
        "章节 1",
        "合同取证与事实",
        "相对方公司名称",
        "企业登记状态",
        "第 3 页图片文字请人工确认",
    ]:
        assert text not in response.text


def test_draft_can_be_activated_and_api_payload_is_whitelisted(tmp_path: Path):
    client, repository, _, _ = build_client(tmp_path)
    record = create_record(repository, tmp_path)
    repository.mark_rule_set_processing(record.rule_set_id)
    repository.mark_rule_set_draft(record.rule_set_id, draft_payload())

    response = client.post(
        f"/rule-sets/{record.rule_set_id}/activate",
        follow_redirects=False,
    )
    payload = client.get(f"/api/rule-sets/{record.rule_set_id}").json()

    assert response.status_code == 303
    assert repository.get_rule_set(record.rule_set_id).status == "active"
    assert payload["status"] == "active"
    assert payload["parsed_rules"]["review_items"][0]["item_id"] == "i1"
    assert "source_path" not in payload
    assert "source_sha256" not in payload


def test_unknown_rule_set_returns_404_for_page_api_and_activation(tmp_path: Path):
    client, _, _, _ = build_client(tmp_path)

    assert client.get("/rule-sets/missing").status_code == 404
    assert client.get("/api/rule-sets/missing").status_code == 404
    assert client.post("/rule-sets/missing/activate").status_code == 404


def test_failed_rule_set_shows_only_saved_safe_error(tmp_path: Path):
    client, repository, _, _ = build_client(tmp_path)
    record = create_record(repository, tmp_path)
    repository.mark_rule_set_processing(record.rule_set_id)
    failed = repository.mark_rule_set_failed(
        record.rule_set_id,
        "规则文件解析失败，请检查文件内容后重试。",
    )

    page = client.get(f"/rule-sets/{record.rule_set_id}")
    payload = client.get(f"/api/rule-sets/{record.rule_set_id}").json()

    assert page.status_code == 200
    assert failed.error_message in page.text
    assert str(tmp_path) not in page.text
    assert payload["error_message"] == failed.error_message
    assert "source_path" not in payload


def test_dashboard_links_to_rule_set_management(tmp_path: Path):
    client, _, _, _ = build_client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert 'href="/rule-sets"' in response.text
    assert "上传或查看规则" in response.text
    assert "规则文件" in response.text
    assert "合同校验" in response.text


def test_active_rule_set_detail_offers_contract_flow(tmp_path: Path):
    client, repository, _, _ = build_client(tmp_path)
    record = create_record(repository, tmp_path)
    repository.mark_rule_set_processing(record.rule_set_id)
    repository.mark_rule_set_draft(record.rule_set_id, draft_payload())
    repository.activate_rule_set(record.rule_set_id)

    page = client.get(f"/rule-sets/{record.rule_set_id}")

    assert "已启用" in page.text
    assert "前往合同校验" in page.text
    assert 'href="/"' in page.text
    assert "query_and_compare" not in page.text
    assert "public_query" not in page.text
