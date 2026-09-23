from __future__ import annotations

import json
from types import SimpleNamespace

from app.evidence_review.schemas import RuleItem
from app.evidence_review.service import EvidenceReviewService
from app.evidence_review.web_service import build_evidence_review_service


def rule_item(
    *,
    item_id: str = "company-check",
    item_kind: str = "review_check",
    evidence_scope: str = "hybrid",
    retrieval_queries: list[str] | None = None,
    fact_requirements: list[dict] | None = None,
    research_requirements: list[dict] | None = None,
) -> RuleItem:
    return RuleItem.model_validate(
        {
            "item_id": item_id,
            "source_number": "1.1",
            "section_path": ["主体资格"],
            "name": "相对方主体信息",
            "rule_text": "结合合同和公开信息核验相对方主体状态",
            "source_pages": [1],
            "item_kind": item_kind,
            "evidence_scope": evidence_scope,
            "decision_mode": "query_and_compare",
            "retrieval_queries": retrieval_queries or [
                "合同相对方的名称和统一社会信用代码"
            ],
            "fact_requirements": fact_requirements
            if fact_requirements is not None
            else [
                {
                    "fact_key": "counterparty_name",
                    "label": "相对方公司名称",
                    "value_type": "string",
                    "required": True,
                },
                {
                    "fact_key": "credit_code",
                    "label": "统一社会信用代码",
                    "value_type": "string",
                    "required": True,
                },
            ],
            "research_requirements": research_requirements
            if research_requirements is not None
            else [
                {
                    "source_type": "public_query",
                    "target_type": "company",
                    "required_fact_keys": [
                        "counterparty_name",
                        "credit_code",
                    ],
                    "query_topics": ["企业登记状态", "经营异常"],
                    "comparison_points": ["主体名称及存续状态"],
                }
            ],
        }
    )


class FakeContractService:
    def __init__(self, results=None, failures=None):
        self.results = results or {}
        self.failures = failures or {}
        self.searches: list[tuple[str, str]] = []

    def search_contract(self, contract_id: str, query: str):
        self.searches.append((contract_id, query))
        if query in self.failures:
            raise self.failures[query]
        return self.results.get(query, [])


class FakeLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return SimpleNamespace(
            content=payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )


def evidence(index: int, text: str, *, page_idx: int = 0) -> dict:
    return {
        "source_object_index": index,
        "page_idx": page_idx,
        "node_type": "text",
        "evidence_text": text,
        "score": 0.98,
        "internal_debug": "must not escape",
    }


def test_page_review_service_constructs_only_the_existing_fact_model(monkeypatch):
    from app.evidence_review import web_service

    calls = []
    monkeypatch.setattr(
        web_service,
        "build_fact_extraction_llm",
        lambda: calls.append("fact") or FakeLLM([]),
    )
    monkeypatch.setattr(
        web_service,
        "build_risk_suggestion_llm",
        lambda: calls.append("second_review") or FakeLLM([]),
        raising=False,
    )

    build_evidence_review_service(contract_service=FakeContractService())

    assert calls == ["fact"]


def test_hybrid_item_returns_evidence_facts_and_pending_research_without_risk():
    query = "合同相对方的名称和统一社会信用代码"
    contract = FakeContractService(
        {
            query: [
                evidence(7, "乙方：示例工程有限公司"),
                evidence(7, "重复片段不应进入结果"),
                evidence(9, "统一社会信用代码：91310000TEST000001"),
            ]
        }
    )
    llm = FakeLLM(
        [
            {
                "extracted_facts": [
                    {
                        "fact_key": "counterparty_name",
                        "label": "相对方公司名称",
                        "value": "示例工程有限公司",
                        "unit": None,
                        "evidence_indices": [0],
                    },
                    {
                        "fact_key": "credit_code",
                        "label": "统一社会信用代码",
                        "value": "91310000TEST000001",
                        "unit": None,
                        "evidence_indices": [1],
                    },
                ],
                "missing_sources": [],
            }
        ]
    )
    service = EvidenceReviewService(contract_service=contract, fact_llm=llm)

    payload = service.extract_item("c1", rule_item()).model_dump(mode="json")

    assert payload["evidence_status"] == "found"
    assert [value["source_object_index"] for value in payload["evidence"]] == [7, 9]
    assert payload["research_package"]["research_status"] == "pending"
    assert payload["research_package"]["query_targets"] == [
        {
            "target_type": "company",
            "fields": {
                "counterparty_name": "示例工程有限公司",
                "credit_code": "91310000TEST000001",
            },
        }
    ]
    assert "risk_status" not in payload
    assert "risk_level" not in payload
    assert "internal_debug" not in json.dumps(payload, ensure_ascii=False)


def test_repeated_fact_with_same_value_merges_evidence_without_retry():
    query = "付款期限"
    contract = FakeContractService({query: [
        evidence(1, "付款期限为30天", page_idx=0),
        evidence(2, "乙方应在30天内付款", page_idx=2),
    ]})
    llm = FakeLLM([{"extracted_facts": [
        {"fact_key": "payment_term", "label": "付款期限", "value": "30", "unit": "天", "evidence_indices": [0]},
        {"fact_key": "payment_term", "label": "付款期限", "value": "30", "unit": "天", "evidence_indices": [1, 0]},
    ], "missing_sources": []}])
    item = rule_item(
        evidence_scope="contract",
        retrieval_queries=[query],
        fact_requirements=[{"fact_key": "payment_term", "label": "付款期限", "value_type": "integer", "required": True}],
        research_requirements=[],
    )

    package = EvidenceReviewService(contract_service=contract, fact_llm=llm).extract_item("c1", item)

    assert package.evidence_status == "found"
    assert [(fact.value, fact.evidence_indices) for fact in package.extracted_facts] == [("30", [0, 1])]
    assert len(llm.prompts) == 1


def test_repeated_fact_with_different_values_preserves_both_and_research_targets():
    query = "付款期限"
    contract = FakeContractService({query: [
        evidence(1, "付款期限为30天", page_idx=0),
        evidence(2, "付款期限为60天", page_idx=2),
    ]})
    llm = FakeLLM([{"extracted_facts": [
        {"fact_key": "payment_term", "label": "模型标签", "value": "30", "unit": "天", "evidence_indices": [0]},
        {"fact_key": "payment_term", "label": "模型标签", "value": "60", "unit": "天", "evidence_indices": [1]},
    ], "missing_sources": []}])
    item = rule_item(
        retrieval_queries=[query],
        fact_requirements=[{"fact_key": "payment_term", "label": "付款期限", "value_type": "integer", "required": True}],
        research_requirements=[{
            "source_type": "public_query", "target_type": "payment", "required_fact_keys": ["payment_term"],
            "query_topics": ["付款期限"], "comparison_points": ["付款期限"],
        }],
    )

    package = EvidenceReviewService(contract_service=contract, fact_llm=llm).extract_item("c1", item)

    assert package.evidence_status == "found"
    assert [(fact.label, fact.value, fact.evidence_indices) for fact in package.extracted_facts] == [
        ("付款期限", "30", [0]), ("付款期限", "60", [1]),
    ]
    assert package.research_package.query_targets[0].fields["payment_term"] == "30 天；60 天"
    assert len(llm.prompts) == 1


def test_contract_item_with_no_results_is_not_found_without_fact_call():
    contract = FakeContractService()
    llm = FakeLLM([])
    item = rule_item(evidence_scope="contract", research_requirements=[])

    package = EvidenceReviewService(
        contract_service=contract,
        fact_llm=llm,
    ).extract_item("c1", item)

    assert package.evidence_status == "not_found"
    assert package.evidence == []
    assert package.research_package is None
    assert llm.prompts == []


def test_internal_material_item_is_source_missing_without_contract_search():
    contract = FakeContractService()
    item = rule_item(
        evidence_scope="internal_material",
        research_requirements=[
            {
                "source_type": "internal_material",
                "target_type": "approval_record",
                "required_fact_keys": [],
                "query_topics": ["内部审批记录"],
                "comparison_points": ["是否已完成审批"],
            }
        ],
    )

    package = EvidenceReviewService(
        contract_service=contract,
        fact_llm=FakeLLM([]),
    ).extract_item("c1", item)

    assert package.evidence_status == "source_missing"
    assert package.missing_sources == ["规则要求的内部材料尚未提供"]
    assert package.research_package.research_status == "pending"
    assert contract.searches == []


def test_external_only_item_builds_research_without_evidence_or_search():
    contract = FakeContractService()
    item = rule_item(evidence_scope="external_query")

    package = EvidenceReviewService(
        contract_service=contract,
        fact_llm=FakeLLM([]),
    ).extract_item("c1", item)

    assert package.evidence_status == "not_found"
    assert package.evidence == []
    assert package.research_package.required is True
    assert package.research_package.missing_identifiers == [
        "counterparty_name",
        "credit_code",
    ]
    assert contract.searches == []


def test_process_control_creates_human_todo_without_search():
    contract = FakeContractService()
    item = rule_item(item_kind="process_control", evidence_scope="contract")

    package = EvidenceReviewService(
        contract_service=contract,
        fact_llm=FakeLLM([]),
    ).extract_item("c1", item)

    assert package.evidence_status == "source_missing"
    assert "流程控制项需要人工确认" in package.missing_sources[0]
    assert contract.searches == []


def test_missing_required_identifiers_are_listed_for_manual_query():
    query = "合同相对方的名称和统一社会信用代码"
    contract = FakeContractService({query: [evidence(1, "合同未填写乙方名称")]})
    llm = FakeLLM([{"extracted_facts": [], "missing_sources": []}])

    package = EvidenceReviewService(
        contract_service=contract,
        fact_llm=llm,
    ).extract_item("c1", rule_item())

    assert package.research_package.missing_identifiers == [
        "counterparty_name",
        "credit_code",
    ]
    assert [
        value.model_dump(mode="json")
        for value in package.research_package.query_targets
    ] == [
        {"target_type": "company", "fields": {}}
    ]


def test_sensitive_fields_are_excluded_from_research_targets():
    requirements = [
        {
            "fact_key": key,
            "label": label,
            "value_type": "string",
            "required": True,
        }
        for key, label in [
            ("counterparty_name", "公司名称"),
            ("bank_account", "银行账号"),
            ("phone_number", "联系电话"),
            ("identity_number", "身份证号"),
            ("personal_address", "个人住址"),
        ]
    ]
    research = [
        {
            "source_type": "public_query",
            "target_type": "company",
            "required_fact_keys": [value["fact_key"] for value in requirements],
            "query_topics": ["主体状态"],
            "comparison_points": ["主体状态"],
        }
    ]
    query = "合同相对方的名称和统一社会信用代码"
    contract = FakeContractService({query: [evidence(1, "主体及联系人信息")]})
    facts = [
        {
            "fact_key": requirement["fact_key"],
            "label": requirement["label"],
            "value": value,
            "unit": None,
            "evidence_indices": [0],
        }
        for requirement, value in zip(
            requirements,
            ["示例公司", "622200001234", "13800138000", "110101199001011234", "某市某街1号"],
            strict=True,
        )
    ]
    item = rule_item(
        fact_requirements=requirements,
        research_requirements=research,
    )

    package = EvidenceReviewService(
        contract_service=contract,
        fact_llm=FakeLLM([{"extracted_facts": facts, "missing_sources": []}]),
    ).extract_item("c1", item)
    research_json = json.dumps(
        package.research_package.model_dump(mode="json"),
        ensure_ascii=False,
    )

    assert package.research_package.query_targets[0].fields == {
        "counterparty_name": "示例公司"
    }
    for sensitive in [
        "622200001234",
        "13800138000",
        "110101199001011234",
        "某市某街1号",
        "bank_account",
        "phone_number",
        "identity_number",
        "personal_address",
    ]:
        assert sensitive not in research_json


def test_run_isolates_item_failure_and_continues_with_progress():
    bad_query = "失败查询"
    good_query = "成功查询"
    contract = FakeContractService(
        results={good_query: [evidence(2, "付款期限为30日")]},
        failures={bad_query: RuntimeError("secret backend path")},
    )
    llm = FakeLLM([{"extracted_facts": [], "missing_sources": []}])
    service = EvidenceReviewService(contract_service=contract, fact_llm=llm)
    events = []

    packages = service.run(
        "c1",
        [
            rule_item(
                item_id="bad",
                evidence_scope="contract",
                retrieval_queries=[bad_query],
                fact_requirements=[],
                research_requirements=[],
            ),
            rule_item(
                item_id="good",
                evidence_scope="contract",
                retrieval_queries=[good_query],
                fact_requirements=[],
                research_requirements=[],
            ),
        ],
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert packages[0].evidence_status == "extraction_failed"
    assert packages[0].item_error == "该审查项取证失败，请人工处理。"
    assert "secret" not in packages[0].item_error
    assert packages[1].evidence_status == "found"
    assert events[-1] == (
        "evidence_item_completed",
        {"completed": 2, "total": 2, "rule_item_id": "good"},
    )
