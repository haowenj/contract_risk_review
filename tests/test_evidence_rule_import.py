from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from app.evidence_review import rule_import
from app.evidence_review.repository import EvidenceReviewRepository
from app.evidence_review.rule_import import (
    ExtractedRuleDocument,
    RuleDocumentExtractor,
    RuleSetImportService,
    build_rule_parse_llm,
    validate_rule_upload,
)


def make_item(
    *,
    item_id: str = "item-1",
    source_number: str | None = "1",
    name: str = "付款安排",
    item_kind: str = "review_check",
    section_path: list[str] | None = None,
) -> dict:
    return {
        "item_id": item_id,
        "source_number": source_number,
        "section_path": section_path or ["商务条件"],
        "name": name,
        "rule_text": f"{name}的原始规则文本",
        "source_pages": [1],
        "item_kind": item_kind,
        "evidence_scope": "contract",
        "decision_mode": "expert_review",
        "retrieval_queries": [f"合同中的{name}"],
        "fact_requirements": [],
        "research_requirements": [],
    }


def make_result(
    *,
    source_number: str | None = "1",
    item_count: int = 1,
) -> dict:
    return {
        "document_title": "动态规则",
        "sections": [
            {
                "section_id": "section-1",
                "source_number": None,
                "title": "商务条件",
                "parent_section_id": None,
                "level": 1,
                "source_pages": [1],
            }
        ],
        "review_items": [
            make_item(
                item_id=f"item-{index + 1}",
                source_number=source_number,
                name=f"检查项{index + 1}",
            )
            for index in range(item_count)
        ],
    }


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        content = (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )
        return SimpleNamespace(content=content)


def build_import_service(
    tmp_path: Path,
    payload,
) -> tuple[EvidenceReviewRepository, RuleSetImportService, str]:
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    source_path = tmp_path / "incoming.md"
    source_path.write_text("任意层级和编号的规则内容", encoding="utf-8")
    record = repository.create_rule_set(
        name="任意规则",
        source_filename="规则.md",
        source_path=source_path,
        source_sha256="will-be-revalidated",
    )
    service = RuleSetImportService(
        repository=repository,
        extractor=RuleDocumentExtractor(
            svr_url="http://mineru",
            backend="hybrid-engine",
            server_url=None,
        ),
        parse_llm=FakeLLM(payload),
        rule_sets_dir=tmp_path / "managed",
        max_upload_bytes=1024 * 1024,
    )
    return repository, service, record.rule_set_id


@pytest.mark.parametrize(
    ("source_number", "item_count"),
    [
        ("一、", 1),
        ("1.2", 2),
        ("（三）", 3),
        ("①", 1),
        ("•", 2),
        (None, 1),
    ],
)
def test_import_preserves_original_labels_and_dynamic_item_counts(
    tmp_path: Path,
    source_number: str | None,
    item_count: int,
):
    repository, service, rule_set_id = build_import_service(
        tmp_path,
        make_result(source_number=source_number, item_count=item_count),
    )

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    assert imported.parsed_rules["summary"]["review_item_count"] == item_count
    assert [
        item["source_number"] for item in imported.parsed_rules["review_items"]
    ] == [source_number] * item_count
    assert repository.get_rule_set(rule_set_id) == imported


def test_import_supports_three_section_levels_and_keeps_parent_nodes_out_of_items(
    tmp_path: Path,
):
    payload = make_result()
    payload["sections"] = [
        {
            "section_id": "s1",
            "source_number": "一、",
            "title": "投标阶段",
            "parent_section_id": None,
            "level": 1,
            "source_pages": [1],
        },
        {
            "section_id": "s2",
            "source_number": "1.",
            "title": "商务条件",
            "parent_section_id": "s1",
            "level": 2,
            "source_pages": [1],
        },
        {
            "section_id": "s3",
            "source_number": "（1）",
            "title": "付款",
            "parent_section_id": "s2",
            "level": 3,
            "source_pages": [1],
        },
    ]
    payload["review_items"] = [
        make_item(section_path=["投标阶段", "商务条件", "付款"]),
        make_item(
            item_id="process-1",
            source_number="备注",
            name="履行审批程序",
            item_kind="process_control",
            section_path=["投标阶段", "商务条件"],
        ),
    ]
    _, service, rule_set_id = build_import_service(tmp_path, payload)

    imported = service.import_rule_set(rule_set_id)

    assert len(imported.parsed_rules["sections"]) == 3
    assert len(imported.parsed_rules["review_items"]) == 2
    assert imported.parsed_rules["summary"] == {
        "section_count": 3,
        "review_item_count": 2,
        "review_check_count": 1,
        "process_control_count": 1,
    }


def test_import_normalizes_nested_section_items_from_schema_weak_models(
    tmp_path: Path,
):
    nested_item = make_item(
        item_id="stage-one-item-5-3",
        source_number="5.3",
        name="主体信息核验",
        section_path=["投标立项审查"],
    )
    payload = {
        "document_title": "动态负面清单",
        "sections": [
            {
                "section_id": "stage-one",
                "source_number": "一、",
                "section_title": "投标立项审查",
                "parent_section_id": None,
                "review_items": [nested_item],
            }
        ],
    }
    _, service, rule_set_id = build_import_service(tmp_path, payload)

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    assert imported.parsed_rules["sections"] == [
        {
            "section_id": "stage-one",
            "source_number": "一、",
            "title": "投标立项审查",
            "parent_section_id": None,
            "level": 1,
            "source_pages": [1],
        }
    ]
    assert len(imported.parsed_rules["review_items"]) == 1
    assert imported.parsed_rules["review_items"][0]["source_number"] == (
        "一-5.3"
    )


def test_import_expands_schema_weak_item_shorthand_before_strict_validation(
    tmp_path: Path,
):
    payload = {
        "document_title": "动态负面清单",
        "sections": [
            {
                "section_id": "stage-one",
                "source_number": "一、",
                "title": "投标立项审查",
                "parent_section_id": None,
                "level": 1,
                "source_pages": [1],
            }
        ],
        "review_items": [
            {
                "item_id": "stage-one-item-5-3",
                "source_number": "一-5.3",
                "rule_text": "核验交易主体的公开信用信息",
                "source_pages": [1],
                "risk_type": "review_check",
                "evidence_scope": "hybrid",
                "retrieval_queries": ["合同相对方名称和统一社会信用代码"],
                "fact_requirements": [
                    "合同相对方名称",
                    "统一社会信用代码",
                ],
                "research_requirements": [
                    "查询交易主体的登记及公开信用信息"
                ],
            }
        ],
    }
    _, service, rule_set_id = build_import_service(tmp_path, payload)

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    item = imported.parsed_rules["review_items"][0]
    assert item["section_path"] == ["投标立项审查"]
    assert item["name"] == "核验交易主体的公开信用信息"
    assert item["item_kind"] == "review_check"
    assert item["decision_mode"] == "query_and_compare"
    assert item["fact_requirements"] == [
        {
            "fact_key": "fact_1_合同相对方名称",
            "label": "合同相对方名称",
            "value_type": "string",
            "required": True,
        },
        {
            "fact_key": "fact_2_统一社会信用代码",
            "label": "统一社会信用代码",
            "value_type": "string",
            "required": True,
        },
    ]
    assert item["research_requirements"] == [
        {
            "source_type": "public_query",
            "target_type": "外部查询对象",
            "required_fact_keys": [
                "fact_1_合同相对方名称",
                "fact_2_统一社会信用代码",
            ],
            "query_topics": ["查询交易主体的登记及公开信用信息"],
            "comparison_points": ["核验交易主体的公开信用信息"],
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        {"document_title": "空", "sections": [], "review_items": []},
        {
            "document_title": "重复编号",
            "sections": [],
            "review_items": [
                make_item(item_id="same"),
                make_item(item_id="same", name="另一项"),
            ],
        },
    ],
)
def test_invalid_model_output_is_saved_as_safe_failure(tmp_path: Path, payload):
    _, service, rule_set_id = build_import_service(tmp_path, payload)

    failed = service.import_rule_set(rule_set_id)

    assert failed.status == "failed"
    assert failed.error_message == "规则文件解析失败，请检查文件内容后重试。"
    assert "not json" not in failed.error_message
    assert failed.parsed_rules == {}


def test_extractor_failure_is_saved_as_safe_failure(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    missing_path = tmp_path / "missing.pdf"
    record = repository.create_rule_set(
        name="损坏规则",
        source_filename="missing.pdf",
        source_path=missing_path,
        source_sha256="unknown",
    )
    service = RuleSetImportService(
        repository=repository,
        extractor=mock.Mock(),
        parse_llm=FakeLLM(make_result()),
        rule_sets_dir=tmp_path / "managed",
        max_upload_bytes=1024,
    )

    failed = service.import_rule_set(record.rule_set_id)

    assert failed.status == "failed"
    assert failed.error_message == "规则文件解析失败，请检查文件内容后重试。"


def test_pdf_rules_use_mineru_content_list_text_blocks(tmp_path: Path):
    def fake_mineru(source_path, raw_path, **_kwargs):
        assert source_path.read_bytes() == b"%PDF-test"
        raw_path.write_text(json.dumps([
            {"type": "text", "text": "一、商务条件", "text_level": 1, "page_idx": 0},
            {"type": "page_number", "text": "1", "page_idx": 0},
            {"type": "table", "text": "表格内容", "page_idx": 0},
            {"type": "text", "text": "1. 核验付款条件", "page_idx": 1},
        ], ensure_ascii=False), encoding="utf-8")

    extractor = RuleDocumentExtractor(
        svr_url="http://mineru",
        backend="hybrid-engine",
        server_url=None,
        parser=fake_mineru,
    )
    upload = validate_rule_upload("规则.pdf", b"%PDF-test", max_bytes=1024)

    document = extractor.extract(upload, tmp_path / "managed")

    assert document.blocks == [
        {"page_number": 1, "text": "一、商务条件", "text_level": 1},
        {"page_number": 2, "text": "1. 核验付款条件"},
    ]
    assert document.page_count == 2


def test_build_llm_uses_strict_rule_parse_json_schema():
    bound = mock.Mock()
    with mock.patch.object(rule_import, "ChatOpenAI") as factory:
        factory.return_value.bind.return_value = bound
        result = build_rule_parse_llm()

    assert result is bound
    response_format = factory.return_value.bind.call_args.kwargs[
        "response_format"
    ]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["title"] == "RuleParseChunkResult"
    assert factory.call_args.kwargs["temperature"] == 0
    assert factory.call_args.kwargs["timeout"] == 300.0
    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["reasoning_effort"] == "none"
    assert factory.call_args.kwargs["max_tokens"] == 8192


def test_import_splits_mineru_text_blocks_and_merges_local_ids(tmp_path: Path):
    class BatchLLM:
        def __init__(self):
            self.prompts: list[str] = []

        def invoke(self, prompt: str):
            self.prompts.append(prompt)
            primary = json.loads(
                prompt.split("<rule_document>\n", 1)[1].split(
                    "\n</rule_document>", 1
                )[0]
            )["blocks"]
            is_last = any("唯一末项" in block["text"] for block in primary)
            page = 13 if is_last else 1
            payload = make_result()
            payload["sections"][0]["source_pages"] = [page]
            payload["review_items"][0]["source_pages"] = [page]
            payload["review_items"][0]["rule_text"] = (
                "唯一末项" if is_last else "唯一首项"
            )
            if is_last:
                duplicate = make_item(item_id="overlap", name="重复上下文项")
                duplicate["rule_text"] = "唯一首项"
                payload["review_items"].insert(0, duplicate)
            return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))

    _, service, rule_set_id = build_import_service(tmp_path, make_result())
    blocks = [
        {"page_number": index, "text": f"第{index}页：规则说明"}
        for index in range(1, 13)
    ] + [{"page_number": 13, "text": "唯一末项"}]
    service.extractor = mock.Mock(
        extract=mock.Mock(
            return_value=ExtractedRuleDocument("动态规则", blocks, 13)
        )
    )
    llm = BatchLLM()
    service.parse_llm = llm

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    assert len(llm.prompts) >= 2
    assert [item["rule_text"] for item in imported.parsed_rules["review_items"]] == [
        "唯一首项",
        "唯一末项",
    ]
    assert len({item["item_id"] for item in imported.parsed_rules["review_items"]}) == 2
    assert imported.parsed_rules["sections"][0]["source_pages"] == [1, 13]


def test_import_fails_if_one_batch_is_truncated(tmp_path: Path):
    class TruncatingLLM:
        def __init__(self):
            self.truncated = False

        def invoke(self, prompt: str):
            primary = json.loads(
                prompt.split("<rule_document>\n", 1)[1].split(
                    "\n</rule_document>", 1
                )[0]
            )["blocks"]
            if any("末项" in block["text"] for block in primary):
                self.truncated = True
                return SimpleNamespace(
                    content='{"document_title":',
                    response_metadata={"finish_reason": "length"},
                )
            return SimpleNamespace(
                content=json.dumps(make_result(), ensure_ascii=False),
                response_metadata={"finish_reason": "stop"},
            )

    _, service, rule_set_id = build_import_service(tmp_path, make_result())
    service.extractor = mock.Mock(
        extract=mock.Mock(
            return_value=ExtractedRuleDocument(
                "动态规则",
                [{"page_number": i, "text": f"第{i}项"} for i in range(1, 13)]
                + [{"page_number": 13, "text": "末项"}],
                13,
            )
        )
    )
    llm = TruncatingLLM()
    service.parse_llm = llm

    failed = service.import_rule_set(rule_set_id)

    assert llm.truncated
    assert failed.status == "failed"
    assert failed.parsed_rules == {}


def test_import_merges_child_section_even_when_model_lists_it_before_parent(
    tmp_path: Path,
):
    class SectionLLM:
        def invoke(self, prompt: str):
            primary = json.loads(
                prompt.split("<rule_document>\n", 1)[1].split(
                    "\n</rule_document>", 1
                )[0]
            )["blocks"]
            payload = make_result()
            if any("末项" in block["text"] for block in primary):
                payload["sections"] = [
                    {
                        "section_id": "child",
                        "source_number": "1",
                        "title": "付款",
                        "parent_section_id": "parent",
                        "level": 2,
                        "source_pages": [13],
                    },
                    {
                        "section_id": "parent",
                        "source_number": None,
                        "title": "交付条件",
                        "parent_section_id": None,
                        "level": 1,
                        "source_pages": [13],
                    },
                ]
                payload["review_items"][0]["rule_text"] = "末项"
                payload["review_items"][0]["section_path"] = ["交付条件", "付款"]
            return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))

    _, service, rule_set_id = build_import_service(tmp_path, make_result())
    service.extractor = mock.Mock(
        extract=mock.Mock(return_value=ExtractedRuleDocument(
            "动态规则",
            [{"page_number": i, "text": f"第{i}项"} for i in range(1, 13)]
            + [{"page_number": 13, "text": "末项"}],
            13,
        ))
    )
    service.parse_llm = SectionLLM()

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    sections = imported.parsed_rules["sections"]
    assert [section["title"] for section in sections] == [
        "商务条件", "交付条件", "付款",
    ]
    assert sections[2]["parent_section_id"] == sections[1]["section_id"]


def test_same_rule_text_on_different_pages_is_not_collapsed(tmp_path: Path):
    class RepeatedRuleLLM:
        def invoke(self, prompt: str):
            primary = json.loads(
                prompt.split("<rule_document>\n", 1)[1].split(
                    "\n</rule_document>", 1
                )[0]
            )["blocks"]
            page = 13 if any("末项" in block["text"] for block in primary) else 1
            payload = make_result()
            payload["sections"][0]["source_pages"] = [page]
            payload["review_items"][0]["source_pages"] = [page]
            return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))

    _, service, rule_set_id = build_import_service(tmp_path, make_result())
    service.extractor = mock.Mock(
        extract=mock.Mock(return_value=ExtractedRuleDocument(
            "动态规则",
            [{"page_number": i, "text": f"第{i}项"} for i in range(1, 13)]
            + [{"page_number": 13, "text": "末项"}],
            13,
        ))
    )
    service.parse_llm = RepeatedRuleLLM()

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    assert [item["source_pages"] for item in imported.parsed_rules["review_items"]] == [
        [1], [13],
    ]


def test_same_cross_page_rule_is_merged_at_batch_boundary(tmp_path: Path):
    class CrossPageLLM:
        def invoke(self, prompt: str):
            primary = json.loads(
                prompt.split("<rule_document>\n", 1)[1].split(
                    "\n</rule_document>", 1
                )[0]
            )["blocks"]
            page = 13 if any("末项" in block["text"] for block in primary) else 12
            payload = make_result()
            payload["sections"][0]["source_pages"] = [page]
            payload["review_items"][0]["source_pages"] = [page]
            return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))

    _, service, rule_set_id = build_import_service(tmp_path, make_result())
    service.extractor = mock.Mock(
        extract=mock.Mock(return_value=ExtractedRuleDocument(
            "动态规则",
            [{"page_number": i, "text": f"第{i}项"} for i in range(1, 13)]
            + [{"page_number": 13, "text": "末项"}],
            13,
        ))
    )
    service.parse_llm = CrossPageLLM()

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    assert [item["source_pages"] for item in imported.parsed_rules["review_items"]] == [
        [12, 13],
    ]


def test_cross_page_duplicate_keeps_additional_fact_requirements(tmp_path: Path):
    class ComplementaryLLM:
        def invoke(self, prompt: str):
            primary = json.loads(
                prompt.split("<rule_document>\n", 1)[1].split(
                    "\n</rule_document>", 1
                )[0]
            )["blocks"]
            is_last = any("末项" in block["text"] for block in primary)
            payload = make_result()
            payload["review_items"][0]["source_pages"] = [13 if is_last else 12]
            if is_last:
                payload["review_items"][0]["retrieval_queries"].append(
                    "合同约定的付款比例是多少"
                )
                payload["review_items"][0]["fact_requirements"] = [{
                    "fact_key": "payment_ratio",
                    "label": "付款比例",
                    "value_type": "percentage",
                    "required": True,
                }]
            return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))

    _, service, rule_set_id = build_import_service(tmp_path, make_result())
    service.extractor = mock.Mock(
        extract=mock.Mock(return_value=ExtractedRuleDocument(
            "动态规则",
            [{"page_number": i, "text": f"第{i}项"} for i in range(1, 13)]
            + [{"page_number": 13, "text": "末项"}],
            13,
        ))
    )
    service.parse_llm = ComplementaryLLM()

    imported = service.import_rule_set(rule_set_id)

    assert imported.status == "draft"
    item = imported.parsed_rules["review_items"][0]
    assert item["source_pages"] == [12, 13]
    assert item["retrieval_queries"] == [
        "合同中的检查项1", "合同约定的付款比例是多少",
    ]
    assert [fact["fact_key"] for fact in item["fact_requirements"]] == [
        "payment_ratio",
    ]


def test_application_has_no_sample_specific_count_or_classification_lookup():
    app_dir = Path(__file__).parents[1] / "app" / "evidence_review"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in app_dir.glob("*.py")
    )

    assert "42项" not in source
    assert "四十二项" not in source
    assert "EXPECTED_ITEM_COUNT" not in source
    assert "CLASSIFICATION_LOOKUP" not in source
