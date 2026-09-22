from pathlib import Path

import pytest

from app.evidence_review.repository import (
    EvidenceReviewRepository,
    RuleSetTransitionError,
)


def parsed_rules() -> dict:
    return {
        "document_title": "审核规则",
        "sections": [],
        "review_items": [{"item_id": "item_1"}],
    }


def test_rule_set_moves_from_queued_to_draft_to_active(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    created = repository.create_rule_set(
        name="通用风险规则",
        source_filename="rules.pdf",
        source_path=tmp_path / "rules" / "source.pdf",
        source_sha256="abc123",
    )

    assert created.status == "queued"
    assert created.version == 1
    repository.mark_rule_set_processing(created.rule_set_id)
    draft = repository.mark_rule_set_draft(created.rule_set_id, parsed_rules())
    assert draft.status == "draft"
    active = repository.activate_rule_set(created.rule_set_id)
    assert active.status == "active"
    assert active.parsed_rules == parsed_rules()


def test_rule_set_versions_increment_for_same_name(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")

    first = repository.create_rule_set(
        name="规则", source_filename="a.md",
        source_path=tmp_path / "a.md", source_sha256="a",
    )
    second = repository.create_rule_set(
        name="规则", source_filename="b.md",
        source_path=tmp_path / "b.md", source_sha256="b",
    )

    assert (first.version, second.version) == (1, 2)


def test_active_rule_set_cannot_be_overwritten(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    record = repository.create_rule_set(
        name="规则", source_filename="rules.md",
        source_path=tmp_path / "rules.md", source_sha256="abc",
    )
    repository.mark_rule_set_processing(record.rule_set_id)
    repository.mark_rule_set_draft(record.rule_set_id, parsed_rules())
    repository.activate_rule_set(record.rule_set_id)

    with pytest.raises(RuleSetTransitionError):
        repository.mark_rule_set_draft(record.rule_set_id, parsed_rules())


def test_failed_rule_set_keeps_safe_error_text(tmp_path: Path):
    repository = EvidenceReviewRepository(tmp_path / "contracts.db")
    record = repository.create_rule_set(
        name="规则", source_filename="rules.pdf",
        source_path=tmp_path / "rules.pdf", source_sha256="abc",
    )
    repository.mark_rule_set_processing(record.rule_set_id)

    failed = repository.mark_rule_set_failed(record.rule_set_id, "规则文件解析失败")

    assert failed.status == "failed"
    assert failed.error_message == "规则文件解析失败"

