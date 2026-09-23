from app.evidence_review.prompts import build_rule_parse_prompt
from app.evidence_review.rule_batching import estimate_tokens, split_rule_document
from app.evidence_review.rule_import import ExtractedRuleDocument


def test_one_mineru_block_with_many_list_items_is_split_by_item_boundaries():
    text = "\n".join(f"{i}. 核验第{i}项合同事实" for i in range(1, 31))
    document = ExtractedRuleDocument(
        title_hint="通用风险清单",
        blocks=[{"page_number": 1, "text": text}],
        page_count=1,
    )

    batches = split_rule_document(document)

    assert len(batches) == 3
    assert [len(batch.document.blocks) for batch in batches] == [12, 12, 6]
    assert [
        block["text"] for batch in batches for block in batch.document.blocks
    ] == text.splitlines()
    assert all(
        estimate_tokens(
            build_rule_parse_prompt(
                batch.document,
                context_before=batch.context_before,
                context_after=batch.context_after,
            )
        )
        <= 6000
        for batch in batches
    )


def test_oversized_unbroken_text_is_split_without_dropping_characters():
    text = "需核验合同约定" * 1500
    document = ExtractedRuleDocument(
        title_hint="通用风险清单",
        blocks=[{"page_number": 2, "text": text}],
        page_count=2,
    )

    batches = split_rule_document(document)

    assert (
        "".join(block["text"] for batch in batches for block in batch.document.blocks)
        == text
    )
    assert all(
        estimate_tokens(
            build_rule_parse_prompt(
                batch.document,
                context_before=batch.context_before,
                context_after=batch.context_after,
            )
        )
        <= 6000
        for batch in batches
    )


def test_later_batch_keeps_earlier_section_heading_as_context():
    document = ExtractedRuleDocument(
        title_hint="通用风险清单",
        blocks=[
            {"page_number": 1, "text": "商务条件", "text_level": 1},
            *[{"page_number": i, "text": f"{i}. 核验第{i}项"} for i in range(1, 14)],
        ],
        page_count=13,
    )

    batches = split_rule_document(document)

    assert len(batches) == 2
    assert batches[1].context_before[0]["text"] == "商务条件"
    assert batches[1].context_before[-1]["text"] == "11. 核验第11项"
