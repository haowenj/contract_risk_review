from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.evidence_review.rule_import import ExtractedRuleDocument


MAX_PRIMARY_BLOCKS = 12
PRIMARY_TOKEN_BUDGET = 3200
CONTEXT_TOKEN_BUDGET = 350
MAX_UNIT_TOKENS = 2750

_LIST_START = re.compile(
    r"(?m)(?=^[ \t]*(?:"
    r"[一二三四五六七八九十百]+[、.．]"
    r"|\d+(?:\.\d+)*[、.．)]"
    r"|[（(][一二三四五六七八九十百\d]+[）)]"
    r"|[①-⑳]|[•●▪-])[ \t]*)"
)


@dataclass(frozen=True)
class RuleParseBatch:
    document: ExtractedRuleDocument
    context_before: list[dict[str, object]]
    context_after: list[dict[str, object]]


def estimate_tokens(text: str) -> int:
    """Conservative estimate for Chinese text without model-specific tokenizer."""
    return (len(text.encode("utf-8")) + 1) // 2


def _clip(text: str, budget: int, *, from_end: bool = False) -> str:
    if estimate_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[-middle:] if from_end else text[:middle]
        if estimate_tokens(candidate) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[-low:] if from_end else text[:low]


def _split_long_text(text: str) -> list[str]:
    if estimate_tokens(text) <= MAX_UNIT_TOKENS:
        return [text]
    pieces = re.split(r"(?<=[。；;\n])", text)
    result: list[str] = []
    current = ""
    for piece in pieces:
        while estimate_tokens(piece) > MAX_UNIT_TOKENS:
            if current:
                result.append(current)
                current = ""
            prefix = _clip(piece, MAX_UNIT_TOKENS)
            result.append(prefix)
            piece = piece[len(prefix) :]
        if estimate_tokens(current + piece) > MAX_UNIT_TOKENS:
            result.append(current)
            current = piece
        else:
            current += piece
    if current:
        result.append(current)
    return result


def _units(document: ExtractedRuleDocument) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    for block in document.blocks:
        text = str(block["text"])
        for segment in _LIST_START.split(text):
            segment = segment.strip()
            if not segment:
                continue
            for piece in _split_long_text(segment):
                unit: dict[str, object] = {
                    "page_number": block["page_number"],
                    "text": piece,
                }
                if isinstance(block.get("text_level"), int):
                    unit["text_level"] = block["text_level"]
                units.append(unit)
    return units


def _estimated_payload_tokens(
    document: ExtractedRuleDocument,
    blocks: list[dict[str, object]],
) -> int:
    return estimate_tokens(
        json.dumps(
            {
                "title_hint": document.title_hint,
                "page_count": document.page_count,
                "blocks": blocks,
            },
            ensure_ascii=False,
        )
    )


def _context(block: dict[str, object], *, from_end: bool) -> dict[str, object]:
    return {
        "page_number": block["page_number"],
        "text": _clip(str(block["text"]), CONTEXT_TOKEN_BUDGET, from_end=from_end),
    }


def _is_heading(block: dict[str, object]) -> bool:
    if isinstance(block.get("text_level"), int):
        return True
    text = str(block["text"])
    return (
        len(text) <= 80
        and "\n" not in text
        and re.match(
            r"^(?:第[一二三四五六七八九十百\d]+[章节]|[一二三四五六七八九十百]+、)",
            text,
        )
        is not None
    )


def _preceding_context(
    units: list[dict[str, object]],
    start: int,
) -> list[dict[str, object]]:
    if start == 0:
        return []
    previous = units[start - 1]
    heading = next(
        (unit for unit in reversed(units[:start]) if _is_heading(unit)),
        None,
    )
    if heading is None or heading is previous:
        return [_context(previous, from_end=True)]
    return [
        {
            "page_number": heading["page_number"],
            "text": _clip(str(heading["text"]), CONTEXT_TOKEN_BUDGET // 2),
        },
        {
            "page_number": previous["page_number"],
            "text": _clip(
                str(previous["text"]),
                CONTEXT_TOKEN_BUDGET // 2,
                from_end=True,
            ),
        },
    ]


def split_rule_document(document: ExtractedRuleDocument) -> list[RuleParseBatch]:
    units = _units(document)
    if not units:
        raise ValueError("规则文件没有可解析的文本块")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(units):
        end = start
        while end < len(units) and end - start < MAX_PRIMARY_BLOCKS:
            candidate = units[start : end + 1]
            if _estimated_payload_tokens(document, candidate) > PRIMARY_TOKEN_BUDGET:
                break
            end += 1
        if end == start:
            raise ValueError("单个规则文本块超过解析预算")
        ranges.append((start, end))
        start = end

    return [
        RuleParseBatch(
            document=type(document)(
                document.title_hint,
                units[start:end],
                document.page_count,
            ),
            context_before=_preceding_context(units, start),
            context_after=[_context(units[end], from_end=False)]
            if end < len(units)
            else [],
        )
        for start, end in ranges
    ]
