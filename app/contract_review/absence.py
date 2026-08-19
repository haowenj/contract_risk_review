from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

from image_searchable_text import image_to_searchable_text
from table_searchable_text import table_to_searchable_text


MAX_ABSENCE_CANDIDATES = 20


@dataclass(frozen=True)
class AbsenceScanResult:
    candidates: list[dict[str, Any]]
    candidate_count: int


def normalize_scan_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _object_text(source_object: dict[str, Any]) -> tuple[str, str] | None:
    node_type = source_object.get("type")
    if node_type == "text":
        text = source_object.get("text")
    elif node_type == "table":
        text = table_to_searchable_text(source_object)
    elif node_type == "image":
        text = image_to_searchable_text(source_object)
    else:
        return None

    if not isinstance(text, str) or not text.strip():
        return None
    return node_type, text


def scan_source_objects(
    source_objects: list[dict[str, Any]],
    primary_keywords: list[str],
    secondary_keywords: list[str],
    *,
    limit: int = MAX_ABSENCE_CANDIDATES,
) -> AbsenceScanResult:
    if limit < 1:
        raise ValueError("limit must be positive")

    normalized_primary_keywords = [
        (keyword, normalize_scan_text(keyword))
        for keyword in primary_keywords
    ]
    if not any(value for _, value in normalized_primary_keywords):
        raise ValueError("primary_keywords must contain a usable keyword")
    normalized_secondary_keywords = [
        (keyword, normalize_scan_text(keyword))
        for keyword in secondary_keywords
    ]
    matches: list[dict[str, Any]] = []
    for source_index, source_object in enumerate(source_objects):
        if not isinstance(source_object, dict):
            raise ValueError("merged content objects must be JSON objects")
        extracted = _object_text(source_object)
        if extracted is None:
            continue

        node_type, evidence_text = extracted
        normalized_text = normalize_scan_text(evidence_text)
        matched_primary_keywords = [
            keyword
            for keyword, normalized_keyword in normalized_primary_keywords
            if normalized_keyword and normalized_keyword in normalized_text
        ]
        if not matched_primary_keywords:
            continue
        matched_secondary_keywords = [
            keyword
            for keyword, normalized_keyword in normalized_secondary_keywords
            if normalized_keyword and normalized_keyword in normalized_text
        ]

        matches.append(
            {
                "source_object_index": source_index,
                "page_idx": source_object.get("page_idx"),
                "node_type": node_type,
                "matched_primary_keywords": matched_primary_keywords,
                "matched_secondary_keywords": matched_secondary_keywords,
                "matched_keywords": [
                    *matched_primary_keywords,
                    *matched_secondary_keywords,
                ],
                "evidence_text": evidence_text,
                "text": evidence_text,
            }
        )

    matches.sort(
        key=lambda value: (
            -len(value["matched_primary_keywords"]),
            -len(value["matched_secondary_keywords"]),
            -max(
                len(normalize_scan_text(keyword))
                for keyword in value["matched_primary_keywords"]
            ),
            value["source_object_index"],
        )
    )
    return AbsenceScanResult(
        candidates=matches[:limit],
        candidate_count=len(matches),
    )
