from __future__ import annotations

from collections.abc import Iterable, Mapping


EVIDENCE_STATUSES = (
    "found",
    "not_found",
    "source_missing",
    "extraction_failed",
)

IDENTIFIER_LABELS = {
    "credit_code": "统一社会信用代码",
    "owner_credit_code": "业主统一社会信用代码",
    "counterparty_credit_code": "相对方统一社会信用代码",
    "company_name": "企业名称",
    "owner_name": "业主名称",
    "counterparty_name": "相对方名称",
    "project_name": "项目名称",
    "project_location": "项目所在地",
    "legal_representative": "法定代表人",
}


def summarize_evidence_statuses(entries: Iterable[Mapping]) -> dict[str, int]:
    counts = {status: 0 for status in EVIDENCE_STATUSES}
    for entry in entries:
        status = entry["evidence_package"]["evidence_status"]
        if status in counts:
            counts[status] += 1
    return counts


def identifier_label(key: str) -> str:
    return IDENTIFIER_LABELS.get(key, "其他查询信息")
