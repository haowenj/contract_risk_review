from __future__ import annotations


STATUS_LABELS = {
    "queued": "排队中",
    "processing": "处理中",
    "ready": "可问答",
    "failed": "失败",
}


def status_label(status: str | None) -> str:
    if status is None:
        return ""
    return STATUS_LABELS.get(status, status)
