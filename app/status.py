from __future__ import annotations


STATUS_LABELS = {
    "queued": "排队中",
    "processing": "处理中",
    "ready": "已就绪",
    "failed": "失败",
}

PROCESSING_STAGE_LABELS = {
    "document_parsing": "正在完成文档解析",
    "vector_index": "文档解析完成，正在构建向量索引",
    "bm25_index": "文档解析完成，向量索引构建完成，正在构建 BM25 索引",
    "completed": "文档解析完成，向量索引构建完成，BM25 索引构建完成",
}


def status_label(status: str | None) -> str:
    if status is None:
        return ""
    return STATUS_LABELS.get(status, status)


def processing_stage_label(stage: str | None) -> str:
    if stage is None:
        return ""
    return PROCESSING_STAGE_LABELS.get(stage, stage)
