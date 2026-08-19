from __future__ import annotations

from typing import Any


_COMMON_METADATA_KEYS = (
    "page_idx",
    "start_page_idx",
    "end_page_idx",
    "source_page_indices",
    "source_bboxes",
    "merged_cross_page",
    "retrieval_context",
)

_IMAGE_METADATA_KEYS = (
    "img_path",
    "image_type",
    "structured_data",
    "ocr_text",
    "ocr_status",
    "verification_status",
    "verification_details",
    "image_processing_status",
    "image_schema_version",
    "image_model",
    "image_caption",
    "image_footnote",
)


def _add_metadata(
    serialized: dict[str, Any],
    metadata: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            serialized[key] = metadata[key]


def serialize_node_result(result: Any) -> dict[str, Any]:
    """Serialize one Vector/Rerank/Selected result for chat and Evaluation."""

    node = result.node
    metadata = getattr(node, "metadata", {}) or {}
    node_type = metadata.get("node_type") or "text"
    evidence_text = getattr(node, "text", "")
    serialized: dict[str, Any] = {
        "source_object_index": metadata.get("source_object_index"),
        "node_id": getattr(node, "node_id", None),
        "node_type": node_type,
        "text": evidence_text,
        "evidence_text": evidence_text,
    }
    _add_metadata(serialized, metadata, _COMMON_METADATA_KEYS)

    if node_type == "table":
        _add_metadata(
            serialized,
            metadata,
            (
                "bbox",
                "table_body",
                "table_caption",
                "table_footnote",
                "img_path",
            ),
        )
    elif node_type == "image":
        _add_metadata(serialized, metadata, ("bbox", *_IMAGE_METADATA_KEYS))

    score = getattr(result, "score", None)
    if score is not None:
        serialized["score"] = score
    retrieval_score = metadata.get("retrieval_score")
    if retrieval_score is not None:
        serialized["retrieval_score"] = retrieval_score
    return serialized
