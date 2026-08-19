from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from app.image_schemas import (
    ImageClassificationError,
    ImageSchemaError,
)
from image_verification import verify_image_data


logger = logging.getLogger(__name__)


IMAGE_DERIVED_DEFAULTS: dict[str, Any] = {
    "image_processing_status": "ready",
    "image_type": None,
    "structured_data": None,
    "ocr_status": "not_started",
    "ocr_text": None,
    "verification_status": "insufficient",
    "verification_details": {},
    "image_schema_version": "image-v1",
    "image_model": None,
    "image_error": None,
}


def _safe_image_path(storage_dir: Path, img_path: Any) -> Path:
    """Resolve a MinerU image reference without allowing path traversal."""

    if not isinstance(img_path, str) or not img_path.strip():
        raise ValueError("image img_path is empty")

    reference = img_path.strip()
    # MinerU references are POSIX paths. Reject both POSIX and host absolute
    # paths, plus backslash variants that could otherwise become traversal on
    # a different host.
    posix_reference = PurePosixPath(reference.replace("\\", "/"))
    local_reference = Path(reference)
    if posix_reference.is_absolute() or local_reference.is_absolute():
        raise ValueError("image img_path must be relative")
    if ".." in posix_reference.parts:
        raise ValueError("image img_path contains path traversal")

    root = storage_dir.expanduser().resolve()
    target = (root / Path(*posix_reference.parts)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("image img_path resolves outside storage directory")
    if not target.is_file():
        raise FileNotFoundError("image file does not exist")
    return target


def _structured_data(extraction: Any) -> tuple[str | None, dict[str, Any]]:
    image_type = getattr(extraction, "image_type", None)
    data = getattr(extraction, "data", None)
    if isinstance(extraction, Mapping):
        image_type = extraction.get("image_type")
        data = extraction.get("data")
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, Mapping):
        raise ImageSchemaError("image extraction data is not an object")
    return image_type if isinstance(image_type, str) else None, copy.deepcopy(dict(data))


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


class ContractImageIngestionService:
    """Enrich MinerU image objects while isolating failures to each image."""

    def __init__(self, *, vision_service: Any, ocr_service: Any) -> None:
        self.vision_service = vision_service
        self.ocr_service = ocr_service

    def enrich_images(
        self,
        objects: list[dict[str, Any]],
        *,
        storage_dir: Path,
    ) -> list[dict[str, Any]]:
        enriched = copy.deepcopy(objects)
        for source_object_index, image in enumerate(enriched):
            if not isinstance(image, dict) or image.get("type") != "image":
                continue

            image.update(copy.deepcopy(IMAGE_DERIVED_DEFAULTS))
            try:
                image_path = _safe_image_path(storage_dir, image.get("img_path"))
            except (FileNotFoundError, ValueError) as exc:
                image["image_processing_status"] = "missing_image"
                image["image_error"] = _error_text(exc)
                logger.warning(
                    "image source_object_index=%s path=%r unavailable: %s",
                    source_object_index,
                    image.get("img_path"),
                    _error_text(exc),
                )
                continue

            try:
                extraction = self.vision_service.classify_and_extract(image_path)
                image_type, structured_data = _structured_data(extraction)
                if image_type not in {"bank_account", "identity_card", "general"}:
                    raise ImageClassificationError(
                        f"unsupported image_type: {image_type!r}"
                    )
            except ImageClassificationError as exc:
                image["image_processing_status"] = "unclassified"
                image["image_error"] = _error_text(exc)
                logger.warning(
                    "image source_object_index=%s could not be classified: %s",
                    source_object_index,
                    _error_text(exc),
                )
                continue
            except ImageSchemaError as exc:
                image["image_processing_status"] = "schema_invalid"
                image["image_error"] = _error_text(exc)
                logger.warning(
                    "image source_object_index=%s returned invalid schema: %s",
                    source_object_index,
                    _error_text(exc),
                )
                continue
            except Exception as exc:  # noqa: BLE001 - image-level degradation
                image["image_processing_status"] = "vl_failed"
                image["image_error"] = _error_text(exc)
                logger.exception(
                    "image source_object_index=%s vision call failed",
                    source_object_index,
                )
                continue

            image["image_type"] = image_type
            image["structured_data"] = structured_data
            image["image_model"] = getattr(
                self.vision_service, "model_name", None
            )
            if not any(
                isinstance(value, str) and value.strip()
                for value in structured_data.values()
            ):
                image["image_processing_status"] = "empty_result"

            if image_type == "general":
                image["ocr_status"] = "not_required"
                image["verification_status"] = "not_required"
                image["verification_details"] = {}
                continue

            try:
                raw_ocr_text = self.ocr_service.extract_text(image_path)
                ocr_text = raw_ocr_text if isinstance(raw_ocr_text, str) else ""
            except Exception as exc:  # noqa: BLE001 - OCR is optional
                image["ocr_status"] = "failed"
                image["verification_status"] = "insufficient"
                image["verification_details"] = {}
                image["image_error"] = f"OCR failed: {_error_text(exc)}"
                logger.exception(
                    "image source_object_index=%s OCR call failed",
                    source_object_index,
                )
                continue

            image["ocr_text"] = ocr_text or None
            image["ocr_status"] = "ready" if ocr_text.strip() else "empty"
            verification = verify_image_data(image_type, structured_data, ocr_text)
            image["verification_status"] = verification.status
            image["verification_details"] = verification.details
            if not ocr_text.strip():
                logger.warning(
                    "image source_object_index=%s OCR returned no usable text",
                    source_object_index,
                )

        return enriched


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write one complete UTF-8 JSON file and replace the destination atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
