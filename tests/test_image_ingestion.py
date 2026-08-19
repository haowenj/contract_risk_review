import json

import pytest

from app.image_ingestion import (
    ContractImageIngestionService,
    write_json_atomic,
)
from app.image_schemas import (
    ImageSchemaError,
    validate_image_extraction,
)


def bank_extraction():
    return validate_image_extraction(
        {
            "image_type": "bank_account",
            "data": {
                "account_name": "甲公司",
                "account_number": "110914414810101",
                "bank_name": "甲银行",
                "bank_branch": None,
            },
        }
    )


def general_extraction():
    return validate_image_extraction(
        {
            "image_type": "general",
            "data": {
                "visible_text": "印章",
                "content_description": "红色圆形印章",
            },
        }
    )


def write_test_image(tmp_path, relative_path):
    image_path = tmp_path / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpeg")
    return image_path


class FakeVision:
    model_name = "test-vl"

    def __init__(self, extraction):
        self.extraction = extraction
        self.calls = []

    def classify_and_extract(self, image_path):
        self.calls.append(image_path)
        return self.extraction


class SequenceVision(FakeVision):
    def __init__(self, extractions):
        self.extractions = list(extractions)
        self.calls = []
        self.model_name = "test-vl"

    def classify_and_extract(self, image_path):
        self.calls.append(image_path)
        return self.extractions.pop(0)


class RaisingVision:
    model_name = "test-vl"

    def __init__(self, error):
        self.error = error

    def classify_and_extract(self, _image_path):
        raise self.error


class FakeOCR:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def extract_text(self, image_path):
        self.calls.append(image_path)
        return self.text


class RaisingOCR(FakeOCR):
    def __init__(self, error):
        super().__init__("")
        self.error = error

    def extract_text(self, image_path):
        self.calls.append(image_path)
        raise self.error


def test_enriches_bank_image_in_place_without_changing_indices(tmp_path):
    image_path = write_test_image(tmp_path, "images/account.jpg")
    objects = [
        {"type": "text", "text": "开户信息"},
        {
            "type": "image",
            "img_path": image_path.relative_to(tmp_path).as_posix(),
            "bbox": [1, 2, 3, 4],
            "page_idx": 2,
        },
    ]
    vision = FakeVision(bank_extraction())
    ocr = FakeOCR("户名：甲公司\n开户银行：甲银行\n账号：110914414810101")
    service = ContractImageIngestionService(
        vision_service=vision,
        ocr_service=ocr,
    )

    enriched = service.enrich_images(objects, storage_dir=tmp_path)

    assert len(enriched) == 2
    assert enriched[0] == objects[0]
    image = enriched[1]
    assert image["img_path"] == "images/account.jpg"
    assert image["image_type"] == "bank_account"
    assert image["structured_data"]["account_number"] == "110914414810101"
    assert image["ocr_status"] == "ready"
    assert image["verification_status"] == "verified"
    assert image["image_processing_status"] == "ready"


def test_missing_image_is_recorded_and_other_images_continue(tmp_path):
    existing = write_test_image(tmp_path, "images/general.jpg")
    objects = [
        {"type": "image", "img_path": "images/missing.jpg"},
        {"type": "image", "img_path": existing.relative_to(tmp_path).as_posix()},
    ]
    service = ContractImageIngestionService(
        vision_service=SequenceVision([general_extraction()]),
        ocr_service=FakeOCR("must not be called"),
    )

    enriched = service.enrich_images(objects, storage_dir=tmp_path)

    assert enriched[0]["image_processing_status"] == "missing_image"
    assert enriched[1]["image_processing_status"] == "ready"


@pytest.mark.parametrize("img_path", ["/etc/passwd", "../outside.jpg", "images/../outside.jpg"])
def test_unsafe_image_path_is_recorded_without_calling_services(tmp_path, img_path):
    vision = FakeVision(general_extraction())
    ocr = FakeOCR("must not be called")
    service = ContractImageIngestionService(
        vision_service=vision,
        ocr_service=ocr,
    )

    image = service.enrich_images(
        [{"type": "image", "img_path": img_path}],
        storage_dir=tmp_path,
    )[0]

    assert image["image_processing_status"] == "missing_image"
    assert vision.calls == []
    assert ocr.calls == []


def test_schema_error_is_recorded_without_ocr(tmp_path):
    image_path = write_test_image(tmp_path, "images/bad.jpg")
    ocr = FakeOCR("must not be called")
    service = ContractImageIngestionService(
        vision_service=RaisingVision(ImageSchemaError("invalid schema")),
        ocr_service=ocr,
    )

    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]

    assert image["image_processing_status"] == "schema_invalid"
    assert image["ocr_status"] == "not_started"
    assert ocr.calls == []


def test_general_never_calls_ocr(tmp_path):
    image_path = write_test_image(tmp_path, "images/general.jpg")
    ocr = FakeOCR("must not be called")
    service = ContractImageIngestionService(
        vision_service=FakeVision(general_extraction()),
        ocr_service=ocr,
    )

    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]

    assert image["ocr_status"] == "not_required"
    assert image["verification_status"] == "not_required"
    assert ocr.calls == []


def test_empty_general_result_is_recorded_without_ocr(tmp_path):
    image_path = write_test_image(tmp_path, "images/empty.jpg")
    empty_general = validate_image_extraction(
        {
            "image_type": "general",
            "data": {
                "visible_text": None,
                "content_description": None,
            },
        }
    )
    ocr = FakeOCR("must not be called")
    service = ContractImageIngestionService(
        vision_service=FakeVision(empty_general),
        ocr_service=ocr,
    )

    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]

    assert image["image_processing_status"] == "empty_result"
    assert image["ocr_status"] == "not_required"
    assert ocr.calls == []


def test_ocr_failure_keeps_bank_structured_data(tmp_path):
    image_path = write_test_image(tmp_path, "images/account.jpg")
    service = ContractImageIngestionService(
        vision_service=FakeVision(bank_extraction()),
        ocr_service=RaisingOCR(RuntimeError("ocr unavailable")),
    )

    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]

    assert image["image_processing_status"] == "ready"
    assert image["structured_data"]["account_number"] == "110914414810101"
    assert image["ocr_status"] == "failed"
    assert image["verification_status"] == "insufficient"


def test_conflict_keeps_vl_value_and_ocr_text(tmp_path):
    image_path = write_test_image(tmp_path, "images/account.jpg")
    service = ContractImageIngestionService(
        vision_service=FakeVision(bank_extraction()),
        ocr_service=FakeOCR(
            "户名：甲公司\n开户银行：甲银行\n账号：110914414810107"
        ),
    )

    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]

    assert image["structured_data"]["account_number"] == "110914414810101"
    assert image["ocr_text"].endswith("107")
    assert image["verification_status"] == "conflict"


def test_atomic_json_write_replaces_complete_utf8_json(tmp_path):
    path = tmp_path / "merged_content_list.json"
    write_json_atomic(path, [{"type": "image", "image_type": "general"}])

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"type": "image", "image_type": "general"}
    ]
    assert not list(tmp_path.glob(".*.tmp"))
