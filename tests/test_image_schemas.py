import pytest

from app.image_schemas import (
    IMAGE_RESPONSE_FORMAT,
    ImageClassificationError,
    ImageSchemaError,
    validate_image_extraction,
)


def test_bank_account_schema_accepts_nullable_fields_and_forbids_extras():
    extraction = validate_image_extraction(
        {
            "image_type": "bank_account",
            "data": {
                "account_name": "甲公司",
                "account_number": "110914414810101",
                "bank_name": "中国甲银行",
                "bank_branch": None,
            },
        }
    )
    assert extraction.image_type == "bank_account"
    assert extraction.data.account_number == "110914414810101"

    with pytest.raises(ImageSchemaError):
        validate_image_extraction(
            {
                "image_type": "bank_account",
                "data": {
                    "account_name": "甲公司",
                    "account_number": "1",
                    "bank_name": "银行",
                    "bank_branch": None,
                    "guessed_field": "forbidden",
                },
            }
        )


def test_identity_and_general_schema_accept_nullable_values():
    identity = validate_image_extraction(
        {
            "image_type": "identity_card",
            "data": {
                "name": "张三",
                "id_number": None,
                "valid_from": "2020.01.01",
                "valid_to": None,
            },
        }
    )
    general = validate_image_extraction(
        {
            "image_type": "general",
            "data": {
                "visible_text": "印章",
                "content_description": None,
            },
        }
    )

    assert identity.data.name == "张三"
    assert general.data.visible_text == "印章"


def test_unknown_type_raises_classification_error():
    with pytest.raises(ImageClassificationError):
        validate_image_extraction({"image_type": "chart", "data": {}})


def test_branch_mismatch_and_missing_fields_raise_schema_error():
    with pytest.raises(ImageSchemaError):
        validate_image_extraction(
            {
                "image_type": "bank_account",
                "data": {
                    "name": "张三",
                    "id_number": None,
                    "valid_from": None,
                    "valid_to": None,
                },
            }
        )

    with pytest.raises(ImageSchemaError):
        validate_image_extraction(
            {
                "image_type": "identity_card",
                "data": {"name": "张三"},
            }
        )


def test_response_format_is_strict_and_has_three_data_branches():
    schema = IMAGE_RESPONSE_FORMAT["json_schema"]["schema"]
    assert IMAGE_RESPONSE_FORMAT["type"] == "json_schema"
    assert IMAGE_RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert len(schema["properties"]["data"]["oneOf"]) == 3
