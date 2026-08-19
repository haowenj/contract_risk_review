from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class ImageSchemaError(ValueError):
    """The VL response has a known image type but violates its data schema."""


class ImageClassificationError(ValueError):
    """The VL response cannot be assigned to one of the supported image types."""


class BankAccountData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_name: str | None
    account_number: str | None
    bank_name: str | None
    bank_branch: str | None


class BankAccountExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_type: Literal["bank_account"]
    data: BankAccountData


class IdentityCardData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None
    id_number: str | None
    valid_from: str | None
    valid_to: str | None


class IdentityCardExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_type: Literal["identity_card"]
    data: IdentityCardData


class GeneralImageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_text: str | None
    content_description: str | None


class GeneralImageExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_type: Literal["general"]
    data: GeneralImageData


ImageExtraction = Annotated[
    BankAccountExtraction | IdentityCardExtraction | GeneralImageExtraction,
    Field(discriminator="image_type"),
]

IMAGE_EXTRACTION_ADAPTER = TypeAdapter(ImageExtraction)


def _nullable_string_schema() -> dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _data_schema(fields: tuple[str, ...]) -> dict[str, Any]:
    properties = {
        field: _nullable_string_schema()
        for field in fields
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(fields),
        "additionalProperties": False,
    }


IMAGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "contract_image_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "image_type": {
                    "type": "string",
                    "enum": ["bank_account", "identity_card", "general"],
                },
                "data": {
                    "oneOf": [
                        _data_schema(
                            (
                                "account_name",
                                "account_number",
                                "bank_name",
                                "bank_branch",
                            )
                        ),
                        _data_schema(
                            ("name", "id_number", "valid_from", "valid_to")
                        ),
                        _data_schema(("visible_text", "content_description")),
                    ]
                },
            },
            "required": ["image_type", "data"],
            "additionalProperties": False,
        },
    },
}


def validate_image_extraction(payload: Any) -> ImageExtraction:
    if not isinstance(payload, dict):
        raise ImageSchemaError("image response must be an object")
    if payload.get("image_type") not in {
        "bank_account",
        "identity_card",
        "general",
    }:
        raise ImageClassificationError(
            f"unsupported image_type: {payload.get('image_type')!r}"
        )
    try:
        return IMAGE_EXTRACTION_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ImageSchemaError("invalid image extraction schema") from exc
