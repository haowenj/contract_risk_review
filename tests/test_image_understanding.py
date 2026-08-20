import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.image_schemas import IMAGE_RESPONSE_FORMAT, ImageSchemaError
from app.image_understanding import ImageUnderstandingService


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def bank_payload():
    return {
        "image_type": "bank_account",
        "data": {
            "account_name": "甲公司",
            "account_number": "110914414810101",
            "bank_name": "甲银行",
            "bank_branch": None,
        },
    }


def test_classify_and_extract_sends_one_image_request(tmp_path):
    image_path = tmp_path / "account.jpg"
    image_path.write_bytes(b"jpeg")
    llm = FakeLLM(bank_payload())
    service = ImageUnderstandingService(model_name="test-vl", llm=llm)

    result = service.classify_and_extract(image_path)

    assert result.image_type == "bank_account"
    assert len(llm.messages) == 1
    content = llm.messages[0][0].content
    assert any(block.get("type") == "image_url" for block in content)
    assert "base64," in json.dumps(content)


def test_classify_and_extract_reads_provider_parsed_payload(tmp_path):
    image_path = tmp_path / "account.png"
    image_path.write_bytes(b"png")
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content="ignored",
        additional_kwargs={"parsed": bank_payload()},
    )
    service = ImageUnderstandingService(model_name="test-vl", llm=llm)

    result = service.classify_and_extract(image_path)

    assert result.data.account_name == "甲公司"


def test_schema_failure_is_exposed_without_retry(tmp_path):
    image_path = tmp_path / "account.jpg"
    image_path.write_bytes(b"jpeg")
    llm = FakeLLM({"image_type": "bank_account", "data": {}})
    service = ImageUnderstandingService(model_name="test-vl", llm=llm)

    with pytest.raises(ImageSchemaError):
        service.classify_and_extract(image_path)
    assert len(llm.messages) == 1


def test_non_json_response_is_reported_as_schema_error(tmp_path):
    image_path = tmp_path / "account.jpg"
    image_path.write_bytes(b"jpeg")
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="not-json")
    service = ImageUnderstandingService(model_name="test-vl", llm=llm)

    with pytest.raises(ImageSchemaError):
        service.classify_and_extract(image_path)


def test_default_llm_binds_strict_image_response_format():
    factory = Mock()
    model = factory.return_value
    bound = model.bind.return_value
    with patch("app.image_understanding.ChatOpenAI", factory):
        service = ImageUnderstandingService(
            model_name="test-vl",
            api_key="test-key",
            base_url="https://llm.test/v1",
            timeout_seconds=120,
        )

    factory.assert_called_once_with(
        model="test-vl",
        api_key="test-key",
        base_url="https://llm.test/v1",
        temperature=0,
        timeout=120,
        max_retries=0,
        reasoning_effort="none",
    )
    model.bind.assert_called_once_with(response_format=IMAGE_RESPONSE_FORMAT)
    assert service._llm is bound
