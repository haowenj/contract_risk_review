from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.image_schemas import (
    IMAGE_RESPONSE_FORMAT,
    ImageExtraction,
    validate_image_extraction,
)


IMAGE_EXTRACTION_PROMPT = """请理解这张合同图片，并严格输出 JSON。

先判断图片类型，只能是 bank_account、identity_card 或 general。
再按该类型的 data 字段提取图片中能够可靠看见的内容。
看不清、图片中没有或无法对应的字段必须输出 null，不得猜测。
bank_account 提取 account_name、account_number、bank_name，可选 bank_branch。
identity_card 提取 name、id_number、valid_from、valid_to；只看到一面时另一面的字段必须为 null。
general 提取 visible_text 和 content_description；visible_text 必须忠实保留可见文字。
只输出符合既定 Schema 的 JSON 对象，不要输出解释、Markdown 或自由总结。
"""


def _mime_type(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"不支持的图片类型：{image_path.suffix or image_path.name}")
    return mime_type


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    raise ValueError("Vision response does not contain text content")


def _response_payload(response: Any) -> Any:
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    parsed = additional_kwargs.get("parsed")
    if parsed is not None:
        return parsed
    content = response if isinstance(response, str) else getattr(response, "content", None)
    raw_text = _content_to_text(content)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Vision response is not valid JSON") from exc


class ImageUnderstandingService:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        llm: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._llm = llm
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=model_name,
                api_key=api_key or os.environ.get("LLM_API_KEY"),
                base_url=base_url or os.environ.get("LLM_BASE_URL"),
                temperature=0,
                timeout=timeout_seconds,
                max_retries=0,
                extra_body={"enable_thinking": False},
            ).bind(response_format=IMAGE_RESPONSE_FORMAT)

    def classify_and_extract(self, image_path: Path) -> ImageExtraction:
        resolved = image_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        mime_type = _mime_type(resolved)
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": IMAGE_EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}"
                    },
                },
            ]
        )
        response = self._llm.invoke([message])
        return validate_image_extraction(_response_payload(response))
