from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _value(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if not isinstance(value, str):
        return None
    value = " ".join(value.split()).strip()
    return value or None


def _verification_line(status: str | None) -> str:
    if status == "verified":
        return "OCR校验：已核验。"
    if status == "conflict":
        return "OCR校验：关键字段存在冲突，需要人工核验。"
    return "OCR校验：信息不足，尚未确认。"


def image_to_searchable_text(image: Mapping[str, Any]) -> str | None:
    image_type = image.get("image_type")
    structured_data = image.get("structured_data")
    if not isinstance(structured_data, Mapping):
        return None

    lines: list[str] = []
    if image_type == "bank_account":
        lines.append("银行账户信息。")
        fields = (
            ("account_name", "户名"),
            ("bank_name", "开户银行"),
            ("bank_branch", "开户支行"),
            ("account_number", "银行账号"),
        )
        for key, label in fields:
            value = _value(structured_data, key)
            if value:
                lines.append(f"{label}：{value}。")
        if len(lines) == 1:
            return None
        lines.append(_verification_line(image.get("verification_status")))
        return "\n".join(lines)

    if image_type == "identity_card":
        lines.append("法人/身份证信息。")
        name = _value(structured_data, "name")
        id_number = _value(structured_data, "id_number")
        valid_from = _value(structured_data, "valid_from")
        valid_to = _value(structured_data, "valid_to")
        if name:
            lines.append(f"姓名：{name}。")
        if id_number:
            lines.append(f"身份证号码：{id_number}。")
        if valid_from or valid_to:
            lines.append(
                f"有效期限：{valid_from or '未识别'} 至 {valid_to or '未识别'}。"
            )
        if len(lines) == 1:
            return None
        lines.append(_verification_line(image.get("verification_status")))
        return "\n".join(lines)

    if image_type == "general":
        description = _value(structured_data, "content_description")
        visible_text = _value(structured_data, "visible_text")
        if description:
            lines.append(f"图片内容：{description}。")
        if visible_text:
            lines.append(f"可见文字：{visible_text}。")
        return "\n".join(lines) or None

    return None
