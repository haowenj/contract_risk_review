from __future__ import annotations

import re


def parse_expected_indices(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return []
    tokens = re.split(r"[\s,，]+", value)
    if any(not token.isdigit() for token in tokens):
        raise ValueError("正确 Node ID 必须是数字")
    return list(dict.fromkeys(int(token) for token in tokens))


def format_expected_indices(indices: list[int]) -> str:
    return ", ".join(str(index) for index in indices)
