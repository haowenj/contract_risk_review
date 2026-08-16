from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any


def _should_remove(item: Any) -> bool:
    if not isinstance(item, dict):
        return False

    item_type = item.get("type")
    if item_type in {"page_number", "header"}:
        return True
    if item_type != "text" or not isinstance(item.get("text"), str):
        return False

    compact_text = "".join(char for char in item["text"] if not char.isspace())
    if not compact_text:
        return True
    return len(compact_text) == 1 and unicodedata.category(compact_text).startswith(
        "P"
    )


def clean_items(items: list[Any]) -> list[Any]:
    return [item for item in items if not _should_remove(item)]


def clean_content_list_file(source: Path, output: Path) -> None:
    with source.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("content list 的 JSON 顶层必须是数组")

    cleaned = clean_items(items)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="对 MinerU content list 做确定性清洗")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    source = args.input_json.expanduser().resolve()
    output = args.output or (
        Path(__file__).resolve().parent / f"{source.stem}_cleaned.json"
    )
    clean_content_list_file(source, output)
    print(f"已写入: {output}")


if __name__ == "__main__":
    main()
