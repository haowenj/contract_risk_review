from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


EDGE_RATIO = 0.2
_COMPLETE_ENDINGS = frozenset("。！？.!?；;…")
_CLOSING_CHARS = frozenset("\"'”’）)]】》」』")
_HEADING_PATTERNS = (
    re.compile(r"^\s*(?:附件|附录)\s*[0-9０-９一二三四五六七八九十百千万]+"),
    re.compile(r"^\s*第\s*[0-9０-９一二三四五六七八九十百千万]+\s*条"),
    re.compile(r"^\s*[一二三四五六七八九十百千万]+[、.．]"),
    re.compile(r"^\s*[0-9０-９]+[、.．.)）]"),
    re.compile(r"^\s*[（(][0-9０-９一二三四五六七八九十百千万]+[）)]"),
    re.compile(r"^\s*[➢•·▪◦○●■□◆◇—–-]\s*"),
)


def _bbox_y(item: Any) -> tuple[float, float] | None:
    if not isinstance(item, dict):
        return None
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    y0, y1 = bbox[1], bbox[3]
    if isinstance(y0, bool) or isinstance(y1, bool):
        return None
    if not isinstance(y0, (int, float)) or not isinstance(y1, (int, float)):
        return None
    return min(float(y0), float(y1)), max(float(y0), float(y1))


def _page_y_bounds(items: list[Any]) -> dict[int, tuple[float, float]]:
    bounds: dict[int, list[float]] = {}
    for item in items:
        if not isinstance(item, dict) or type(item.get("page_idx")) is not int:
            continue
        y_range = _bbox_y(item)
        if y_range is None:
            continue
        page_idx = item["page_idx"]
        page_bounds = bounds.setdefault(page_idx, [])
        page_bounds.extend(y_range)
    return {page: (min(values), max(values)) for page, values in bounds.items()}


def _is_at_bottom(item: dict[str, Any], bounds: dict[int, tuple[float, float]]) -> bool:
    y_range = _bbox_y(item)
    page_idx = item.get("page_idx")
    page_bounds = bounds.get(page_idx)
    if y_range is None or page_bounds is None:
        return False
    minimum, maximum = page_bounds
    span = maximum - minimum
    return span > 0 and y_range[1] >= minimum + (1 - EDGE_RATIO) * span


def _is_at_top(item: dict[str, Any], bounds: dict[int, tuple[float, float]]) -> bool:
    y_range = _bbox_y(item)
    page_idx = item.get("page_idx")
    page_bounds = bounds.get(page_idx)
    if y_range is None or page_bounds is None:
        return False
    minimum, maximum = page_bounds
    span = maximum - minimum
    return span > 0 and y_range[0] <= minimum + EDGE_RATIO * span


def _looks_like_new_heading(text: str) -> bool:
    return any(pattern.match(text) for pattern in _HEADING_PATTERNS)


def _has_complete_ending(text: str) -> bool:
    text = text.rstrip()
    while text and text[-1] in _CLOSING_CHARS:
        text = text[:-1].rstrip()
    return bool(text) and text[-1] in _COMPLETE_ENDINGS


def _ends_with_colon(text: str) -> bool:
    text = text.rstrip()
    while text and text[-1] in _CLOSING_CHARS:
        text = text[:-1].rstrip()
    return bool(text) and text[-1] in {"：", ":"}


def _can_merge(
    previous: Any,
    next_item: Any,
    bounds: dict[int, tuple[float, float]],
) -> bool:
    if not isinstance(previous, dict) or not isinstance(next_item, dict):
        return False
    if previous.get("type") != "text" or next_item.get("type") != "text":
        return False
    if "text_level" in previous:
        return False
    if "text_level" in next_item:
        return False

    previous_page = previous.get("page_idx")
    next_page = next_item.get("page_idx")
    if type(previous_page) is not int or type(next_page) is not int:
        return False
    if next_page != previous_page + 1:
        return False
    if not _is_at_bottom(previous, bounds) or not _is_at_top(next_item, bounds):
        return False

    previous_text = previous.get("text")
    next_text = next_item.get("text")
    if not isinstance(previous_text, str) or not previous_text.strip():
        return False
    if not isinstance(next_text, str) or not next_text.strip():
        return False
    if _looks_like_new_heading(next_text):
        return False
    if _ends_with_colon(previous_text):
        return False
    return not _has_complete_ending(previous_text)


def _merge_result(
    current: dict[str, Any],
    source_items: list[dict[str, Any]],
    next_item: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(current)
    merged["text"] = merged["text"].rstrip() + next_item["text"].lstrip()
    all_sources = [*source_items, next_item]
    merged["start_page_idx"] = all_sources[0]["page_idx"]
    merged["end_page_idx"] = all_sources[-1]["page_idx"]
    merged["source_page_indices"] = [item["page_idx"] for item in all_sources]
    merged["source_bboxes"] = [copy.deepcopy(item.get("bbox")) for item in all_sources]
    merged["merged_cross_page"] = True
    return merged


def merge_items(items: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    bounds = _page_y_bounds(items)
    merged_items: list[Any] = []
    logs: list[dict[str, Any]] = []
    index = 0

    while index < len(items):
        current = copy.deepcopy(items[index])
        source_items = [items[index]] if isinstance(items[index], dict) else []
        next_index = index + 1

        while source_items and next_index < len(items):
            previous = source_items[-1]
            next_item = items[next_index]
            if not _can_merge(previous, next_item, bounds):
                break

            current = _merge_result(current, source_items, next_item)
            logs.append(
                {
                    "previous_index": next_index - 1,
                    "next_index": next_index,
                    "previous_page_idx": previous["page_idx"],
                    "next_page_idx": next_item["page_idx"],
                    "a": copy.deepcopy(previous),
                    "b": copy.deepcopy(next_item),
                    "merged": copy.deepcopy(current),
                }
            )
            source_items.append(next_item)
            next_index += 1

        merged_items.append(current)
        index = next_index if next_index > index + 1 else index + 1

    return merged_items, logs


def merge_content_list_file(
    source: Path,
    output: Path,
    log_output: Path | None = None,
) -> list[dict[str, Any]]:
    with source.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("content list 的 JSON 顶层必须是数组")

    merged_items, logs = merge_items(items)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(merged_items, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if log_output is not None:
        log_output.parent.mkdir(parents=True, exist_ok=True)
        with log_output.open("w", encoding="utf-8") as handle:
            json.dump(logs, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return logs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="合并 MinerU 跨页段落并保存详细日志")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)

    source = args.input_json.expanduser().resolve()
    project_dir = Path(__file__).resolve().parent
    output = args.output or project_dir / f"{source.stem}_merged.json"
    logs = merge_content_list_file(source, output, args.log)

    for log in logs:
        print(f"跨页合并日志: {json.dumps(log, ensure_ascii=False)}")
    print(f"实际合并数量: {len(logs)}")
    print(f"已写入: {output}")
    if args.log is not None:
        print(f"合并日志已写入: {args.log}")


if __name__ == "__main__":
    main()
