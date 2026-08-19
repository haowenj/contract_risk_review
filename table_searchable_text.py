from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any

from image_searchable_text import image_to_searchable_text


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()
        if tag == "tr":
            if self._current_row is not None:
                self._finish_row()
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            if self._current_cell is not None:
                self._finish_cell()
            self._current_cell = []
        elif tag == "br" and self._current_cell is not None:
            self._current_cell.append(" ")
        elif tag in {"li", "p", "div"} and self._current_cell is not None:
            self._current_cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def _finish_cell(self) -> None:
        if self._current_cell is None:
            return
        if self._current_row is None:
            self._current_cell = None
            return
        self._current_row.append(" ".join("".join(self._current_cell).split()))
        self._current_cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._current_row:
            self.rows.append(self._current_row)
        self._current_row = None

    def finish(self) -> list[list[str]]:
        self._finish_row()
        return self.rows


def _field_values(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [item for item in value if isinstance(item, str)]
    else:
        values = []
    return [" ".join(value.split()) for value in values if value.strip()]


def _parse_rows(table_body: Any) -> list[list[str]]:
    if not isinstance(table_body, str) or not table_body.strip():
        return []
    parser = _TableHTMLParser()
    parser.feed(table_body)
    parser.close()
    return parser.finish()


def table_to_searchable_text(table: Mapping[str, Any]) -> str:
    """Convert one MinerU table object into deterministic plain text."""
    sections: list[str] = []

    for caption in _field_values(table.get("table_caption")):
        sections.append(f"表格标题：{caption}")

    for row_index, row in enumerate(_parse_rows(table.get("table_body")), start=1):
        sections.append(f"第{row_index}行：{' | '.join(row)}")

    for footnote in _field_values(table.get("table_footnote")):
        sections.append(f"表格注释：{footnote}")

    image_text = image_to_searchable_text(table)
    if image_text:
        sections.append(f"图片识别补充：\n{image_text}")

    return "\n".join(sections) or "表格"
