from unittest import TestCase

from table_searchable_text import table_to_searchable_text


class TableSearchableTextTest(TestCase):
    def test_table_to_searchable_text_preserves_rows_and_caption_footnote(self):
        table = {
            "table_caption": ["付款计划"],
            "table_body": (
                "<table><tr><th>期次</th><th>比例</th></tr>"
                "<tr><td>1</td><td>30%</td></tr></table>"
            ),
            "table_footnote": ["以到账为准"],
        }

        self.assertEqual(
            table_to_searchable_text(table),
            "表格标题：付款计划\n"
            "第1行：期次 | 比例\n"
            "第2行：1 | 30%\n"
            "表格注释：以到账为准",
        )

    def test_table_to_searchable_text_uses_available_fields_when_body_is_missing(self):
        self.assertEqual(
            table_to_searchable_text(
                {"table_caption": [], "table_footnote": ["说明"]}
            ),
            "表格注释：说明",
        )

    def test_table_to_searchable_text_removes_nested_html_but_keeps_cell_order(self):
        searchable = table_to_searchable_text(
            {
                "table_body": (
                    "<table><tr><td><strong>甲方</strong><br>付款</td>"
                    "<td>乙方 &amp; 供应商</td></tr></table>"
                )
            }
        )

        self.assertEqual(searchable, "第1行：甲方 付款 | 乙方 & 供应商")
        self.assertNotIn("<strong>", searchable)
        self.assertNotIn("<table>", searchable)

    def test_table_to_searchable_text_uses_fixed_text_when_all_fields_are_empty(self):
        self.assertEqual(table_to_searchable_text({}), "表格")
