import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


class CleanMineruDataTest(TestCase):
    def test_clean_items_applies_only_deterministic_rules(self):
        from clean_mineru_data import clean_items

        body = {"type": "text", "text": "合同编号 C-1"}
        placeholder = {"type": "text", "text": "[38]"}
        footer = {"type": "footer", "text": "页脚"}
        missing_text = {"type": "text"}
        non_string_text = {"type": "text", "text": 0}
        items = [
            {"type": "page_number", "text": "2"},
            {"type": "header", "text": "页眉"},
            {"type": "text", "text": " \n\t\u2003 "},
            {"type": "text", "text": " \u3002 "},
            body,
            placeholder,
            footer,
            missing_text,
            non_string_text,
            {"type": "image", "img_path": "image.jpg"},
            "非字典对象",
        ]

        cleaned = clean_items(items)

        self.assertEqual(
            cleaned,
            [
                body,
                placeholder,
                footer,
                missing_text,
                non_string_text,
                {"type": "image", "img_path": "image.jpg"},
                "非字典对象",
            ],
        )

    def test_main_reads_json_and_writes_cleaned_json(self):
        import clean_mineru_data

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw_content_list.json"
            output = root / "cleaned.json"
            source.write_text(
                json.dumps(
                    [
                        {"type": "header", "text": "页眉"},
                        {"type": "text", "text": "正文"},
                        {"type": "text", "text": "。"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            clean_mineru_data.main([str(source), "--output", str(output)])

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                [{"type": "text", "text": "正文"}],
            )
