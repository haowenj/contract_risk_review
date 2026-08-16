import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


class MergeCrossPageParagraphsTest(TestCase):
    def test_merge_keeps_page_information_and_logs_full_pair(self):
        from merge_cross_page_paragraphs import merge_items

        items = [
            {
                "type": "text",
                "text": "上一页段落未完",
                "bbox": [10, 800, 300, 950],
                "page_idx": 0,
            },
            {
                "type": "text",
                "text": "下一页继续完成。",
                "bbox": [10, 10, 300, 150],
                "page_idx": 1,
            },
            {
                "type": "image",
                "bbox": [10, 800, 300, 950],
                "page_idx": 1,
            },
        ]
        original = copy.deepcopy(items)

        merged, logs = merge_items(items)

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            merged[0],
                {
                    "type": "text",
                    "text": "上一页段落未完下一页继续完成。",
                "bbox": [10, 800, 300, 950],
                "page_idx": 0,
                "start_page_idx": 0,
                "end_page_idx": 1,
                "source_page_indices": [0, 1],
                "source_bboxes": [[10, 800, 300, 950], [10, 10, 300, 150]],
                "merged_cross_page": True,
            },
        )
        self.assertEqual(items, original)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["a"], items[0])
        self.assertEqual(logs[0]["b"], items[1])
        self.assertEqual(logs[0]["merged"], merged[0])

    def test_does_not_merge_when_any_safety_condition_fails(self):
        from merge_cross_page_paragraphs import merge_items

        cases = [
            (
                "A 不在底部",
                [
                    {"type": "text", "text": "A", "bbox": [0, 100, 10, 200], "page_idx": 0},
                    {"type": "text", "text": "B", "bbox": [0, 0, 10, 100], "page_idx": 1},
                    {"type": "image", "bbox": [0, 0, 10, 950], "page_idx": 0},
                ],
            ),
            (
                "B 不在顶部",
                [
                    {"type": "text", "text": "A", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "B", "bbox": [0, 800, 10, 950], "page_idx": 1},
                    {"type": "image", "bbox": [0, 0, 10, 950], "page_idx": 1},
                ],
            ),
            (
                "B 有 text_level",
                [
                    {"type": "text", "text": "A", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "B", "bbox": [0, 0, 10, 100], "page_idx": 1, "text_level": 1},
                ],
            ),
            (
                "A 有 text_level",
                [
                    {"type": "text", "text": "上一页段落未完", "text_level": 2, "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "下一页继续完成。", "bbox": [0, 0, 10, 100], "page_idx": 1},
                ],
            ),
            (
                "A 以中文冒号结束",
                [
                    {"type": "text", "text": "上一页段落：", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "下一页继续说明", "bbox": [0, 0, 10, 100], "page_idx": 1},
                ],
            ),
            (
                "A 以英文冒号结束",
                [
                    {"type": "text", "text": "上一页段落:", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "下一页继续说明", "bbox": [0, 0, 10, 100], "page_idx": 1},
                ],
            ),
            (
                "B 像新条款",
                [
                    {"type": "text", "text": "A", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "第 2 条 付款方式", "bbox": [0, 0, 10, 100], "page_idx": 1},
                ],
            ),
            (
                "B 像附件标题",
                [
                    {"type": "text", "text": "A", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "附件 4《合同变更申请单》", "bbox": [0, 0, 10, 100], "page_idx": 1},
                ],
            ),
            (
                "B 像项目符号",
                [
                    {"type": "text", "text": "A", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "➢支持模块化扩展", "bbox": [0, 0, 10, 100], "page_idx": 1},
                ],
            ),
            (
                "A 已结束",
                [
                    {"type": "text", "text": "A。", "bbox": [0, 800, 10, 950], "page_idx": 0},
                    {"type": "text", "text": "B", "bbox": [0, 0, 10, 100], "page_idx": 1},
                ],
            ),
        ]

        for name, items in cases:
            with self.subTest(name=name):
                merged, logs = merge_items(items)
                self.assertEqual(merged, items)
                self.assertEqual(logs, [])

    def test_cli_writes_merged_json_and_log(self):
        import merge_cross_page_paragraphs

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "cleaned.json"
            output = root / "merged.json"
            log_output = root / "merge_log.json"
            source.write_text(
                json.dumps(
                    [
                        {"type": "text", "text": "A", "bbox": [0, 800, 10, 950], "page_idx": 0},
                        {"type": "text", "text": "B", "bbox": [0, 0, 10, 100], "page_idx": 1},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                merge_cross_page_paragraphs.main(
                    [
                        str(source),
                        "--output",
                        str(output),
                        "--log",
                        str(log_output),
                    ]
                )

            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["text"], "AB")
            saved_logs = json.loads(log_output.read_text(encoding="utf-8"))
            self.assertEqual(len(saved_logs), 1)
            self.assertIn('"a"', stdout.getvalue())
            self.assertIn('"merged"', stdout.getvalue())
