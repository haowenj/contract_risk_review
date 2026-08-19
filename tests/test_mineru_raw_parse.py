import io
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import httpx


class MinerURawParseTest(TestCase):
    def test_invalid_image_reference_is_skipped_without_writing_outside_output(self):
        import mineru_raw_parse

        raw_content = b'[{"type":"image","img_path":"../escape.jpg"}]'
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("contract_content_list.json", raw_content)
            archive.writestr("escape.jpg", b"must-not-be-written")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content_member, _, content_list = (
                mineru_raw_parse._read_content_list_archive(
                    archive_buffer.getvalue()
                )
            )
            mineru_raw_parse._write_referenced_images(
                archive_buffer.getvalue(),
                content_member=content_member,
                content_list=content_list,
                output_dir=root,
            )

            self.assertFalse((root.parent / "escape.jpg").exists())
            self.assertFalse((root / "escape.jpg").exists())

    def test_main_defaults_output_to_project_directory(self):
        import mineru_raw_parse

        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "contract.pdf"
            pdf_path.write_bytes(b"%PDF-test")
            expected_output = (
                Path(mineru_raw_parse.__file__).resolve().parent
                / "contract_mineru_raw.json"
            )

            with patch.object(mineru_raw_parse, "run_parse") as run_parse:
                mineru_raw_parse.main([str(pdf_path)])

            run_parse.assert_called_once()
            self.assertEqual(run_parse.call_args.args[1], expected_output)

    def test_saves_raw_content_list_and_prints_statistics(self):
        import mineru_raw_parse

        raw_content = (
            b'[{"type":"text","page_idx":0},\n '
            b'{"type":"table","img_path":"images/table.jpg",'
            b'"page_idx":1}, '
            b'{"type":"image","img_path":"images/account.jpg",'
            b'"page_idx":1}, {"page_idx":1}]'
        )
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("contract_content_list.json", raw_content)
            archive.writestr("images/account.jpg", b"jpeg-bytes")
            archive.writestr("images/table.jpg", b"table-jpeg-bytes")

        status_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal status_calls
            if request.method == "POST" and request.url.path == "/tasks":
                request_body = request.read()
                self.assertIn(b"return_content_list", request_body)
                self.assertIn(b"response_format_zip", request_body)
                return httpx.Response(
                    202,
                    json={
                        "task_id": "task-1",
                        "status_url": "http://mineru.test/status/task-1",
                        "result_url": "http://mineru.test/result/task-1",
                    },
                )
            if request.url.path == "/status/task-1":
                status_calls += 1
                status = "pending" if status_calls == 1 else "completed"
                return httpx.Response(200, json={"status": status})
            if request.url.path == "/result/task-1":
                return httpx.Response(
                    200,
                    headers={"content-type": "application/zip"},
                    content=archive_buffer.getvalue(),
                )
            return httpx.Response(404)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "contract.pdf"
            output_path = root / "raw.json"
            pdf_path.write_bytes(b"%PDF-test")
            client = httpx.Client(transport=httpx.MockTransport(handler))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                mineru_raw_parse.run_parse(
                    pdf_path,
                    output_path,
                    client=client,
                    svr_url="http://mineru.test",
                    backend="hybrid-engine",
                    server_url=None,
                    poll_interval=0,
                )
            client.close()

            self.assertEqual(output_path.read_bytes(), raw_content)
            self.assertIn("页数: 2", stdout.getvalue())
            self.assertIn("解析对象总数: 4", stdout.getvalue())
            self.assertIn("text: 1", stdout.getvalue())
            self.assertIn("table: 1", stdout.getvalue())
            self.assertIn("image: 1", stdout.getvalue())
            self.assertIn("<missing>: 1", stdout.getvalue())
            self.assertEqual(
                (root / "images" / "account.jpg").read_bytes(),
                b"jpeg-bytes",
            )
            self.assertEqual(
                (root / "images" / "table.jpg").read_bytes(),
                b"table-jpeg-bytes",
            )
