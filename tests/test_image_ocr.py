import io
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import httpx


class ImageOCRTest(TestCase):
    def test_mineru_image_ocr_service_delegates_to_injected_runner(self):
        from app.image_ocr import MinerUImageOCRService

        calls = []

        def runner(image_path, *, svr_url, backend, server_url):
            calls.append((image_path, svr_url, backend, server_url))
            return "OCR text"

        image_path = Path("/tmp/test-account.jpg")
        service = MinerUImageOCRService(
            svr_url="http://mineru.test",
            backend="hybrid-engine",
            server_url=None,
            runner=runner,
        )

        self.assertEqual(service.extract_text(image_path), "OCR text")
        self.assertEqual(
            calls,
            [(image_path, "http://mineru.test", "hybrid-engine", None)],
        )

    def test_run_image_ocr_uses_ocr_mode_and_returns_text_in_order(self):
        import mineru_raw_parse

        raw_content = json.dumps(
            [
                {"type": "text", "text": "户名：甲公司"},
                {"type": "text", "text": "账号：110914414810101"},
            ],
            ensure_ascii=False,
        ).encode("utf-8")
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("ocr_content_list.json", raw_content)

        status_calls = 0
        request_bodies: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal status_calls
            if request.method == "POST" and request.url.path == "/tasks":
                request_bodies.append(request.read())
                return httpx.Response(
                    202,
                    json={
                        "task_id": "ocr-task",
                        "status_url": "http://mineru.test/status/ocr-task",
                        "result_url": "http://mineru.test/result/ocr-task",
                    },
                )
            if request.url.path == "/status/ocr-task":
                status_calls += 1
                return httpx.Response(
                    200,
                    json={"status": "completed" if status_calls > 1 else "pending"},
                )
            if request.url.path == "/result/ocr-task":
                return httpx.Response(
                    200,
                    headers={"content-type": "application/zip"},
                    content=archive_buffer.getvalue(),
                )
            return httpx.Response(404)

        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "account.jpg"
            image_path.write_bytes(b"jpeg-bytes")
            client = httpx.Client(transport=httpx.MockTransport(handler))
            text = mineru_raw_parse.run_image_ocr(
                image_path,
                svr_url="http://mineru.test",
                backend="hybrid-engine",
                server_url=None,
                client=client,
                poll_interval=0,
            )
            client.close()

        self.assertEqual(text, "户名：甲公司\n账号：110914414810101")
        self.assertEqual(len(request_bodies), 1)
        self.assertIn(b'name="parse_method"', request_bodies[0])
        self.assertIn(b"ocr", request_bodies[0])
        self.assertIn(b'name="return_images"', request_bodies[0])
        self.assertIn(b"false", request_bodies[0])
