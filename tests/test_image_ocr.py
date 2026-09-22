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

    def test_run_image_ocr_falls_back_silently_to_v1_zip(self):
        import mineru_raw_parse

        structured_content = {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {"type": "text", "content": "户名：乙公司"},
                        {"type": "text", "content": "账号：123456"},
                    ],
                }
            ]
        }
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr(
                "structured_content.json",
                json.dumps(structured_content, ensure_ascii=False),
            )

        job_requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/tasks":
                return httpx.Response(404, json={"detail": "Not Found"})
            if request.method == "POST" and request.url.path == "/v1/uploads":
                return httpx.Response(
                    200,
                    json={
                        "id": "upload-ocr",
                        "upload_url": "/v1/uploads/upload-ocr/content",
                    },
                )
            if request.method == "PUT" and request.url.path.endswith("/content"):
                return httpx.Response(200, json={})
            if request.method == "POST" and request.url.path.endswith("/complete"):
                return httpx.Response(200, json={"file": {"id": "image-input"}})
            if request.method == "POST" and request.url.path == "/v1/parse/jobs":
                job_requests.append(json.loads(request.content))
                return httpx.Response(202, json={"job_id": "ocr-job"})
            if request.method == "GET" and request.url.path == "/v1/parse/jobs/ocr-job":
                return httpx.Response(
                    200,
                    json={
                        "status": "completed",
                        "files": [
                            {
                                "status": "completed",
                                "output_files": {
                                    "zip": {"file_id": "ocr-output", "bytes": 10}
                                },
                            }
                        ],
                    },
                )
            if (
                request.method == "GET"
                and request.url.path == "/v1/files/ocr-output/content"
            ):
                return httpx.Response(
                    200,
                    headers={"content-type": "application/zip"},
                    content=archive_buffer.getvalue(),
                )
            if request.method == "DELETE" and request.url.path.startswith(
                "/v1/files/"
            ):
                return httpx.Response(200, json={})
            return httpx.Response(500)

        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "account.jpg"
            image_path.write_bytes(b"jpeg-bytes")
            client = httpx.Client(transport=httpx.MockTransport(handler))
            with self.assertNoLogs("mineru_raw_parse", level="WARNING"):
                text = mineru_raw_parse.run_image_ocr(
                    image_path,
                    svr_url="http://mineru.test",
                    backend="hybrid-engine",
                    server_url=None,
                    client=client,
                    poll_interval=0,
                )
            client.close()

        self.assertEqual(text, "户名：乙公司\n账号：123456")
        self.assertEqual(len(job_requests), 1)
        self.assertEqual(job_requests[0]["ocr_mode"], "ocr")
        self.assertEqual(job_requests[0]["output_formats"], ["zip"])
