from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx
from dotenv import load_dotenv


DEFAULT_SVR_URL = "http://127.0.0.1:7100"
DEFAULT_BACKEND = "hybrid-engine"
POLL_INTERVAL_SECONDS = 2.0
TASK_TIMEOUT_SECONDS = 30 * 60

BASE_PARSE_FORM = {
    "parse_method": "auto",
    "effort": "medium",
    "formula_enable": "true",
    "table_enable": "true",
    "image_analysis": "false",
    "return_md": "false",
    "return_middle_json": "false",
    "return_model_output": "false",
    "return_content_list": "true",
    "return_images": "true",
    "response_format_zip": "true",
}


def _validate_pdf_path(pdf_path: Path) -> Path:
    resolved = pdf_path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"PDF 文件不存在或不是普通文件：{pdf_path}")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件必须是 PDF：{pdf_path}")
    return resolved


def _response_json(response: httpx.Response, label: str) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"MinerU {label}响应不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"MinerU {label}响应必须是 JSON 对象")
    return payload


def _extract_raw_content_list(archive_bytes: bytes) -> bytes:
    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            matches = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.replace("\\", "/").endswith("_content_list.json")
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "MinerU 结果中应有 1 个 *_content_list.json，"
                    f"实际找到 {len(matches)} 个"
                )
            return archive.read(matches[0])
    except RuntimeError:
        raise
    except (OSError, ValueError, BadZipFile) as exc:
        raise RuntimeError(f"无法读取 MinerU 结果 ZIP：{exc}") from exc


def _print_statistics(content_list: list[object]) -> None:
    page_count = len(
        {
            item["page_idx"]
            for item in content_list
            if isinstance(item, dict) and item.get("page_idx") is not None
        }
    )
    type_counts = Counter(
        item.get("type", "<missing>")
        if isinstance(item, dict)
        else "<non-object>"
        for item in content_list
    )

    print(f"页数: {page_count}")
    print(f"解析对象总数: {len(content_list)}")
    print("type 统计:")
    for item_type in sorted(type_counts, key=str):
        print(f"  {item_type}: {type_counts[item_type]}")


def run_parse(
    pdf_path: Path,
    output_path: Path,
    *,
    svr_url: str,
    backend: str,
    server_url: str | None,
    client: httpx.Client | None = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    pdf_path = _validate_pdf_path(pdf_path)
    if backend not in {"hybrid-engine", "hybrid-http-client"}:
        raise ValueError(f"不支持的 MinerU backend：{backend}")
    if backend == "hybrid-http-client" and not (server_url or "").strip():
        raise ValueError("hybrid-http-client 模式下 server_url 不能为空")

    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0),
        follow_redirects=True,
    )
    try:
        form = {**BASE_PARSE_FORM, "backend": backend}
        if server_url:
            form["server_url"] = server_url.rstrip("/")

        try:
            with pdf_path.open("rb") as pdf_file:
                response = http_client.post(
                    f"{svr_url.rstrip('/')}/tasks",
                    data=form,
                    files={"files": (pdf_path.name, pdf_file, "application/pdf")},
                )
        except (OSError, httpx.HTTPError) as exc:
            raise RuntimeError(f"提交 MinerU 任务失败：{exc}") from exc
        if response.status_code != 202:
            raise RuntimeError(
                f"提交 MinerU 任务失败：HTTP {response.status_code} {response.text}"
            )

        submission = _response_json(response, "任务提交")
        task_id = submission.get("task_id")
        status_url = submission.get("status_url")
        result_url = submission.get("result_url")
        if not all(
            isinstance(value, str) and value
            for value in (task_id, status_url, result_url)
        ):
            raise RuntimeError("MinerU 返回了无效任务响应")

        deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                response = http_client.get(status_url)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"查询 MinerU 任务失败：{exc}") from exc
            if response.status_code != 200:
                raise RuntimeError(
                    f"查询 MinerU 任务失败：HTTP {response.status_code} {response.text}"
                )

            status_payload = _response_json(response, "任务状态")
            status = status_payload.get("status")
            if status in {"pending", "processing"}:
                time.sleep(poll_interval)
                continue
            if status == "completed":
                break
            if status == "failed":
                raise RuntimeError(f"MinerU 任务失败：{status_payload}")
            raise RuntimeError(f"MinerU 返回未知任务状态：{status!r}")
        else:
            raise RuntimeError("等待 MinerU 任务超时")

        try:
            response = http_client.get(result_url)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"下载 MinerU 结果失败：{exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"下载 MinerU 结果失败：HTTP {response.status_code} {response.text}"
            )
        if "application/zip" not in response.headers.get("content-type", "").lower():
            raise RuntimeError("MinerU 结果不是 ZIP")

        raw_content = _extract_raw_content_list(response.content)
        try:
            content_list = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MinerU content list 不是有效 JSON") from exc
        if not isinstance(content_list, list):
            raise RuntimeError("MinerU content list 必须是 JSON 数组")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw_content)
        _print_statistics(content_list)
    finally:
        if owns_client:
            http_client.close()


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="调用 MinerU 并保存原始 content list")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--svr-url")
    parser.add_argument("--backend")
    parser.add_argument("--server-url")
    args = parser.parse_args(argv)

    pdf_path = _validate_pdf_path(args.input_pdf)
    project_dir = Path(__file__).resolve().parent
    output_path = args.output or project_dir / f"{pdf_path.stem}_mineru_raw.json"
    run_parse(
        pdf_path,
        output_path,
        svr_url=args.svr_url or os.environ.get("PDF_TRANS_MINERU_URL", DEFAULT_SVR_URL),
        backend=args.backend
        or os.environ.get("PDF_TRANS_MINERU_BACKEND", DEFAULT_BACKEND),
        server_url=(
            args.server_url
            if args.server_url is not None
            else os.environ.get("PDF_TRANS_MINERU_SERVER_URL")
        ),
    )


if __name__ == "__main__":
    main()
