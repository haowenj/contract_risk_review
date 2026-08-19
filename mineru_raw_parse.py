from __future__ import annotations

import argparse
import io
import json
import logging
import mimetypes
import os
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import httpx
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

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

OCR_PARSE_FORM = {
    "parse_method": "ocr",
    "effort": "medium",
    "formula_enable": "false",
    "table_enable": "false",
    "image_analysis": "false",
    "return_md": "false",
    "return_middle_json": "false",
    "return_model_output": "false",
    "return_content_list": "true",
    "return_images": "false",
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


def _normalize_zip_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError(f"MinerU ZIP 包含不安全路径：{name}")
    return str(path)


def _parse_content_list(raw_content: bytes) -> list[object]:
    try:
        content_list = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MinerU content list 不是有效 JSON") from exc
    if not isinstance(content_list, list):
        raise RuntimeError("MinerU content list 必须是 JSON 数组")
    return content_list


def _read_content_list_archive(
    archive_bytes: bytes,
) -> tuple[str, bytes, list[object]]:
    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            matches = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and _normalize_zip_name(info.filename).endswith(
                    "_content_list.json"
                )
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "MinerU 结果中应有 1 个 *_content_list.json，"
                    f"实际找到 {len(matches)} 个"
                )
            content_member = _normalize_zip_name(matches[0].filename)
            raw_content = archive.read(matches[0])
            return content_member, raw_content, _parse_content_list(raw_content)
    except RuntimeError:
        raise
    except (OSError, ValueError, BadZipFile) as exc:
        raise RuntimeError(f"无法读取 MinerU 结果 ZIP：{exc}") from exc


def _extract_raw_content_list(archive_bytes: bytes) -> bytes:
    """Backward-compatible JSON-only ZIP reader used by existing callers."""
    _, raw_content, _ = _read_content_list_archive(archive_bytes)
    return raw_content


def _safe_image_reference(img_path: str) -> str:
    if not isinstance(img_path, str) or not img_path.strip():
        raise RuntimeError("MinerU image 的 img_path 不能为空")
    normalized = img_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError(f"MinerU image 的 img_path 不安全：{img_path}")
    return str(path)


def _write_referenced_images(
    archive_bytes: bytes,
    *,
    content_member: str,
    content_list: list[object],
    output_dir: Path,
) -> None:
    image_paths: set[str] = set()
    for item in content_list:
        if not (
            isinstance(item, dict)
            and item.get("type") in {"image", "table"}
            and isinstance(item.get("img_path"), str)
        ):
            continue
        try:
            image_paths.add(_safe_image_reference(item["img_path"]))
        except RuntimeError as exc:
            logger.warning("skip unsafe MinerU image reference: %s", exc)
    if not image_paths:
        return

    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            members: dict[str, list[str]] = {}
            for info in archive.infolist():
                if info.is_dir():
                    continue
                normalized = _normalize_zip_name(info.filename)
                members.setdefault(normalized, []).append(info.filename)

            content_parent = PurePosixPath(content_member).parent
            resolved_output_dir = output_dir.resolve()
            for image_path in sorted(image_paths):
                member_path = str(content_parent / image_path)
                if len(members.get(member_path, [])) != 1:
                    logger.warning(
                        "MinerU result has no unique image member for img_path=%s",
                        image_path,
                    )
                    continue

                output_path = output_dir / Path(image_path)
                resolved_output_path = output_path.resolve()
                if not resolved_output_path.is_relative_to(resolved_output_dir):
                    logger.warning(
                        "skip image outside contract directory: %s",
                        image_path,
                    )
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(archive.read(members[member_path][0]))
    except RuntimeError:
        raise
    except (OSError, ValueError, BadZipFile) as exc:
        raise RuntimeError(f"无法提取 MinerU 图片：{exc}") from exc


def _run_task(
    input_path: Path,
    *,
    form: Mapping[str, str],
    svr_url: str,
    backend: str,
    server_url: str | None,
    client: httpx.Client | None,
    poll_interval: float,
) -> bytes:
    if backend not in {
        "pipeline",
        "vlm-engine",
        "hybrid-engine",
        "vlm-http-client",
        "hybrid-http-client",
    }:
        raise ValueError(f"不支持的 MinerU backend：{backend}")
    if backend in {"hybrid-http-client", "vlm-http-client"} and not (
        server_url or ""
    ).strip():
        raise ValueError(f"{backend} 模式下 server_url 不能为空")

    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0),
        follow_redirects=True,
    )
    try:
        request_form = {**form, "backend": backend}
        if server_url:
            request_form["server_url"] = server_url.rstrip("/")
        content_type = mimetypes.guess_type(input_path.name)[0] or (
            "application/octet-stream"
        )

        try:
            with input_path.open("rb") as input_file:
                response = http_client.post(
                    f"{svr_url.rstrip('/')}/tasks",
                    data=request_form,
                    files={
                        "files": (input_path.name, input_file, content_type)
                    },
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
        return response.content
    finally:
        if owns_client:
            http_client.close()


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
    archive_bytes = _run_task(
        pdf_path,
        form=BASE_PARSE_FORM,
        svr_url=svr_url,
        backend=backend,
        server_url=server_url,
        client=client,
        poll_interval=poll_interval,
    )
    content_member, raw_content, content_list = _read_content_list_archive(
        archive_bytes
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw_content)
    _write_referenced_images(
        archive_bytes,
        content_member=content_member,
        content_list=content_list,
        output_dir=output_path.parent,
    )
    _print_statistics(content_list)


def _validate_image_path(image_path: Path) -> Path:
    resolved = image_path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"图片不存在或不是普通文件：{image_path}")
    return resolved


def _content_list_text(content_list: list[object]) -> str:
    texts: list[str] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        value = item.get("text")
        if not isinstance(value, str) or not value.strip():
            value = item.get("content")
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return "\n".join(texts)


def run_image_ocr(
    image_path: Path,
    *,
    svr_url: str,
    backend: str,
    server_url: str | None,
    client: httpx.Client | None = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> str:
    image_path = _validate_image_path(image_path)
    archive_bytes = _run_task(
        image_path,
        form=OCR_PARSE_FORM,
        svr_url=svr_url,
        backend=backend,
        server_url=server_url,
        client=client,
        poll_interval=poll_interval,
    )
    _, _, content_list = _read_content_list_archive(archive_bytes)
    return _content_list_text(content_list)


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
