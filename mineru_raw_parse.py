from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import mimetypes
import os
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, is_zipfile

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


def _v1_response_json(
    response: httpx.Response,
    label: str,
) -> dict[str, object]:
    if response.status_code not in {200, 202}:
        raise RuntimeError(
            f"MinerU v1 {label}失败：HTTP {response.status_code}"
        )
    return _response_json(response, f"v1 {label}")


class _LegacyEndpointUnavailable(RuntimeError):
    """Signal that this server exposes MinerU v1 instead of /tasks."""


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


def _structured_annotation_texts(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        content.strip()
        for annotation in value
        if isinstance(annotation, dict)
        and isinstance((content := annotation.get("content")), str)
        and content.strip()
    ]


def _structured_content_list(raw_content: bytes) -> list[object]:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MinerU structured content 不是有效 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        raise RuntimeError("MinerU structured content 缺少 pages")

    content_list: list[object] = []
    for fallback_page_idx, page in enumerate(payload["pages"]):
        if not isinstance(page, dict):
            continue
        page_idx = page.get("page_idx")
        if not isinstance(page_idx, int):
            page_idx = fallback_page_idx
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            content = block.get("content")
            item: dict[str, object] = {"page_idx": page_idx}
            bbox = block.get("bbox")
            if isinstance(bbox, list):
                item["bbox"] = bbox

            if block_type == "table":
                item["type"] = "table"
                if isinstance(content, str) and content.strip():
                    item["table_body"] = content.strip()
                item["table_caption"] = _structured_annotation_texts(
                    block.get("captions")
                )
                item["table_footnote"] = _structured_annotation_texts(
                    block.get("footnotes")
                )
            elif block_type in {"image", "chart"}:
                item["type"] = "image"
                item["image_caption"] = _structured_annotation_texts(
                    block.get("captions")
                )
                item["image_footnote"] = _structured_annotation_texts(
                    block.get("footnotes")
                )
            else:
                item["type"] = (
                    block_type
                    if block_type in {"header", "footer", "page_number"}
                    else "text"
                )
                if isinstance(content, str) and content.strip():
                    item["text"] = content.strip()
                level = block.get("level")
                if block_type == "paragraph_title" and isinstance(level, int):
                    item["text_level"] = level

            image_source = block.get("image_source")
            if isinstance(image_source, str) and image_source.strip():
                item["img_path"] = image_source.strip()
            content_list.append(item)
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
            if len(matches) == 1:
                content_member = _normalize_zip_name(matches[0].filename)
                raw_content = archive.read(matches[0])
                return content_member, raw_content, _parse_content_list(raw_content)
            if len(matches) > 1:
                raise RuntimeError(
                    "MinerU 结果中应有 1 个 *_content_list.json，"
                    f"实际找到 {len(matches)} 个"
                )

            structured_matches = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and PurePosixPath(_normalize_zip_name(info.filename)).name
                == "structured_content.json"
            ]
            if len(structured_matches) != 1:
                raise RuntimeError(
                    "MinerU 结果中既无 content list，也无唯一的 "
                    "structured_content.json"
                )
            content_member = _normalize_zip_name(
                structured_matches[0].filename
            )
            content_list = _structured_content_list(
                archive.read(structured_matches[0])
            )
            raw_content = json.dumps(
                content_list,
                ensure_ascii=False,
            ).encode("utf-8")
            return content_member, raw_content, content_list
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


def _run_legacy_task(
    input_path: Path,
    *,
    form: Mapping[str, str],
    svr_url: str,
    backend: str,
    server_url: str | None,
    client: httpx.Client,
    poll_interval: float,
) -> bytes:
    request_form = {**form, "backend": backend}
    if server_url:
        request_form["server_url"] = server_url.rstrip("/")
    content_type = mimetypes.guess_type(input_path.name)[0] or (
        "application/octet-stream"
    )

    try:
        with input_path.open("rb") as input_file:
            response = client.post(
                f"{svr_url.rstrip('/')}/tasks",
                data=request_form,
                files={"files": (input_path.name, input_file, content_type)},
            )
    except (OSError, httpx.HTTPError) as exc:
        raise RuntimeError(f"提交 MinerU 任务失败：{exc}") from exc
    if response.status_code in {404, 405}:
        raise _LegacyEndpointUnavailable
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
            response = client.get(status_url)
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
        response = client.get(result_url)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"下载 MinerU 结果失败：{exc}") from exc
    if response.status_code != 200:
        raise RuntimeError(
            f"下载 MinerU 结果失败：HTTP {response.status_code} {response.text}"
        )
    if "application/zip" not in response.headers.get("content-type", "").lower():
        raise RuntimeError("MinerU 结果不是 ZIP")
    return response.content


def _run_v1_task(
    input_path: Path,
    *,
    form: Mapping[str, str],
    svr_url: str,
    backend: str,
    client: httpx.Client,
    poll_interval: float,
) -> bytes:
    base_url = svr_url.rstrip("/")
    input_file_id: str | None = None
    output_file_id: str | None = None
    remove_input_file = False
    try:
        try:
            content = input_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"读取 MinerU 输入文件失败：{exc}") from exc
        digest = hashlib.sha256(content).hexdigest()
        content_type = mimetypes.guess_type(input_path.name)[0] or (
            "application/octet-stream"
        )
        upload = _v1_response_json(
            client.post(
                f"{base_url}/v1/uploads",
                json={
                    "filename": input_path.name,
                    "bytes": len(content),
                    "mime_type": content_type,
                    "purpose": "parse",
                    "sha256sum": digest,
                },
            ),
            "创建上传",
        )
        upload_id = upload.get("id")
        if not isinstance(upload_id, str) or not upload_id:
            raise RuntimeError("MinerU v1 返回了无效 upload id")

        file_payload = upload.get("file")
        if upload.get("status") != "completed" or not isinstance(
            file_payload, dict
        ):
            upload_url = upload.get("upload_url")
            if not isinstance(upload_url, str) or not upload_url:
                upload_url = f"{base_url}/v1/uploads/{upload_id}/content"
            elif upload_url.startswith("/"):
                upload_url = f"{base_url}{upload_url}"
            headers = upload.get("upload_headers")
            if not isinstance(headers, dict):
                headers = {"content-type": "application/octet-stream"}
            upload_response = client.put(
                upload_url,
                headers={str(key): str(value) for key, value in headers.items()},
                content=content,
            )
            if upload_response.status_code != 200:
                raise RuntimeError(
                    "MinerU v1 上传内容失败："
                    f"HTTP {upload_response.status_code}"
                )
            completed = _v1_response_json(
                client.post(
                    f"{base_url}/v1/uploads/{upload_id}/complete",
                    json={"sha256sum": digest},
                ),
                "完成上传",
            )
            file_payload = completed.get("file")
            remove_input_file = True

        input_file_id = (
            file_payload.get("id") if isinstance(file_payload, dict) else None
        )
        if not isinstance(input_file_id, str) or not input_file_id:
            raise RuntimeError("MinerU v1 返回了无效 input file id")

        ocr_mode = "ocr" if form.get("parse_method") == "ocr" else "auto"
        tier = "basic" if backend == "pipeline" else "standard"
        job = _v1_response_json(
            client.post(
                f"{base_url}/v1/parse/jobs",
                json={
                    "files": [
                        {
                            "source": {
                                "type": "file_id",
                                "file_id": input_file_id,
                            }
                        }
                    ],
                    "tier": tier,
                    "ocr_mode": ocr_mode,
                    "output_formats": ["zip"],
                },
            ),
            "创建解析任务",
        )
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("MinerU v1 返回了无效 job id")

        deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            job = _v1_response_json(
                client.get(f"{base_url}/v1/parse/jobs/{job_id}"),
                "查询解析任务",
            )
            status = job.get("status")
            if status in {"queued", "running"}:
                time.sleep(poll_interval)
                continue
            if status in {"completed", "partial"}:
                break
            raise RuntimeError(f"MinerU v1 解析任务未完成：{status}")
        else:
            raise RuntimeError("等待 MinerU v1 解析任务超时")

        files = job.get("files")
        first_file = files[0] if isinstance(files, list) and files else None
        output_files = (
            first_file.get("output_files")
            if isinstance(first_file, dict)
            else None
        )
        zip_ref = output_files.get("zip") if isinstance(output_files, dict) else None
        output_file_id = zip_ref.get("file_id") if isinstance(zip_ref, dict) else None
        if not isinstance(output_file_id, str) or not output_file_id:
            raise RuntimeError("MinerU v1 结果缺少 ZIP")

        response = client.get(f"{base_url}/v1/files/{output_file_id}/content")
        if response.status_code != 200:
            raise RuntimeError(
                "下载 MinerU v1 结果失败："
                f"HTTP {response.status_code}"
            )
        if not is_zipfile(io.BytesIO(response.content)):
            raise RuntimeError("MinerU v1 结果不是 ZIP")
        return response.content
    except httpx.HTTPError as exc:
        raise RuntimeError(f"调用 MinerU v1 失败：{exc}") from exc
    finally:
        cleanup_file_ids = [output_file_id]
        if remove_input_file:
            cleanup_file_ids.append(input_file_id)
        for file_id in cleanup_file_ids:
            if not file_id:
                continue
            try:
                client.delete(f"{base_url}/v1/files/{file_id}")
            except httpx.HTTPError:
                logger.warning("Could not remove temporary MinerU file")


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
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0),
        follow_redirects=True,
    )
    try:
        try:
            return _run_legacy_task(
                input_path,
                form=form,
                svr_url=svr_url,
                backend=backend,
                server_url=server_url,
                client=http_client,
                poll_interval=poll_interval,
            )
        except _LegacyEndpointUnavailable:
            return _run_v1_task(
                input_path,
                form=form,
                svr_url=svr_url,
                backend=backend,
                client=http_client,
                poll_interval=poll_interval,
            )
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
