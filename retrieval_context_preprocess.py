import json
import logging
import os
import re
import sys
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from mineru_to_nodes import (
    DEBUG_SOURCE_OBJECT_INDICES,
    INPUT_PATH,
    RETRIEVAL_CONTEXT_PATH,
)
from image_searchable_text import image_to_searchable_text
from table_searchable_text import table_to_searchable_text

MAX_RETRIEVAL_CONTEXT_CHARS = 160
CONTEXT_LLM_CONCURRENCY = 5
MAX_CONTEXT_GENERATION_ATTEMPTS = 3
CONTEXT_LLM_MODEL = (
    os.getenv("RETRIEVAL_CONTEXT_LLM_MODEL") or os.environ["LLM_MODEL"]
)
CONTEXT_LLM_TIMEOUT_SECONDS = int(
    os.getenv("RETRIEVAL_CONTEXT_LLM_TIMEOUT_SECONDS", "120")
)
logger = logging.getLogger(__name__)

load_dotenv()

context_llm = ChatOpenAI(
    model=CONTEXT_LLM_MODEL,
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    temperature=0,
    timeout=CONTEXT_LLM_TIMEOUT_SECONDS,
    max_retries=0,
    reasoning_effort="none",
)

ContextGenerator = Callable[[str, list[str]], str | None]
ContextPromptBuilder = Callable[[str, list[str]], str]


def _text_objects(objects: list[dict]) -> list[tuple[int, dict]]:
    return [
        (source_index, obj)
        for source_index, obj in enumerate(objects)
        if obj.get("type") == "text"
        and isinstance(obj.get("text"), str)
        and obj["text"].strip()
    ]


def _table_objects(objects: list[dict]) -> list[tuple[int, dict]]:
    return [
        (source_index, obj)
        for source_index, obj in enumerate(objects)
        if obj.get("type") == "table"
    ]


def _build_section_paths(
    text_objects: list[tuple[int, dict]],
) -> list[list[str]]:
    heading_stack: list[tuple[int, str]] = []
    section_paths: list[list[str]] = []

    for _, obj in text_objects:
        heading_text = obj["text"].strip()
        text_level = obj.get("text_level")

        if heading_text and text_level is not None:
            try:
                level = int(text_level)
            except (TypeError, ValueError):
                level = len(heading_stack) + 1

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))

        section_paths.append([text for _, text in heading_stack])

    return section_paths


def _section_path_before_index(
    objects: list[dict],
    target_index: int,
) -> list[str]:
    heading_stack: list[tuple[int, str]] = []

    for obj in objects[:target_index]:
        if obj.get("type") != "text":
            continue
        heading_text = obj.get("text")
        if not isinstance(heading_text, str) or not heading_text.strip():
            continue
        text_level = obj.get("text_level")
        if text_level is None:
            continue
        try:
            level = int(text_level)
        except (TypeError, ValueError):
            level = len(heading_stack) + 1

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text.strip()))

    return [text for _, text in heading_stack]


def build_context_prompt(chunk_text: str, section_path: list[str]) -> str:
    section_text = " > ".join(section_path) or "未识别到章节标题"

    return f"""你是合同检索预处理器。请为当前 chunk 生成一段简短的 retrieval_context。

目标是补充当前 chunk 在全文中的定位、所属章节或条款，以及只有在原文明确支持时才有助于检索的上下文。
只使用下方章节路径和当前 chunk 明确提供的信息。
不得猜测章节层级、编号含义、期数、先后顺序、因果关系或条款关系。
不要重复当前 chunk 已明确出现的金额、比例、期限、日期、主体或责任，也不要总结、改写或扩写当前 chunk。
如果某项上下文无法确定就省略；如果没有可安全补充的信息，输出空字符串。
只输出 retrieval_context，不要输出 Markdown、引号、前缀或解释。

章节路径（由全文 MinerU text_level 推导）：
{section_text}

当前 chunk：
{chunk_text.strip()}
"""


def build_table_context_prompt(
    searchable_text: str,
    section_path: list[str],
) -> str:
    section_text = " > ".join(section_path) or "未识别到章节标题"

    return f"""你是合同检索预处理器。请为当前 table 生成一段简短的 retrieval_context。

目标是补充表格在全文中的章节定位、表格主题或表格用途。
只使用下方章节路径和当前表格明确提供的信息，不得猜测或补充表格没有出现的事实。
如果没有可安全补充的信息，输出空字符串。
只输出 retrieval_context，不要输出 Markdown、引号、前缀或解释。

章节路径：
{section_text}

当前表格 searchable_text：
{searchable_text.strip()}
"""


def build_image_context_prompt(
    searchable_text: str,
    section_path: list[str],
    nearby_texts: list[str],
) -> str:
    section_text = " > ".join(section_path) or "未识别到章节标题"
    nearby_text = "\n".join(nearby_texts) or "未找到附近正文"

    return f"""你是合同检索预处理器。请为当前合同图片生成一段简短的 retrieval_context。

目标是补充图片在全文中的章节定位、附近正文说明或图片用途。
只使用下方章节路径、附近正文和图片 searchable_text 中明确提供的信息。
不得猜测图片类型、字段值、主体关系或附近正文没有说明的事实。
不要重复图片 searchable_text 已经明确出现的账号、姓名、身份证号码等字段。
如果没有可安全补充的信息，输出空字符串。
只输出 retrieval_context，不要输出 Markdown、引号、前缀或解释。

章节路径：
{section_text}

图片 searchable_text：
{searchable_text.strip()}

附近正文：
{nearby_text}
"""


def _normalize_context(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    context = " ".join(value.split()).strip().strip('"“”')
    if not context:
        return None

    if len(context) > MAX_RETRIEVAL_CONTEXT_CHARS:
        context = context[:MAX_RETRIEVAL_CONTEXT_CHARS].rstrip("，,；;：: ")
    return context or None


def _message_content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts) or None


def _fallback_context(section_path: list[str]) -> str | None:
    if not section_path:
        return None
    return f"文档章节：{' > '.join(section_path)}"


def _generate_one_context(
    chunk_text: str,
    section_path: list[str],
    *,
    source_object_index: int,
    llm: Any | None,
    context_generator: ContextGenerator | None,
    prompt_builder: ContextPromptBuilder = build_context_prompt,
) -> str | None:
    normalized_chunk = " ".join(chunk_text.split()).strip()

    for attempt in range(1, MAX_CONTEXT_GENERATION_ATTEMPTS + 1):
        try:
            if context_generator is not None:
                context = context_generator(chunk_text, section_path)
            else:
                model = context_llm if llm is None else llm
                response = model.invoke(prompt_builder(chunk_text, section_path))
                content = (
                    response
                    if isinstance(response, str)
                    else getattr(response, "content", None)
                )
                context = _message_content_to_text(content)

            context = _normalize_context(context)
            if context == normalized_chunk or (
                len(normalized_chunk) >= 20
                and normalized_chunk in (context or "")
            ):
                context = None
            return context or _fallback_context(section_path)
        except Exception as exc:
            if attempt < MAX_CONTEXT_GENERATION_ATTEMPTS:
                logger.warning(
                    "retrieval_context generation failed for "
                    "source_object_index=%s (attempt %s/%s), retrying: %s",
                    source_object_index,
                    attempt,
                    MAX_CONTEXT_GENERATION_ATTEMPTS,
                    exc,
                )
            else:
                logger.warning(
                    "retrieval_context generation failed for "
                    "source_object_index=%s after %s attempts: %s",
                    source_object_index,
                    MAX_CONTEXT_GENERATION_ATTEMPTS,
                    exc,
                )

    return _fallback_context(section_path)


def generate_contexts(
    objects: list[dict],
    *,
    llm: Any | None = None,
    context_generator: ContextGenerator | None = None,
    concurrency: int = CONTEXT_LLM_CONCURRENCY,
) -> dict[int, str | None]:
    text_objects = _text_objects(objects)
    section_paths = _build_section_paths(text_objects)
    contexts: list[str | None] = []
    if text_objects:
        worker_count = min(max(1, concurrency), len(text_objects))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _generate_one_context,
                    obj["text"],
                    section_paths[text_index],
                    source_object_index=source_index,
                    llm=llm,
                    context_generator=context_generator,
                )
                for text_index, (source_index, obj) in enumerate(text_objects)
            ]
            contexts = [future.result() for future in futures]

    return {
        **{
            source_index: contexts[text_index]
            for text_index, (source_index, _) in enumerate(text_objects)
        },
        **_generate_table_contexts(
            objects,
            llm=llm,
            context_generator=context_generator,
            concurrency=concurrency,
        ),
        **_generate_image_contexts(
            objects,
            llm=llm,
            context_generator=context_generator,
            concurrency=concurrency,
        ),
    }


def _generate_table_contexts(
    objects: list[dict],
    *,
    llm: Any | None,
    context_generator: ContextGenerator | None,
    concurrency: int,
) -> dict[int, str | None]:
    table_objects = _table_objects(objects)
    if not table_objects:
        return {}

    worker_count = min(max(1, concurrency), len(table_objects))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _generate_one_context,
                table_to_searchable_text(obj),
                _section_path_before_index(objects, source_index),
                source_object_index=source_index,
                llm=llm,
                context_generator=context_generator,
                prompt_builder=build_table_context_prompt,
            )
            for source_index, obj in table_objects
        ]
        contexts = [future.result() for future in futures]

    return {
        source_index: contexts[table_index]
        for table_index, (source_index, _) in enumerate(table_objects)
    }


def _nearby_texts(
    objects: list[dict],
    target_index: int,
    *,
    max_each_side: int = 2,
    max_chars: int = 600,
) -> list[str]:
    before = [
        obj["text"].strip()
        for obj in reversed(objects[:target_index])
        if obj.get("type") == "text"
        and isinstance(obj.get("text"), str)
        and obj["text"].strip()
    ][:max_each_side]
    after = [
        obj["text"].strip()
        for obj in objects[target_index + 1 :]
        if obj.get("type") == "text"
        and isinstance(obj.get("text"), str)
        and obj["text"].strip()
    ][:max_each_side]
    ordered = [*reversed(before), *after]
    limited: list[str] = []
    remaining = max_chars
    for text in ordered:
        if remaining <= 0:
            break
        value = text[:remaining]
        limited.append(value)
        remaining -= len(value)
    return limited


def _generate_image_contexts(
    objects: list[dict],
    *,
    llm: Any | None,
    context_generator: ContextGenerator | None,
    concurrency: int,
) -> dict[int, str | None]:
    image_objects: list[tuple[int, dict, str]] = []
    for source_index, obj in enumerate(objects):
        if obj.get("type") != "image":
            continue
        searchable_text = image_to_searchable_text(obj)
        if searchable_text:
            image_objects.append((source_index, obj, searchable_text))

    if not image_objects:
        return {}

    def generate_image_context(
        source_index: int,
        obj: dict,
        searchable_text: str,
    ) -> str | None:
        section_path = _section_path_before_index(objects, source_index)
        nearby = _nearby_texts(objects, source_index)
        chunk_text = searchable_text
        if context_generator is not None:
            # Keep the public context_generator contract unchanged. The image
            # searchable text is the chunk; nearby evidence is supplied to the
            # default LLM prompt below.
            return _generate_one_context(
                chunk_text,
                section_path,
                source_object_index=source_index,
                llm=llm,
                context_generator=context_generator,
            )

        return _generate_one_context(
            chunk_text,
            section_path,
            source_object_index=source_index,
            llm=llm,
            context_generator=None,
            prompt_builder=lambda text, path: build_image_context_prompt(
                text,
                path,
                nearby,
            ),
        )

    worker_count = min(max(1, concurrency), len(image_objects))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(generate_image_context, source_index, obj, text)
            for source_index, obj, text in image_objects
        ]
        contexts = [future.result() for future in futures]

    return {
        source_index: contexts[image_index]
        for image_index, (source_index, _, _) in enumerate(image_objects)
    }


def save_retrieval_contexts(
    contexts: Mapping[int, str | None],
    path: Path = RETRIEVAL_CONTEXT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "source_object_index": source_index,
            "retrieval_context": contexts[source_index],
        }
        for source_index in sorted(contexts)
    ]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_context_debug(
    text_objects: list[tuple[int, dict]],
    contexts: Mapping[int, str | None],
    *,
    source_object_indices: Collection[int] = DEBUG_SOURCE_OBJECT_INDICES,
    file: Any | None = None,
) -> None:
    stream = sys.stdout if file is None else file
    selected = set(source_object_indices)

    for source_index, obj in text_objects:
        if source_index not in selected:
            continue
        print(f"source_object_index: {source_index}", file=stream)
        print("原始 node.text:", file=stream)
        print(obj["text"], file=stream)
        print("retrieval_context:", file=stream)
        print(contexts.get(source_index) or "<empty>", file=stream)


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        objects = json.load(file)

    contexts = generate_contexts(objects)
    save_retrieval_contexts(contexts)
    print_context_debug(_text_objects(objects), contexts)
    print(f"已保存 retrieval_context: {RETRIEVAL_CONTEXT_PATH}")


if __name__ == "__main__":
    main()
