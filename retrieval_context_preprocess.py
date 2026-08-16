import json
import logging
import os
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

MAX_RETRIEVAL_CONTEXT_CHARS = 160
CONTEXT_LLM_CONCURRENCY = 5
CONTEXT_LLM_MODEL = os.getenv("RETRIEVAL_CONTEXT_LLM_MODEL", "qwen3.7-plus")
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
    extra_body={"enable_thinking": False},
)

ContextGenerator = Callable[[str, list[str]], str | None]


def _text_objects(objects: list[dict]) -> list[tuple[int, dict]]:
    return [
        (source_index, obj)
        for source_index, obj in enumerate(objects)
        if obj.get("type") == "text"
        and isinstance(obj.get("text"), str)
        and obj["text"].strip()
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
) -> str | None:
    try:
        if context_generator is not None:
            context = context_generator(chunk_text, section_path)
        else:
            model = context_llm if llm is None else llm
            response = model.invoke(build_context_prompt(chunk_text, section_path))
            content = (
                response
                if isinstance(response, str)
                else getattr(response, "content", None)
            )
            context = _message_content_to_text(content)

        context = _normalize_context(context)
        normalized_chunk = " ".join(chunk_text.split()).strip()
        if context == normalized_chunk or (
            len(normalized_chunk) >= 20
            and normalized_chunk in (context or "")
        ):
            context = None
    except Exception as exc:
        logger.warning(
            "retrieval_context generation failed for source_object_index=%s: %s",
            source_object_index,
            exc,
        )
        context = None

    return context or _fallback_context(section_path)


def generate_contexts(
    objects: list[dict],
    *,
    llm: Any | None = None,
    context_generator: ContextGenerator | None = None,
    concurrency: int = CONTEXT_LLM_CONCURRENCY,
) -> dict[int, str | None]:
    text_objects = _text_objects(objects)
    section_paths = _build_section_paths(text_objects)
    if not text_objects:
        return {}

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
        source_index: contexts[text_index]
        for text_index, (source_index, _) in enumerate(text_objects)
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
