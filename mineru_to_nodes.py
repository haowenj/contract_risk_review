import json
import os
import sys
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from image_searchable_text import image_to_searchable_text
from table_searchable_text import table_to_searchable_text

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(
    os.getenv("RAG_INPUT_PATH", str(PROJECT_DIR / "data" / "merged_content_list.json"))
).expanduser()
RETRIEVAL_CONTEXT_PATH = Path(
    os.getenv(
        "RAG_RETRIEVAL_CONTEXT_PATH",
        str(PROJECT_DIR / "data" / "retrieval_context.json"),
    )
).expanduser()
DEBUG_SOURCE_OBJECT_INDICES = {111, 112, 113, 114}

load_dotenv()

# DashScope text-embedding-v4 accepts at most 10 input.contents per request.
EMBEDDING_BATCH_SIZE = 10

embedding_model = OpenAIEmbedding(
    model_name=os.environ["LLM_EMBEDDING_MODEL"],
    api_key=os.environ["LLM_API_KEY"],
    api_base=os.environ["LLM_BASE_URL"],
    embed_batch_size=EMBEDDING_BATCH_SIZE,
)


def load_retrieval_contexts(
    path: Path = RETRIEVAL_CONTEXT_PATH,
) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(
            "Retrieval context file not found: "
            f"{path}. Run retrieval_context_preprocess.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Retrieval context file must contain a JSON list")

    contexts: dict[int, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Each retrieval context entry must be an object")

        source_index = entry.get("source_object_index")
        context = entry.get("retrieval_context")
        if type(source_index) is not int:
            raise ValueError("source_object_index must be an integer")
        if context is not None and not isinstance(context, str):
            raise ValueError("retrieval_context must be a string or null")
        if isinstance(context, str) and context.strip():
            contexts[source_index] = context

    return contexts


def build_nodes(
    objects: list[dict],
    *,
    retrieval_contexts: Mapping[int, str] | None = None,
) -> list[TextNode]:
    text_objects = []

    for source_index, obj in enumerate(objects):
        if obj.get("type") != "text":
            continue

        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        text_objects.append((source_index, obj))

    nodes = []

    for source_index, obj in text_objects:
        # Keep MinerU's original evidence text unchanged.
        text = obj["text"]
        retrieval_context = (
            retrieval_contexts.get(source_index)
            if retrieval_contexts is not None
            else None
        )

        metadata = {
            "node_type": "text",
            "retrieval_context": retrieval_context,
            "text_level": obj.get("text_level"),
            "page_idx": obj.get("page_idx"),
            "bbox": obj.get("bbox"),
            "start_page_idx": obj.get("start_page_idx"),
            "end_page_idx": obj.get("end_page_idx"),
            "source_page_indices": obj.get("source_page_indices"),
            "source_bboxes": obj.get("source_bboxes"),
            "merged_cross_page": obj.get("merged_cross_page", False),
            "source_object_index": source_index,
        }

        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None and value != ""
        }

        excluded_embed_metadata_keys = [
            key
            for key in metadata
            if key != "retrieval_context"
        ]

        nodes.append(
            TextNode(
                id_=f"test_hetong_{source_index}",
                text=text,
                metadata=metadata,
                excluded_embed_metadata_keys=excluded_embed_metadata_keys,
            )
        )

    for source_index, obj in enumerate(objects):
        if obj.get("type") != "table":
            continue

        retrieval_context = (
            retrieval_contexts.get(source_index)
            if retrieval_contexts is not None
            else None
        )
        metadata = {
            "node_type": "table",
            "retrieval_context": retrieval_context,
            "source_object_index": source_index,
            "page_idx": obj.get("page_idx"),
            "bbox": obj.get("bbox"),
            "table_body": obj.get("table_body"),
            "table_caption": obj.get("table_caption"),
            "table_footnote": obj.get("table_footnote"),
            "img_path": obj.get("img_path"),
        }
        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
        }
        excluded_embed_metadata_keys = [
            key
            for key in metadata
            if key != "retrieval_context"
        ]

        nodes.append(
            TextNode(
                id_=f"test_hetong_{source_index}",
                text=table_to_searchable_text(obj),
                metadata=metadata,
                excluded_embed_metadata_keys=excluded_embed_metadata_keys,
            )
        )

    for source_index, obj in enumerate(objects):
        if obj.get("type") != "image":
            continue

        searchable_text = image_to_searchable_text(obj)
        if not searchable_text:
            continue

        retrieval_context = (
            retrieval_contexts.get(source_index)
            if retrieval_contexts is not None
            else None
        )
        metadata = {
            "node_type": "image",
            "retrieval_context": retrieval_context,
            "source_object_index": source_index,
            "page_idx": obj.get("page_idx"),
            "bbox": obj.get("bbox"),
            "img_path": obj.get("img_path"),
            "image_type": obj.get("image_type"),
            "structured_data": obj.get("structured_data"),
            "ocr_text": obj.get("ocr_text") or "",
            "ocr_status": obj.get("ocr_status"),
            "verification_status": obj.get("verification_status"),
            "verification_details": obj.get("verification_details", {}),
            "image_processing_status": obj.get("image_processing_status"),
            "image_schema_version": obj.get("image_schema_version"),
            "image_model": obj.get("image_model"),
            "image_caption": obj.get("image_caption"),
            "image_footnote": obj.get("image_footnote"),
        }
        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
        }
        excluded_embed_metadata_keys = [
            key
            for key in metadata
            if key != "retrieval_context"
        ]

        nodes.append(
            TextNode(
                id_=f"test_hetong_{source_index}",
                text=searchable_text,
                metadata=metadata,
                excluded_embed_metadata_keys=excluded_embed_metadata_keys,
            )
        )

    return nodes


def print_embedding_debug(
    nodes: list[TextNode],
    *,
    source_object_indices: Collection[int] | None = None,
    file: Any | None = None,
) -> None:
    stream = sys.stdout if file is None else file
    selected = (
        set(source_object_indices)
        if source_object_indices is not None
        else None
    )

    for node in nodes:
        source_index = node.metadata.get("source_object_index")
        if selected is not None and source_index not in selected:
            continue

        print(f"=== Node {node.node_id} ===", file=stream)
        print("source_object_index:", source_index, file=stream)
        print("原始 node.text:", file=stream)
        print(node.text, file=stream)
        print("生成的 retrieval_context:", file=stream)
        print(node.metadata.get("retrieval_context", "<empty>"), file=stream)
        print("Embedding 实际使用的内容:", file=stream)
        print(node.get_content(metadata_mode=MetadataMode.EMBED), file=stream)


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        objects = json.load(file)

    retrieval_contexts = load_retrieval_contexts()
    nodes = build_nodes(objects, retrieval_contexts=retrieval_contexts)

    print_embedding_debug(
        nodes,
        source_object_indices=DEBUG_SOURCE_OBJECT_INDICES,
    )

    index = VectorStoreIndex(nodes, embed_model=embedding_model)
    retriever = index.as_retriever(similarity_top_k=10)

    results = retriever.retrieve("合同的付款方式是什么？")
    for result in results:
        print(result.score)
        print(result.text)
        print(result.metadata)


if __name__ == "__main__":
    main()
