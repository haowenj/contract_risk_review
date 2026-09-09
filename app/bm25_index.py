from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import jieba
from llama_index.core.schema import NodeWithScore, TextNode


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:[._/%-][a-z0-9]+)*")
_CJK_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")
_INDEX_FILENAME = "index.json"


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese text while retaining useful Latin and numeric terms."""

    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.lower()):
        if _CJK_PATTERN.match(chunk):
            tokens.extend(
                token.strip()
                for token in jieba.lcut_for_search(chunk)
                if token.strip()
            )
        else:
            tokens.append(chunk)
    return tokens


def _node_search_text(node: Any) -> str:
    text = getattr(node, "text", "")
    metadata = getattr(node, "metadata", {}) or {}
    retrieval_context = metadata.get("retrieval_context")
    if isinstance(retrieval_context, str) and retrieval_context.strip():
        return f"{text}\n{retrieval_context}"
    return str(text)


class BM25Index:
    """A small persisted BM25 index over the already-built LlamaIndex Nodes."""

    def __init__(
        self,
        nodes: list[Any],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.nodes = list(nodes)
        self.k1 = k1
        self.b = b
        self._tokenized_documents = [
            tokenize(_node_search_text(node)) for node in self.nodes
        ]
        self._document_frequencies: Counter[str] = Counter()
        for document in self._tokenized_documents:
            self._document_frequencies.update(set(document))
        self._document_lengths = [
            len(document) for document in self._tokenized_documents
        ]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    def retrieve(self, query: str, *, top_k: int = 10) -> list[NodeWithScore]:
        if top_k <= 0 or not self.nodes:
            return []

        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []

        document_count = len(self.nodes)
        scores: list[tuple[float, int]] = []
        for document_index, document in enumerate(self._tokenized_documents):
            term_frequencies = Counter(document)
            document_length = self._document_lengths[document_index]
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = term_frequencies.get(term, 0)
                document_frequency = self._document_frequencies.get(term, 0)
                if not term_frequency or not document_frequency:
                    continue
                idf = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_normalizer = (
                    1.0
                    - self.b
                    + self.b
                    * document_length
                    / self._average_document_length
                    if self._average_document_length
                    else 1.0
                )
                score += (
                    idf
                    * (term_frequency * (self.k1 + 1.0))
                    / (term_frequency + self.k1 * length_normalizer)
                    * query_frequency
                )
            scores.append((score, document_index))

        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            NodeWithScore(node=self.nodes[index], score=float(score))
            for score, index in scores[:top_k]
        ]

    def persist(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "nodes": [node.to_dict() for node in self.nodes],
        }
        target = index_dir / _INDEX_FILENAME
        temporary = index_dir / f".{_INDEX_FILENAME}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Index":
        path = Path(index_dir) / _INDEX_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"persisted BM25 index not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"persisted BM25 index is invalid: {path}") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
            raise ValueError(f"persisted BM25 index must contain nodes: {path}")
        try:
            nodes = [TextNode.from_dict(node) for node in payload["nodes"]]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"persisted BM25 nodes are invalid: {path}") from exc
        return cls(
            nodes,
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )
