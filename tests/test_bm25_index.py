from pathlib import Path
from tempfile import TemporaryDirectory

from app.bm25_index import BM25Index, tokenize
from llama_index.core.schema import TextNode


def node(node_id: str, text: str, retrieval_context: str):
    return TextNode(
        id_=node_id,
        text=text,
        metadata={
            "source_object_index": int(node_id.rsplit("-", 1)[-1]),
            "retrieval_context": retrieval_context,
        },
    )


def test_chinese_tokenizer_emits_searchable_terms():
    terms = tokenize("本合同约定首期付款方式为银行转账")

    assert "付款" in terms
    assert "方式" in terms
    assert "银行" in terms


def test_bm25_returns_relevant_chinese_node_with_original_score():
    nodes = [
        node("node-1", "本合同约定首期付款比例为30%。", "付款条款"),
        node("node-2", "乙方应当按时交付货物。", "交付条款"),
    ]
    index = BM25Index(nodes)

    results = index.retrieve("首期付款比例", top_k=2)

    assert [result.node.node_id for result in results] == ["node-1", "node-2"]
    assert results[0].node.text == nodes[0].text
    assert results[0].node.metadata["retrieval_context"] == "付款条款"
    assert results[0].score > results[1].score


def test_bm25_persistence_preserves_node_id_text_and_retrieval_context():
    original_nodes = [
        node("node-7", "付款方式为银行转账。", "付款章节"),
    ]
    index = BM25Index(original_nodes)

    with TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir) / "bm25_index"
        index.persist(index_dir)
        loaded = BM25Index.load(index_dir)

    loaded_node = loaded.nodes[0]
    assert loaded_node.node_id == "node-7"
    assert loaded_node.text == "付款方式为银行转账。"
    assert loaded_node.metadata["retrieval_context"] == "付款章节"
    assert loaded.retrieve("付款方式", top_k=1)[0].node.node_id == "node-7"
