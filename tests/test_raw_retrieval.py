from types import SimpleNamespace

from app.rag_pipeline import RAGPipeline


def result_for(node_id: str, score: float):
    return SimpleNamespace(
        node=SimpleNamespace(
            node_id=node_id,
            text=node_id,
            metadata={"source_object_index": int(node_id.rsplit("-", 1)[-1])},
        ),
        score=score,
    )


class FakeVectorRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query):
        return self.results


class FakeVectorIndex:
    def __init__(self, results):
        self.retriever = FakeVectorRetriever(results)

    def as_retriever(self, *, similarity_top_k):
        assert similarity_top_k == 10
        return self.retriever


class FakeBM25Index:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, *, top_k):
        assert query == "问题"
        assert top_k == 10
        return self.results


def test_raw_retrieval_supports_vector_bm25_and_hybrid_without_rerank_or_llm():
    vector_results = [result_for("node-1", 0.9), result_for("node-2", 0.8)]
    bm25_results = [result_for("node-2", 2.0), result_for("node-3", 1.5)]
    pipeline = RAGPipeline()

    vector = pipeline.retrieve_raw(
        FakeVectorIndex(vector_results),
        "问题",
        retrieval_mode="vector",
    )
    bm25 = pipeline.retrieve_raw(
        FakeVectorIndex(vector_results),
        "问题",
        retrieval_mode="bm25",
        bm25_index=FakeBM25Index(bm25_results),
    )
    hybrid = pipeline.retrieve_raw(
        FakeVectorIndex(vector_results),
        "问题",
        retrieval_mode="hybrid",
        bm25_index=FakeBM25Index(bm25_results),
    )

    assert [item.node.node_id for item in vector["results"]] == ["node-1", "node-2"]
    assert [item.node.node_id for item in bm25["results"]] == ["node-2", "node-3"]
    assert [item.node.node_id for item in hybrid["results"]] == [
        "node-1",
        "node-2",
        "node-3",
    ]
    assert hybrid["results"][1].score == 0.8
    assert hybrid["retrieval_mode"] == "hybrid"
