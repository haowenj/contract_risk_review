# Production BM25 Hybrid Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已持久化的 BM25 索引接入正式合同审查，使向量与 BM25 分别召回、按 `node_id` 去重后进入现有 rerank/Evidence/风险判断链路，并让检索不足直接收敛为现有 `insufficient` 结果。

**Architecture:** 保持 `ContractProcessor` 的解析和双索引构建不变，仅在 `ContractService.search_contract` 同时加载两个索引，并让 `RAGPipeline.retrieve_evidence` 在已有向量召回入口中增加 BM25 候选合并。审查图删除查询改写、第二次检索和关键词全文兜底的正式路由；rerank、Evidence Selector、Evidence 序列化及后续审查输出继续复用。

**Tech Stack:** Python 3.12, FastAPI, LlamaIndex, LangGraph, Pydantic, jieba BM25, pytest, persisted JSON/vector indexes.

**Spec:** `docs/superpowers/specs/2026-09-09-production-bm25-hybrid-review-design.md`

## Global Constraints

- 解析、清洗、跨页合并、`retrieval_context` 和 Node 构建逻辑保持不变。
- 向量索引成功后，才基于同一批 Node 构建 BM25；查询时禁止临时构建 BM25。
- 向量与 BM25 原始分数不做加权、归一化或直接混合排序。
- 混合候选按 `node_id` 去重后统一进入现有 rerank。
- 正式审查不再调用查询改写 LLM、关键词全文扫描或 `absence_result` 路由。
- 保留历史 `absence_check` 结果的模型、持久化数据和页面展示兼容性。
- 保留现有审查结果格式、审查项逻辑和前端展示。
- 每次正式混合检索日志至少包含向量数、BM25 数、去重候选数、rerank 数、最终 Evidence Node ID/数量和各阶段耗时。

---

### Task 1: Add failing tests for production hybrid retrieval

**Files:**
- Modify: `tests/test_rag_pipeline.py`
- Create: `tests/test_contract_service_retrieval.py`

**Interfaces:**
- Consumes: existing `RAGPipeline.retrieve_evidence`, `ContractService.search_contract`, fake result helpers from existing tests.
- Produces: executable expectations for `bm25_index=` support, candidate deduplication, raw-score preservation, rerank input, diagnostics logging, and persisted BM25 loading.

- [x] **Step 1: Write the failing pipeline test**

Add a test with two vector results (`node-1`, `node-2`) and two BM25 results (`node-2`, `node-3`). Use a fake reranker that records its input and a selector that selects all three source indices. Assert:

```python
result = RAGPipeline().retrieve_evidence(
    vector_index,
    "付款期限",
    bm25_index=bm25_index,
    reranker=reranker,
    selector_llm=selector,
    fallback_on_empty_selection=False,
)

assert reranker.node_ids == ["node-1", "node-2", "node-3"]
assert result["retrieval_diagnostics"]["vector_count"] == 2
assert result["retrieval_diagnostics"]["bm25_count"] == 2
assert result["retrieval_diagnostics"]["merged_candidate_count"] == 3
assert result["retrieval_diagnostics"]["reranked_count"] == 3
assert result["selected_nodes"][1].node.node_id == "node-2"
assert result["selected_nodes"][2].node.metadata["retrieval_score"] == 1.5
```

The fake BM25-only `node-3` must retain its BM25 score as `retrieval_score`; the duplicate `node-2` must appear once and no vector/BM25 score arithmetic may be asserted or required.

- [x] **Step 2: Run the focused pipeline test and verify the expected red failure**

Run: `uv run --with pytest python -m pytest tests/test_rag_pipeline.py -k hybrid -q`

Expected: FAIL because `retrieve_evidence` does not yet accept `bm25_index` or return hybrid diagnostics.

- [x] **Step 3: Write the failing ContractService loading test**

Create a focused test with a ready contract, a fake `IndexManager` exposing distinct vector and BM25 objects, and a fake RAG pipeline recording arguments. Assert:

```python
service.search_contract("c1", "付款期限")

assert manager.get_calls == [contract]
assert manager.get_bm25_calls == [contract]
assert pipeline.calls[0]["index"] is vector_index
assert pipeline.calls[0]["bm25_index"] is bm25_index
```

Also add a separate test where `get_bm25` raises `FileNotFoundError`; the public call must raise a clear error mentioning `BM25` and `重新解析`, rather than returning vector-only evidence.

- [x] **Step 4: Run the focused service tests and verify the expected red failure**

Run: `uv run --with pytest python -m pytest tests/test_contract_service_retrieval.py -q`

Expected: FAIL because `ContractService.search_contract` currently loads only the vector index and never passes a BM25 index.

### Task 2: Implement persisted hybrid recall and diagnostics

**Files:**
- Modify: `app/rag_pipeline.py:15-105`
- Modify: `app/service.py:99-138`
- Modify: `tests/test_rag_pipeline.py`
- Modify: `tests/test_contract_service_retrieval.py`

**Interfaces:**
- Consumes: `IndexManager.get`, `IndexManager.get_bm25`, `BM25Index.retrieve`, `retrieval_evaluation._record_vector_scores`, `JinaReranker.postprocess_nodes`, existing Evidence selector.
- Produces: `RAGPipeline.retrieve_evidence(..., bm25_index=None)` with hybrid diagnostics, and `ContractService.search_contract` that always loads both persisted indexes for formal search.

- [x] **Step 1: Add the minimal hybrid retrieval implementation**

In `RAGPipeline.retrieve_evidence`, keep the current vector-only behavior when `bm25_index is None` for non-review legacy callers, and use this production path when it is provided:

```python
vector_results = vector_retriever.retrieve(question)[:TOP_K]
bm25_results = bm25_index.retrieve(question, top_k=TOP_K)
for result in [*vector_results, *bm25_results]:
    result.node.metadata["retrieval_score"] = result.score
merged_results = dedupe([*vector_results, *bm25_results], key=node.node_id)
reranked_results = reranker.postprocess_nodes(
    merged_results,
    query_str=question,
)[:RERANK_TOP_N]
```

Deduplication must preserve the first result for a Node ID, so a Node found by both indexes keeps the vector result’s raw score while BM25-only Nodes keep their BM25 raw score. Store raw scores in `node.metadata["retrieval_score"]`; `result.score` remains available for the reranker’s final score. Do not calculate a combined score.

- [x] **Step 2: Add stage timing and structured hybrid logs**

Use `time.perf_counter()` around vector retrieval, BM25 retrieval, deduplication, rerank, Evidence selection, and the full operation. Add a `retrieval_diagnostics` dictionary to the pipeline result with integer counts and millisecond timings. Emit one `logger.info` record containing `vector_count`, `bm25_count`, `merged_candidate_count`, `reranked_count`, `evidence_count`, `evidence_node_ids`, `vector_ms`, `bm25_ms`, `merge_ms`, `rerank_ms`, `evidence_ms`, and `total_ms`.

Keep the existing `vector_results`, `reranked_results`, `selected_indices`, and `selected_nodes` keys so current serializers and callers remain valid. Preserve the existing selector fallback behavior; the removed feature is the contract-review query-rewrite/full-text fallback, not the existing selector error safety net.

- [x] **Step 3: Wire `ContractService.search_contract` to both persisted indexes**

Load the vector index using `self.index_manager.get(contract)` and BM25 using `self.index_manager.get_bm25(contract)`. Call:

```python
self.rag_pipeline.retrieve_evidence(
    index,
    query,
    bm25_index=bm25_index,
    fallback_on_empty_selection=False,
)
```

Wrap a missing BM25 directory/file as a clear `RuntimeError` whose message says the BM25 index is unavailable and the contract must be re-parsed (`重新解析`). Never catch the error and return vector-only results.

- [x] **Step 4: Run the focused tests and verify green**

Run: `uv run --with pytest python -m pytest tests/test_rag_pipeline.py -k hybrid tests/test_contract_service_retrieval.py -q`

Expected: PASS, including assertions that the reranker receives three deduplicated candidates and that the log/diagnostic counts are present.

### Task 3: Add failing tests for removing the formal retrieval fallback

**Files:**
- Modify: `tests/test_contract_review_graph.py`
- Modify: `tests/test_contract_review_service.py`

**Interfaces:**
- Consumes: `build_contract_review_graph`, `ContractReviewNodes`, `ContractReviewService`, existing fake LLM and ContractService fixtures.
- Produces: tests proving one retrieval attempt, no query-rewrite LLM, no content-object scan, and direct existing insufficient result behavior.

- [x] **Step 1: Replace retry/absence graph expectations with the desired single-attempt expectation**

Change the empty-retrieval graph test to construct `ContractReviewNodes` without a query-rewrite LLM and invoke the graph with an empty evidence response. Assert:

```python
assert len(contract_service.searches) == 1
assert contract_service.content_loads == []
assert result.evidence_status == "insufficient"
assert result.risk_status == "needs_review"
assert result.evidence == []
```

Assert that the review LLM and query-rewrite LLM are not invoked on empty retrieval and that no `retrieval_query_rewritten`, `absence_check_started`, `absence_candidates_found`, or `absence_confirmed` event is emitted.

- [x] **Step 2: Update service construction expectations**

Change the default LLM construction test to expect two `ChatOpenAI` constructions: rule parser and risk reviewer. Remove production-service tests that expect a rewritten query or an absence scan; replace them with a test that an empty search is finalized as `insufficient` after one call.

- [x] **Step 3: Run the focused review tests and verify the expected red failure**

Run: `uv run --with pytest python -m pytest tests/test_contract_review_graph.py tests/test_contract_review_service.py -q`

Expected: FAIL because the graph still routes empty retrieval to `rewrite_query` and `absence_check`, and the default builder still creates a third LLM.

### Task 4: Remove the formal fallback route while preserving result compatibility

**Files:**
- Modify: `app/contract_review/graph.py:11-94`
- Modify: `app/contract_review/nodes.py:57-380`
- Modify: `app/contract_review/service.py:45-151`
- Modify: `tests/test_contract_review_graph.py`
- Modify: `tests/test_contract_review_service.py`

**Interfaces:**
- Consumes: hybrid `ContractService.search_contract`, existing `Evidence`, `RiskDecision`, `ReviewResult`, `insufficient_result`, and current progress callback.
- Produces: one-shot review graph with the existing output schema and no formal query-rewrite/full-text fallback.

- [x] **Step 1: Simplify graph routing**

Change `route_after_retrieve` to return `risk_decision` when `retrieved_evidence` is non-empty and `insufficient_result` otherwise. Change `route_after_risk_decision` to always return `finalize_review_item`. Remove `rewrite_query`, `absence_check`, and `absence_result` graph nodes and edges. Keep `insufficient_result -> finalize_review_item`.

- [x] **Step 2: Remove query-rewrite and absence execution from review nodes**

Remove the `query_rewrite_llm` constructor dependency and the node methods that execute query rewrite or source-object keyword scanning. Keep the existing `ContractReviewState` fields and `prepare_review_item` reset payload unchanged so graph callers and persisted historical results remain compatible. In `insufficient_result`, replace the stale “两次检索” wording with a single hybrid-retrieval wording. In `finalize_review_item`, use `state["retrieved_evidence"]` as the only current evidence source and leave `absence_check=None` for new runs.

- [x] **Step 3: Stop building the removed LLM**

Remove `RetrievalQueryRewrite` from the formal service imports and constructor wiring. `build_contract_review_service` must create only `ReviewItemList` and `RiskDecision` LLMs. Keep `AbsenceCheckMetadata`, historical `ReviewResult.absence_check`, `app/contract_review/absence.py`, and the old prompt/schema definitions because existing standalone tests and stored-result rendering cover them; no active review graph node or default service may reference them.

- [x] **Step 4: Run the focused review tests and verify green**

Run: `uv run --with pytest python -m pytest tests/test_contract_review_graph.py tests/test_contract_review_service.py -q`

Expected: PASS, with one search per item, direct insufficient results for empty hybrid Evidence, no content scan, and two default LLM constructions.

### Task 5: Preserve existing pages, diagnostics, and integration behavior

**Files:**
- Modify: `app/review_logging.py` to preserve the generic bounded event journal while accepting the new `retrieval_diagnostics` payload without changing public review output.
- Modify: `tests/test_review_service.py` to assert that the journal preserves hybrid retrieval diagnostics and still renders historical event payloads.
- Modify: `tests/test_app_review_page.py` to retain the existing historical `absence_check` rendering assertion while adding no new-run fallback output.
- Modify: `tests/test_contract_pipeline.py`, `tests/test_index_manager.py` to retain the existing dual-index version/cache assertions and confirm this task does not alter parse-time persistence.

**Interfaces:**
- Consumes: the unchanged review result serializer/page and `RAGPipeline` diagnostics.
- Produces: backward-compatible page/API payloads, bounded run journals, and explicit diagnostics for new hybrid retrieval.

- [x] **Step 1: Add/adjust diagnostics assertions**

Ensure a review progress/journal event can carry `retrieval_diagnostics` and that diagnostics are bounded by the existing journal limits. Assert that `ReviewResult` and page payloads still expose the same Evidence fields and that stored historical `absence_check` data continues to render.

- [x] **Step 2: Run all affected tests**

Run: `uv run --with pytest python -m pytest tests/test_rag_pipeline.py tests/test_contract_service_retrieval.py tests/test_contract_review_graph.py tests/test_contract_review_service.py tests/test_review_service.py tests/test_app_review_page.py tests/test_contract_pipeline.py tests/test_index_manager.py -q`

Expected: PASS with no changes to the parse/index persistence contract beyond the already verified BM25 implementation.

### Task 6: Verify with full suite and a real-contract smoke test

**Files:**
- No production files expected.
- Read-only runtime data: `data/contracts/41e1115b-e285-4aaa-860f-f608de4b4929/index/` and `data/contracts/41e1115b-e285-4aaa-860f-f608de4b4929/bm25_index/`.

**Interfaces:**
- Consumes: committed tests, persisted vector/BM25 indexes, configured rerank/LLM endpoints if available.
- Produces: fresh verification evidence for the acceptance checklist.

- [x] **Step 1: Validate persisted Node identity before the smoke query**

Read the real contract’s vector docstore and BM25 `index.json` metadata and compare the sets of `node_id`, original Node text, and `retrieval_context`. Do not rewrite either index or contract data.

- [x] **Step 2: Run one real hybrid retrieval smoke test**

Use the real contract’s loaded indexes and a Chinese query such as `乙方未经甲方同意能否进行分包？`. Execute the formal `ContractService.search_contract` path with the configured reranker/selector when those services are available. Record vector count, BM25 count, merged count, reranked count, final Evidence IDs, and elapsed timings. If an external endpoint is unavailable, run the real persisted-index candidate merge with a test reranker and report the external dependency separately.

- [x] **Step 3: Run the full test suite**

Run: `uv run --with pytest python -m pytest -q`

Expected: exit code 0 and zero failed tests.

- [x] **Step 4: Inspect the final diff and commit implementation**

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff -- app/rag_pipeline.py app/service.py app/contract_review/graph.py app/contract_review/nodes.py app/contract_review/service.py
```

Confirm there are no unrelated data/index changes, then commit:

```bash
git add app tests docs/superpowers/plans/2026-09-09-production-bm25-hybrid-review.md
git commit -m "feat: integrate BM25 into contract review"
```
