# Contract Evaluation Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已持久化合同索引之上增加可配置、可单题/全量运行、可追溯的 Evaluation 页面，并让聊天和评测共享同一个不带 Evaluation 语义的单题 RAG Pipeline。

**Architecture:** 保留 `retrieval_evaluation.py` 的检索、重排、选择、回答算法和脚本入口，抽出 `app/rag_pipeline.py` 的 `RAGPipeline.run()` 作为唯一单题调用链。`EvaluationService` 在 Pipeline 原始结果之上计算 Recall、绑定 index version、保存 config snapshot 和完整 JSON 结果；FastAPI 页面通过 `BackgroundTasks` 调度运行，启动时回收遗留 run。

**Tech Stack:** Python 3.14+, FastAPI, Starlette TestClient, Jinja2 templates, SQLite (`sqlite3`), LlamaIndex persisted index, existing DashScope reranker and LangChain structured LLMs.

**Spec:** `docs/superpowers/specs/2026-08-17-evaluation-page-design.md`

## Global Constraints

- 评测正确节点 ID 使用现有 `source_object_index` 整数，不使用 LlamaIndex UUID `node_id`。
- 聊天和评测都必须调用 `RAGPipeline.run()` 的 `retrieve → rerank → selector → answer` 链路；聊天不构造空 gold，不调用评测入口。
- 评测只能加载 `<contract.storage_dir>/index`，禁止因运行评测重新解析 PDF 或重新向量化。
- 每次成功建索引生成新的 `contracts.index_version`；旧 `evaluation_cases` 与新版本不自动迁移，版本不一致时禁止运行。
- `evaluation_runs.config_snapshot` 至少保存 `vector_top_k`、`rerank_top_k`、rerank/selector/answer 模型名和 `pipeline_version`。
- 单题和全量评测使用 FastAPI `BackgroundTasks`；应用创建/启动时将遗留 `queued`/`processing` run 标记为 `failed`。
- 不新增 ORM、向量数据库、任务队列、Vue、React 或独立前端工程。
- 不覆盖或回滚任务开始前已有的工作区改动；本地 `docs/superpowers/.DS_Store` 不纳入本次提交。

---

### Task 1: Extract the shared single-question RAG Pipeline

**Files:**
- Create: `app/rag_pipeline.py`
- Create: `tests/test_rag_pipeline.py`
- Modify: `retrieval_evaluation.py`
- Modify: `app/qa.py`
- Modify: `tests/test_app_qa.py`

**Interfaces:**
- `retrieval_evaluation.retrieve_and_rerank(index: Any, query: str, *, reranker: Any | None = None) -> tuple[list[Any], list[Any]]` performs vector Top10 and rerank only.
- `retrieval_evaluation.filter_nodes_by_indices(reranked_nodes: list[Any], selected_indices: list[int]) -> list[Any]` is the public name for the current private filtering helper.
- `app.rag_pipeline.RAGPipeline.run(index, question, *, reranker=None, selector_llm=None, answer_llm=None) -> dict[str, Any]` returns raw result objects with `query`, `vector_results`, `reranked_results`, `selected_indices`, `selected_nodes`, and `llm_summary`.
- `app.qa.answer_question(index, question, *, debug=False, reranker=None, selector_llm=None, answer_llm=None, pipeline=None) -> dict[str, Any]` remains the chat serialization boundary and delegates to `RAGPipeline.run()`.

**Behavior:**
- `RAGPipeline.run()` rejects a blank question.
- It does not accept expected IDs, compute Recall, or call `retrieval_evaluation.run_evaluation()`.
- The output `llm_summary.answer` and `selected_nodes` use the existing selector fallback and answer fallback behavior.
- `retrieval_evaluation.run_evaluation()` remains backward-compatible for the CLI and existing tests, but internally uses `retrieve_and_rerank()` before adding evaluation-only metrics.

- [ ] **Step 1: Write the failing Pipeline test**

```python
def test_pipeline_runs_retrieve_rerank_selector_and_answer_without_gold_semantics():
    index = FakeIndex([result_for(10, "证据", 0.8)])
    pipeline = RAGPipeline()

    result = pipeline.run(
        index,
        "问题",
        reranker=FakeReranker(),
        selector_llm=FakeLLM({"evidence_indices": [10]}),
        answer_llm=FakeLLM({"answer": "答案"}),
    )

    assert result["query"] == "问题"
    assert result["selected_indices"] == [10]
    assert result["llm_summary"] == {"answer": "答案", "evidence_indices": [10]}
    assert "expected_source_object_indices" not in result
```

- [ ] **Step 2: Run the focused test and verify it fails for the missing Pipeline**

Run: `uv run pytest tests/test_rag_pipeline.py::test_pipeline_runs_retrieve_rerank_selector_and_answer_without_gold_semantics -q`

Expected: FAIL because `app.rag_pipeline.RAGPipeline` does not exist.

- [ ] **Step 3: Add the low-level retrieve/rerank helper and Pipeline**

Refactor the existing retrieval body into this helper without changing constants or reranker behavior:

```python
def retrieve_and_rerank(index: Any, query: str, *, reranker: Any | None = None):
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    vector_results = list(retriever.retrieve(query))[:TOP_K]
    _record_vector_scores(vector_results)
    active_reranker = build_reranker() if reranker is None else reranker
    reranked_results = list(
        active_reranker.postprocess_nodes(vector_results, query_str=query)
    )[:RERANK_TOP_N]
    return vector_results, reranked_results
```

Implement `RAGPipeline.run()` by calling that helper, then `select_evidence()`, `filter_nodes_by_indices()`, and `generate_answer()`. Keep the existing prompts and fallback messages unchanged.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `uv run pytest tests/test_rag_pipeline.py::test_pipeline_runs_retrieve_rerank_selector_and_answer_without_gold_semantics -q`

Expected: PASS.

- [ ] **Step 5: Update chat serialization and preserve existing CLI tests**

Change `app.qa.answer_question()` to call the injected/default `RAGPipeline.run()` and serialize its raw `selected_nodes` and `reranked_results`. Keep `debug=False` returning `debug=None`; keep `debug=True` returning at most 10 reranked candidates, selected evidence, and final answer. Change `generate_summaries()`/`run_evaluation()` only enough to use the new low-level helper and retain their current output keys.

- [ ] **Step 6: Run the Pipeline, QA, and existing retrieval tests**

Run: `uv run pytest tests/test_rag_pipeline.py tests/test_app_qa.py tests/test_retrieval_evaluation.py -q`

Expected: all focused tests pass, including the existing checks for selector fallback, answer fallback, reranker request mapping, and CLI evaluation output.

- [ ] **Step 7: Commit the shared Pipeline**

```bash
git add app/rag_pipeline.py app/qa.py retrieval_evaluation.py tests/test_rag_pipeline.py tests/test_app_qa.py
git commit -m "refactor: share single question rag pipeline"
```

### Task 2: Add contract index versions and version-aware index caching

**Files:**
- Modify: `app/models.py`
- Modify: `app/db.py`
- Modify: `app/index_manager.py`
- Modify: `app/pipeline.py`
- Modify: `tests/test_app_db.py`
- Modify: `tests/test_index_manager.py`
- Modify: `tests/test_contract_pipeline.py`
- Modify: `tests/test_app_api.py`
- Modify: `tests/test_app_page.py`

**Interfaces:**
- `ContractRecord` gains `index_version: str | None`; `to_dict()` exposes it.
- `ContractRepository.update_status(contract_id, status, error_message=None, *, index_version: str | None = None) -> ContractRecord` writes the version supplied by the caller and clears it when the contract is not ready.
- `IndexManager.put(contract_id: str, index: Any, *, index_version: str | None = None) -> None` and `get(contract: ContractRecord) -> Any` use `(contract_id, index_version)` as the cache key.
- `IndexManager.clear(contract_id: str | None = None) -> None` removes all versions for one contract or all cache entries.
- `ContractProcessor.process()` generates a new UUID `index_version` only after `storage_context.persist()` succeeds, then caches the index with that version and marks the contract ready with the same version.

**Migration behavior:**
- On an existing database, `_initialize()` checks `PRAGMA table_info(contracts)` and runs `ALTER TABLE contracts ADD COLUMN index_version TEXT` if the column is absent.
- When processing starts, the contract is marked `processing` with no active version; on failure it is `failed` with no active version. This prevents evaluation from using a version whose build did not complete.

- [ ] **Step 1: Write failing model and repository tests**

```python
def test_ready_contract_persists_index_version():
    repository = ContractRepository(db_path)
    contract = repository.create("contract.pdf", storage_dir)

    ready = repository.update_status(
        contract.contract_id,
        "ready",
        index_version="index-v2",
    )

    assert ready.index_version == "index-v2"
    assert repository.get(contract.contract_id).to_dict()["index_version"] == "index-v2"

def test_existing_contract_table_is_migrated_with_index_version():
    create_legacy_contracts_table_without_index_version(db_path)
    repository = ContractRepository(db_path)

    assert "index_version" in table_columns(db_path, "contracts")
    assert repository.get("legacy-id").index_version is None
```

- [ ] **Step 2: Run the focused repository tests and verify they fail**

Run: `uv run pytest tests/test_app_db.py -q`

Expected: FAIL because `ContractRecord` and the `contracts` table do not yet have `index_version`.

- [ ] **Step 3: Implement the model, schema migration, and status update**

Add the nullable field, include it in row conversion and payloads, add the migration check, and update SQL so a non-ready status clears the active version while `ready` stores the supplied version. Keep existing calls such as `update_status(contract_id, "ready")` valid by allowing `index_version=None`.

- [ ] **Step 4: Run repository tests and verify they pass**

Run: `uv run pytest tests/test_app_db.py -q`

Expected: PASS, including all pre-existing SQLite connection lifecycle tests.

- [ ] **Step 5: Write the failing version-aware cache test**

```python
def test_cache_does_not_reuse_an_index_from_another_contract_version():
    with TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir) / "index"
        index_dir.mkdir()
        manager = IndexManager(embedding_model=object())
        first = object()
        second = object()
        manager.put("c1", first, index_version="v1")
        contract = SimpleNamespace(
            contract_id="c1",
            index_version="v2",
            storage_dir=temp_dir,
        )

        with patch("app.index_manager.load_index_from_storage", return_value=second):
            assert manager.get(contract) is second
```

- [ ] **Step 6: Run the cache test and verify it fails**

Run: `uv run pytest tests/test_index_manager.py::test_cache_does_not_reuse_an_index_from_another_contract_version -q`

Expected: FAIL because the current cache key contains only `contract_id`.

- [ ] **Step 7: Implement version-aware cache and ingestion versioning**

Use a tuple cache key, clear all old entries at the start of `ContractProcessor.process()`, generate `str(uuid.uuid4())` after persist, call `put(contract_id, index, index_version=index_version)`, and update status to `ready` with that version. Update test fixtures that manually create ready contracts to pass `index_version="v1"`.

- [ ] **Step 8: Run index and pipeline tests**

Run: `uv run pytest tests/test_index_manager.py tests/test_contract_pipeline.py -q`

Expected: PASS, with the existing persist-before-ready ordering unchanged and the new version assertions passing.

- [ ] **Step 9: Commit index version support**

```bash
git add app/models.py app/db.py app/index_manager.py app/pipeline.py tests/test_app_db.py tests/test_index_manager.py tests/test_contract_pipeline.py tests/test_app_api.py tests/test_app_page.py
git commit -m "feat: bind contracts to index versions"
```

### Task 3: Add evaluation case and run persistence

**Files:**
- Create: `app/evaluation_models.py`
- Create: `app/evaluation_db.py`
- Create: `tests/test_evaluation_db.py`

**Interfaces:**
- `EvaluationCase` dataclass fields: `case_id`, `contract_id`, `index_version`, `question`, `expected_source_object_indices`, `sort_order`, `created_at`, `updated_at`.
- `EvaluationRun` dataclass fields: `run_id`, `contract_id`, `scope`, `status`, `index_version`, `pipeline_version`, `config_snapshot`, `created_at`, `started_at`, `completed_at`, `error_message`.
- `EvaluationRunItem` dataclass fields: `run_id`, `case_id`, `question_snapshot`, `expected_source_object_indices_snapshot`, `result`.
- `EvaluationRepository(database_path: Path)` creates `evaluation_cases`, `evaluation_runs`, and `evaluation_run_items` with foreign keys and JSON text fields.
- `list_cases(contract_id: str) -> list[EvaluationCase]` returns `sort_order` ascending.
- `get_case(contract_id: str, case_id: int) -> EvaluationCase | None` only returns a case from the requested contract.
- `replace_cases(contract_id: str, index_version: str, entries: list[tuple[str, list[int]]]) -> list[EvaluationCase]` atomically replaces all cases for that contract and writes the current index version.
- `create_run(contract_id: str, scope: str, index_version: str, config_snapshot: dict[str, Any], cases: list[EvaluationCase]) -> EvaluationRun` stores a UUID run and case snapshots.
- `mark_processing`, `save_item`, `mark_ready`, `mark_failed`, `get_run`, `list_run_items`, `latest_run` provide the run lifecycle.
- `recover_incomplete_runs(reason: str) -> int` marks every `queued` or `processing` run as `failed`, sets `completed_at`, and stores the reason.

**Schema details:**
- `evaluation_cases.expected_source_object_indices` and run snapshots are JSON arrays.
- `evaluation_runs.config_snapshot` and `evaluation_run_items.result_json` are JSON objects.
- `evaluation_runs.status` is constrained to `queued`, `processing`, `ready`, and `failed`; `scope` is constrained to `single` and `all`.
- Deleting a contract cascades to cases, runs, and run items.

- [ ] **Step 1: Write failing persistence tests**

```python
def test_replace_cases_binds_current_index_version_and_preserves_order():
    cases = repository.replace_cases(
        "c1",
        "index-v3",
        [("第二题", [20, 21]), ("第一题", [10])],
    )

    assert [case.question for case in cases] == ["第二题", "第一题"]
    assert [case.index_version for case in cases] == ["index-v3", "index-v3"]
    assert cases[0].expected_source_object_indices == [20, 21]

def test_run_snapshot_and_recovery_are_persisted():
    run = repository.create_run(
        "c1", "all", "index-v3",
        {"vector_top_k": 10, "rerank_top_k": 10, "pipeline_version": "rag-v1"},
        cases,
    )
    repository.mark_processing(run.run_id)
    repository.save_item(run.run_id, cases[0], {"query": "第二题", "vector_results": []})
    repository.recover_incomplete_runs("service restarted")

    loaded = repository.get_run(run.run_id)
    assert loaded.status == "failed"
    assert loaded.error_message == "service restarted"
    assert loaded.config_snapshot["pipeline_version"] == "rag-v1"
    assert repository.list_run_items(run.run_id)[0].result["query"] == "第二题"
```

- [ ] **Step 2: Run the persistence tests and verify they fail**

Run: `uv run pytest tests/test_evaluation_db.py -q`

Expected: FAIL because the evaluation models, tables, and repository do not exist.

- [ ] **Step 3: Implement evaluation models and SQLite repository**

Use one short-lived `sqlite3.connect()` per method, `sqlite3.Row`, `PRAGMA foreign_keys=ON`, UTC ISO timestamps, a write lock, and transactions. `replace_cases()` must delete and insert inside one transaction, so invalid input cannot leave a partially replaced set. Deserialize JSON into fresh Python lists/dicts on every read.

- [ ] **Step 4: Run the persistence tests and verify they pass**

Run: `uv run pytest tests/test_evaluation_db.py -q`

Expected: PASS.

- [ ] **Step 5: Commit evaluation persistence**

```bash
git add app/evaluation_models.py app/evaluation_db.py tests/test_evaluation_db.py
git commit -m "feat: persist evaluation cases and runs"
```

### Task 4: Implement evaluation metrics, result serialization, and EvaluationService

**Files:**
- Create: `app/evaluation_metrics.py`
- Create: `app/evaluation_service.py`
- Create: `tests/test_evaluation_service.py`
- Create: `tests/test_evaluation_metrics.py`
- Modify: `app/rag_pipeline.py`

**Interfaces:**
- `PIPELINE_VERSION = "rag-v1"` lives in `app.rag_pipeline` and is used by the evaluation config snapshot.
- `build_config_snapshot() -> dict[str, Any]` returns `vector_top_k`, `rerank_top_k`, `rerank_model`, `selector_model`, `answer_model`, and `pipeline_version` from the current retrieval configuration.
- `build_evaluation_result(pipeline_result: dict[str, Any], expected_source_object_indices: list[int]) -> dict[str, Any]` adds `expected_source_object_indices`, vector/rerank scores and ranks, and four Recall values without changing the Pipeline result.
- `serialize_pipeline_result(result: dict[str, Any]) -> dict[str, Any]` converts every node result in vector/reranked/selected collections into JSON-safe dictionaries containing numeric source index, real `node_id`, text, page metadata, retrieval context, retrieval score, and final score.
- `EvaluationService.list_cases(contract_id: str) -> list[EvaluationCase]`.
- `EvaluationService.default_cases() -> list[tuple[str, list[int]]]` returns the existing script queries for an empty contract evaluation set.
- `EvaluationService.save_cases(contract_id: str, entries: list[tuple[str, list[int]]]) -> list[EvaluationCase]` validates nonblank questions and integer IDs, requires a ready contract with a current `index_version`, and writes cases bound to that version.
- `EvaluationService.create_single_run(contract_id: str, case_id: int) -> EvaluationRun` and `create_all_run(contract_id: str) -> EvaluationRun` reject stale case versions and snapshot the current index version/config.
- `EvaluationService.execute_run(run_id: str) -> EvaluationRun` loads the run snapshot, loads the persisted index once, calls `RAGPipeline.run()` once per case, adds evaluation metrics, saves each item, and marks the run ready or failed.
- `EvaluationService.get_run_payload(run_id: str) -> dict[str, Any]` returns JSON-safe run metadata and item results for the page/API.
- `EvaluationService.recover_interrupted_runs() -> int` delegates to `EvaluationRepository.recover_incomplete_runs()`.

**Metric semantics:**
- `vector_recall_at_5/10` uses the `vector_results` order from the shared Pipeline.
- `rerank_recall_at_5/10` uses the `reranked_results` order from the shared Pipeline.
- An empty expected list produces `None`, matching the existing script.
- UI labels distinguish Vector and Rerank values; no aggregate average is added in this version.

- [ ] **Step 1: Write failing metrics and service tests**

```python
def test_evaluation_metrics_compare_gold_only_after_pipeline_runs():
    result = {
        "query": "问题",
        "vector_results": [result_for(1), result_for(7), result_for(8)],
        "reranked_results": [result_for(7), result_for(1), result_for(8)],
        "selected_nodes": [result_for(7)],
        "llm_summary": {"answer": "答案", "evidence_indices": [7]},
    }

    evaluated = build_evaluation_result(result, [7])

    assert evaluated["vector_recall_at_5"] == 1.0
    assert evaluated["rerank_recall_at_5"] == 1.0
    assert evaluated["vector_ranks"][7] == 2
    assert evaluated["rerank_ranks"][7] == 1

def test_service_rejects_case_bound_to_old_index_version():
    service = build_evaluation_service(contract_index_version="index-v2")
    old_case = repository.replace_cases("c1", "index-v1", [("问题", [7])])[0]

    with pytest.raises(EvaluationStaleError):
        service.create_single_run("c1", old_case.case_id)
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `uv run pytest tests/test_evaluation_metrics.py tests/test_evaluation_service.py -q`

Expected: FAIL because the metrics adapter, service, and stale-version error do not exist.

- [ ] **Step 3: Implement metric enrichment and JSON serialization**

Reuse `retrieval_evaluation.recall_at_k()` and the existing source-index extraction semantics. Keep raw Pipeline results untouched for the in-process caller; create a new JSON-safe dict for persistence. Serialize `score` and `retrieval_score` only when present, and retain absent page metadata as absent keys.

- [ ] **Step 4: Run metrics tests and verify they pass**

Run: `uv run pytest tests/test_evaluation_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Implement EvaluationService run lifecycle**

For `create_single_run()` and `create_all_run()`, load the contract, require `status == "ready"` and nonempty `index_version`, load cases, require every selected case version to equal the contract version, and call `EvaluationRepository.create_run()` with `build_config_snapshot()`.

For `execute_run()`, call `mark_processing()`, fetch the contract again, call `IndexManager.get(contract)` exactly once, iterate the stored case snapshots in order, run the shared Pipeline, enrich and serialize each result, save the item immediately, and mark ready only after all items succeed. Catch any exception, log it, mark the run failed with `str(exc)`, and return the failed run.

- [ ] **Step 6: Add service tests for one index load and complete result persistence**

```python
def test_execute_all_run_reuses_one_persisted_index_and_saves_full_results():
    run = service.create_all_run("c1")
    result = service.execute_run(run.run_id)

    assert result.status == "ready"
    index_manager.get.assert_called_once()
    assert len(repository.list_run_items(run.run_id)) == 2
    saved = repository.list_run_items(run.run_id)[0].result
    assert "vector_results" in saved
    assert "reranked_results" in saved
    assert "vector_recall_at_10" in saved

def test_execute_run_marks_failed_when_index_load_fails():
    index_manager.get.side_effect = FileNotFoundError("missing index")
    run = service.create_single_run("c1", case_id)

    result = service.execute_run(run.run_id)

    assert result.status == "failed"
    assert result.error_message == "missing index"
```

- [ ] **Step 7: Run service tests and verify they pass**

Run: `uv run pytest tests/test_evaluation_metrics.py tests/test_evaluation_service.py -q`

Expected: PASS.

- [ ] **Step 8: Commit evaluation execution**

```bash
git add app/evaluation_metrics.py app/evaluation_service.py app/rag_pipeline.py tests/test_evaluation_metrics.py tests/test_evaluation_service.py
git commit -m "feat: execute and persist evaluation results"
```

### Task 5: Wire Pipeline and EvaluationService into ContractService and recover interrupted runs

**Files:**
- Modify: `app/service.py`
- Modify: `app/api.py`
- Modify: `tests/test_app_qa.py`
- Modify: `tests/test_app_api.py`
- Create: `tests/test_evaluation_recovery.py`

**Interfaces:**
- `ContractService(..., rag_pipeline: RAGPipeline | None = None, evaluation_service: EvaluationService | None = None)` accepts injected dependencies while preserving the existing four positional constructor arguments.
- `ContractService.ask()` calls `answer_question(..., pipeline=self.rag_pipeline)`.
- `ContractService.recover_interrupted_evaluation_runs() -> int` delegates to `self.evaluation_service.recover_interrupted_runs()`.
- `ContractService` exposes evaluation facade methods used by routes: `list_evaluation_cases`, `save_evaluation_cases`, `create_single_evaluation_run`, `create_all_evaluation_run`, `execute_evaluation_run`, and `get_evaluation_run_payload`.
- `build_default_service()` constructs one `RAGPipeline`, one `EvaluationRepository`, and one `EvaluationService` sharing the same `ContractRepository` and `IndexManager`.

- [ ] **Step 1: Write failing wiring and startup recovery tests**

```python
def test_chat_service_uses_injected_shared_pipeline():
    pipeline = Mock()
    service = ContractService(repository, settings, Mock(), index_manager, rag_pipeline=pipeline)
    pipeline.run.return_value = {
        "query": "问题",
        "vector_results": [],
        "reranked_results": [],
        "selected_nodes": [],
        "selected_indices": [],
        "llm_summary": {"answer": "答案", "evidence_indices": []},
    }
    repository.update_status(contract.contract_id, "ready", index_version="index-v1")

    service.ask(contract.contract_id, "问题")

    pipeline.run.assert_called_once()

def test_app_creation_recovers_interrupted_runs():
    evaluation_service = Mock()
    service = ContractService(repository, settings, Mock(), Mock(), evaluation_service=evaluation_service)

    create_app(settings=settings, service=service)

    evaluation_service.recover_interrupted_runs.assert_called_once_with()
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `uv run pytest tests/test_app_qa.py tests/test_app_api.py tests/test_evaluation_recovery.py -q`

Expected: FAIL because the constructor has no shared Pipeline/evaluation dependency and app creation does not recover runs.

- [ ] **Step 3: Implement dependency wiring and startup recovery**

Add optional dependency arguments after the existing constructor arguments. Build defaults only when not injected. Call recovery once in `create_app()` immediately after selecting `active_service`, before serving routes. Do not run any RAG call during recovery.

- [ ] **Step 4: Run wiring and existing app tests**

Run: `uv run pytest tests/test_app_qa.py tests/test_app_api.py tests/test_evaluation_recovery.py tests/test_app_page.py -q`

Expected: PASS.

- [ ] **Step 5: Commit service wiring**

```bash
git add app/service.py app/api.py tests/test_app_qa.py tests/test_app_api.py tests/test_evaluation_recovery.py
git commit -m "feat: wire evaluation service and startup recovery"
```

### Task 6: Add Evaluation routes and server-rendered page

**Files:**
- Create: `app/templates/evaluation.html`
- Create: `tests/test_app_evaluation_page.py`
- Modify: `app/api.py`
- Modify: `app/templates/index.html`
- Modify: `tests/test_app_page.py`

**Interfaces:**
- `GET /contracts/{contract_id}/evaluation` renders the contract, saved cases, default cases for an empty set, latest run payload, selected run ID, and page error.
- `POST /contracts/{contract_id}/evaluation/config` accepts repeated `question` and `expected_source_object_indices` form fields, validates and replaces the saved case set, then redirects to the Evaluation page.
- `POST /contracts/{contract_id}/evaluation/cases/{case_id}/run` creates a single run, schedules `active_service.execute_evaluation_run(run_id)`, and redirects with `run_id`.
- `POST /contracts/{contract_id}/evaluation/run-all` creates an all-case run, schedules it with `BackgroundTasks`, and redirects with `run_id`.
- `GET /api/contracts/{contract_id}/evaluation/runs/{run_id}` returns a JSON-safe run payload and rejects a run from another contract.

**Form semantics:**
- `expected_source_object_indices` accepts commas, spaces, or newlines; `parse_expected_indices("111, 112\\n113")` returns `[111, 112, 113]` with duplicates removed in input order.
- A blank question or malformed ID returns the same page with the submitted rows and a visible error; no partial replacement occurs.
- A case bound to an old `index_version` renders as stale and its test button is disabled until the user saves the current configuration.
- A non-ready contract returns 409 for JSON and a page error for HTML; a missing contract returns 404.

- [ ] **Step 1: Write failing route and template tests**

```python
def test_ready_contract_has_evaluation_entry_and_page_shows_saved_case():
    contract = make_ready_contract(index_version="index-v1")
    evaluation_repository.replace_cases(contract.contract_id, "index-v1", [("付款方式？", [111])])

    response = client.get(f"/contracts/{contract.contract_id}/evaluation")

    assert response.status_code == 200
    assert "Evaluation" in home_client.get("/").text
    assert "付款方式？" in response.text
    assert 'name="question"' in response.text
    assert 'name="expected_source_object_indices"' in response.text

def test_all_run_route_schedules_background_execution():
    contract = make_ready_contract(index_version="index-v1")
    evaluation_repository.replace_cases(contract.contract_id, "index-v1", [("问题", [1])])

    response = client.post(f"/contracts/{contract.contract_id}/evaluation/run-all", follow_redirects=False)

    assert response.status_code == 303
    assert "run_id=" in response.headers["location"]
    service.execute_evaluation_run.assert_called_once()
```

- [ ] **Step 2: Run focused page tests and verify they fail**

Run: `uv run pytest tests/test_app_evaluation_page.py -q`

Expected: FAIL because the Evaluation routes/template and dashboard link do not exist.

- [ ] **Step 3: Implement parsing and HTML/JSON route helpers**

Add a small route-local parser or `app/evaluation_forms.py` helper with this exact behavior:

```python
def parse_expected_indices(value: str) -> list[int]:
    tokens = re.split(r"[\\s,，]+", value.strip())
    if not value.strip():
        return []
    if any(not token.isdigit() for token in tokens):
        raise ValueError("正确 Node ID 必须是数字")
    return list(dict.fromkeys(int(token) for token in tokens))
```

Render the form rows with stable `case_id` values when saved and an empty hidden state for new rows. Use the existing markdown filter for answer text. Add a ready-only `Evaluation` link beside `开始问答` in `index.html`.

- [ ] **Step 4: Implement run scheduling and status endpoint**

Use `BackgroundTasks.add_task(active_service.execute_evaluation_run, run.run_id)` in both run POST handlers. Keep the POST response a 303 redirect. The status endpoint returns `active_service.get_evaluation_run_payload(run_id)` and uses HTTP 404 for missing/mismatched runs.

- [ ] **Step 5: Run route tests and verify they pass**

Run: `uv run pytest tests/test_app_evaluation_page.py tests/test_app_page.py tests/test_app_api.py -q`

Expected: PASS, including non-ready rejection and existing upload/chat behavior.

- [ ] **Step 6: Add page JavaScript for row editing and polling**

Implement only native browser behavior in `evaluation.html`:

```javascript
function addCaseRow() { /* clone the row template and clear its values */ }
function removeCaseRow(button) { /* remove the row, keep one blank row */ }
function pollRun(runId) {
  /* fetch the status endpoint every 2 seconds until ready or failed */
}
```

The rendered page must show per item: expected numeric IDs, vector actual IDs, rerank actual IDs, Vector Recall@5/10, Rerank Recall@5/10, candidate node text/scores/pages, selected evidence, and final answer. Do not add aggregate averages or version comparison UI.

- [ ] **Step 7: Commit the Evaluation page**

```bash
git add app/api.py app/templates/index.html app/templates/evaluation.html tests/test_app_evaluation_page.py tests/test_app_page.py
git commit -m "feat: add contract evaluation page"
```

### Task 7: Add regression coverage and verify the complete feature

**Files:**
- Modify: `tests/test_retrieval_evaluation.py`
- Modify: `tests/test_app_qa.py`
- Modify: `tests/test_app_api.py`
- Modify: `tests/test_app_page.py`
- Modify: `tests/test_evaluation_db.py`
- Modify: `tests/test_evaluation_service.py`
- Modify: `docs/superpowers/specs/2026-08-17-evaluation-page-design.md` only if implementation reveals a confirmed contract mismatch

- [ ] **Step 1: Add explicit shared-chain regression assertions**

Assert that chat calls `RAGPipeline.run()` without an expected-ID argument and that Evaluation calls the same Pipeline once per saved case, then computes metrics outside it. Assert that a full run calls `IndexManager.get()` once, not once per case.

- [ ] **Step 2: Add stale-version and recovery regression assertions**

Cover these exact transitions:

```text
ready/index-v1 + case/index-v1 → run allowed
ready/index-v2 + case/index-v1 → run rejected as stale
queued/processing run + app startup → failed with interruption message
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`

Expected: all tests pass with no collection errors or warnings caused by the feature.

- [ ] **Step 4: Run static/syntax checks**

Run: `uv run python -m compileall -q app retrieval_evaluation.py main.py`

Expected: exit code 0.

- [ ] **Step 5: Perform a route smoke test with test doubles**

Use a temporary SQLite database, a ready `ContractRecord` with `index_version="index-v1"`, a fake persisted index, fake reranker/LLMs, and the Starlette `TestClient` to verify this sequence:

```text
GET /                         → ready contract has Evaluation link
GET /contracts/{id}/evaluation → saved case form and no result yet
POST /contracts/{id}/evaluation/config → 303 and case persisted
POST /contracts/{id}/evaluation/cases/{case_id}/run → 303 with run_id
GET /api/contracts/{id}/evaluation/runs/{run_id} → ready result with Recall fields
```

- [ ] **Step 6: Inspect the final diff and status**

Run: `git diff --check`, `git status --short`, and `git diff --stat HEAD~7..HEAD`.

Expected: only the planned source/tests/docs files are committed; pre-existing or unrelated files remain untouched.

- [ ] **Step 7: Commit final regression coverage**

```bash
git add tests
git commit -m "test: cover evaluation page workflow"
```

## Plan Self-Review

- Spec coverage: shared Pipeline is Task 1; index/document version and cache invalidation are Task 2; evaluation case/run persistence and snapshots are Task 3; complete results and metrics are Task 4; startup recovery is Task 5; page and both run scopes are Task 6; end-to-end verification is Task 7.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps are required; all route names, model fields, test commands, and commit boundaries are explicit.
- Type consistency: `RAGPipeline.run()` returns the raw result consumed by `build_evaluation_result()` and `app.qa`; `EvaluationService.execute_run()` persists the enriched serialized result through `EvaluationRepository.save_item()`; API routes schedule the exact `execute_evaluation_run(run_id)` facade.
