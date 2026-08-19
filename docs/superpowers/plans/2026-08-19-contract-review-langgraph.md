# Contract Review LangGraph Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a synchronous LangGraph backend that parses natural-language review rules, retrieves real contract Evidence through the existing RAG stages, produces one strictly validated risk result per item, and aggregates a deterministic summary.

**Architecture:** Add a thin evidence-only method to the existing RAG pipeline and expose it through `ContractService.search_contract()`. Keep review schemas, prompts, LangGraph state/nodes/graph, dependency construction, and the progress callback inside `app/contract_review/`; use a reducer to accumulate sequential results and let Python attach immutable RAG Evidence to LLM decisions.

**Tech Stack:** Python 3.14, Pydantic 2, LangChain OpenAI, LlamaIndex, LangGraph 1.2, pytest

**Spec:** `docs/superpowers/specs/2026-08-19-contract-review-langgraph-design.md`

## Global Constraints

- Do not modify Web pages or add API routes.
- Do not add review-run database tables, BackgroundTasks, parallel execution, Agents, ReAct, tool-calling Agents, second retrieval, checkpointers, human-in-the-loop, or report generation.
- Preserve text, table, and image RAG; chat; Evaluation; and ingestion behavior.
- `ContractReviewService` preflight may only read the contract record and check `ready`; it must not load an index or trigger RAG.
- The first RAG call must occur in the first `review_item` node.
- LLMs never produce or modify Evidence citation metadata; Python attaches the serialized RAG Evidence.
- Rule parsing, risk LLM, retrieval, and Schema failures terminate the run instead of being converted into `needs_review`.
- Empty successfully retrieved Evidence becomes `needs_review / insufficient` and never “the contract has no clause.”
- Progress events remain transport-neutral and JSON-serializable for later Web reuse.
- Execute inline without multi-agent delegation.

---

## File Map

- Modify `pyproject.toml` and `uv.lock`: declare and lock LangGraph.
- Modify `app/rag_pipeline.py`: expose the evidence-only RAG stages and let `run()` reuse them.
- Modify `app/service.py`: expose ready-contract Evidence search without Answer Generator.
- Create `app/contract_review/__init__.py`: public review service exports.
- Create `app/contract_review/state.py`: LangGraph State and result reducer.
- Create `app/contract_review/schemas.py`: strict runtime models and LLM response parsing.
- Create `app/contract_review/prompts.py`: rule parsing and risk decision prompts.
- Create `app/contract_review/nodes.py`: three nodes and structured progress events.
- Create `app/contract_review/graph.py`: sequential loop and conditional routing.
- Create `app/contract_review/service.py`: preflight, graph invocation, default dependencies.
- Create `scripts/test_contract_review.py`: fixed-input command-line smoke runner.
- Modify `tests/test_rag_pipeline.py`: evidence-only pipeline regression tests.
- Modify `tests/test_app_qa.py`: contract search Evidence serialization tests.
- Create `tests/test_contract_review_schemas.py`: strict schema and cross-field tests.
- Create `tests/test_contract_review_graph.py`: real graph loop, accumulation, insufficiency, callbacks, and failure propagation.
- Create `tests/test_contract_review_service.py`: preflight and public output tests.

---

### Task 1: Add the evidence-only RAG boundary

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `app/rag_pipeline.py`
- Modify: `app/service.py`
- Test: `tests/test_rag_pipeline.py`
- Test: `tests/test_app_qa.py`

**Interfaces:**
- Produces: `RAGPipeline.retrieve_evidence(index, question, *, reranker=None, selector_llm=None) -> dict[str, Any]`
- Produces: `ContractService.search_contract(contract_id: str, query: str) -> list[dict[str, Any]]`
- Preserves: `RAGPipeline.run(...)` and `ContractService.ask(...)` external results.

- [ ] **Step 1: Declare LangGraph and refresh the lock**

Add this direct dependency to `pyproject.toml`:

```toml
"langgraph>=1.2.11",
```

Run:

```bash
uv lock
```

Expected: exit 0 and `uv.lock` contains the resolved LangGraph packages.

- [ ] **Step 2: Write failing tests for evidence-only retrieval**

Add a test that calls `retrieve_evidence()` with the current fake index, reranker, and selector LLM and asserts the literal stage result:

```python
def test_pipeline_retrieves_reranks_and_selects_evidence_without_answer_generation():
    index = FakeIndex([result_for(10, "证据")])

    result = RAGPipeline().retrieve_evidence(
        index,
        "问题",
        reranker=FakeReranker(),
        selector_llm=FakeLLM({"evidence_indices": [10]}),
    )

    assert result["query"] == "问题"
    assert result["selected_indices"] == [10]
    assert [item.node.text for item in result["selected_nodes"]] == ["证据"]
    assert "llm_summary" not in result
```

Name the mutation caught: accidentally calling Answer Generator or omitting one of the existing retrieval stages changes the result or raises due to missing Answer LLM configuration.

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
uv run pytest tests/test_rag_pipeline.py::test_pipeline_retrieves_reranks_and_selects_evidence_without_answer_generation -v
```

Expected: FAIL because `RAGPipeline` has no `retrieve_evidence` method.

- [ ] **Step 4: Implement the minimal evidence-only pipeline and reuse it from `run()`**

Implement this boundary in `app/rag_pipeline.py`:

```python
def retrieve_evidence(
    self,
    index: Any,
    question: str,
    *,
    reranker: Any | None = None,
    selector_llm: Any | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("question must not be empty")
    vector_results, reranked_results = retrieval_evaluation.retrieve_and_rerank(
        index,
        question,
        reranker=reranker,
    )
    selected_indices = retrieval_evaluation.select_evidence(
        question,
        reranked_results,
        llm=selector_llm,
    )
    selected_nodes = retrieval_evaluation.filter_nodes_by_indices(
        reranked_results,
        selected_indices,
    )
    return {
        "query": question,
        "vector_results": vector_results,
        "reranked_results": reranked_results,
        "selected_indices": selected_indices,
        "selected_nodes": selected_nodes,
    }
```

Change `run()` to call this method and then only add `llm_summary` after `generate_answer()`.

- [ ] **Step 5: Run the pipeline tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_rag_pipeline.py -v
```

Expected: all tests pass, including the pre-existing full Answer path.

- [ ] **Step 6: Write failing tests for `ContractService.search_contract()`**

Use a fake pipeline that returns real node-like selected results. Assert the service loads the ready contract index once, calls only `retrieve_evidence`, and returns serialized fields with literal citation values:

```python
self.assertEqual(evidence[0]["source_object_index"], 12)
self.assertEqual(evidence[0]["page_idx"], 4)
self.assertEqual(evidence[0]["node_type"], "image")
```

Also assert a blank query raises `ValueError`, a missing contract raises `ContractNotFoundError`, and a non-ready contract raises `ContractNotReadyError`.

Name the mutation caught: routing through `answer_question()` would call the wrong pipeline method and would make the fake fail.

- [ ] **Step 7: Run the focused service tests and verify RED**

Run:

```bash
uv run pytest tests/test_app_qa.py -k search_contract -v
```

Expected: FAIL because `ContractService.search_contract` does not exist.

- [ ] **Step 8: Implement `search_contract()` without changing `ask()` output**

Add:

```python
def search_contract(self, contract_id: str, query: str) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    contract = self.repository.get(contract_id)
    if contract is None:
        raise ContractNotFoundError(contract_id)
    if contract.status != "ready":
        raise ContractNotReadyError(contract)
    index = self.index_manager.get(contract)
    retrieval = self.rag_pipeline.retrieve_evidence(index, query)
    return [
        serialize_node_result(result)
        for result in retrieval.get("selected_nodes", [])
    ]
```

Import and reuse `serialize_node_result`; do not invoke `answer_question()`.

- [ ] **Step 9: Verify RAG and chat compatibility**

Run:

```bash
uv run pytest tests/test_rag_pipeline.py tests/test_app_qa.py tests/test_evaluation_service.py -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit the RAG boundary**

```bash
git add pyproject.toml uv.lock app/rag_pipeline.py app/service.py tests/test_rag_pipeline.py tests/test_app_qa.py
git commit -m "feat: expose contract rag evidence retrieval"
```

---

### Task 2: Define strict review schemas and prompts

**Files:**
- Create: `app/contract_review/__init__.py`
- Create: `app/contract_review/schemas.py`
- Create: `app/contract_review/prompts.py`
- Create: `tests/test_contract_review_schemas.py`

**Interfaces:**
- Produces: `ReviewItem`, `ReviewItemList`, `Evidence`, `RiskDecision`, `ReviewResult`, `ReviewSummary`
- Produces: `parse_llm_response(response, model_type)` for Pydantic validation.
- Produces: `build_parse_review_rules_prompt(review_rule_text)` and `build_review_item_prompt(item, evidence)`.

- [ ] **Step 1: Write failing strict-schema tests**

Cover these independently derived behaviors:

```python
def test_review_item_list_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="review item ids must be unique"):
        ReviewItemList.model_validate({"review_items": [ITEM, ITEM]})


def test_risk_decision_requires_level_for_risk():
    with pytest.raises(ValidationError, match="risk_level"):
        RiskDecision.model_validate({**DECISION, "risk_status": "risk", "risk_level": None})


def test_insufficient_evidence_requires_needs_review():
    with pytest.raises(ValidationError, match="insufficient"):
        RiskDecision.model_validate(
            {**DECISION, "risk_status": "no_obvious_risk", "evidence_status": "insufficient"}
        )


def test_review_item_rejects_extra_llm_fields():
    with pytest.raises(ValidationError, match="extra"):
        ReviewItem.model_validate({**ITEM, "invented_rule": "行业惯例"})
```

Also test that an Evidence model preserves real table/image extra metadata while requiring RAG-owned `source_object_index`, `node_type`, and `evidence_text`.

- [ ] **Step 2: Run schema tests and verify RED**

Run:

```bash
uv run pytest tests/test_contract_review_schemas.py -v
```

Expected: collection FAIL because `app.contract_review.schemas` does not exist.

- [ ] **Step 3: Implement strict Pydantic models**

Use `ConfigDict(extra="forbid")` for LLM-owned models and `ConfigDict(extra="allow")` only for RAG-owned Evidence so existing text/table/image metadata survives. Use exact literals:

```python
RiskStatus = Literal["risk", "no_obvious_risk", "needs_review"]
RiskLevel = Literal["high", "medium", "low"]
EvidenceStatus = Literal["found", "insufficient"]
```

Use `Field(min_length=1)` and a before validator to strip strings. Use `model_validator(mode="after")` for duplicate IDs and risk/evidence cross-field invariants. Define summary fields as non-negative integers.

- [ ] **Step 4: Implement robust LLM payload parsing**

Support the response shapes already accepted by the project:

```python
def parse_llm_response(response: Any, model_type: type[ModelT]) -> ModelT:
    if isinstance(response, dict):
        payload = response
    else:
        parsed = (getattr(response, "additional_kwargs", {}) or {}).get("parsed")
        payload = parsed if parsed is not None else json.loads(_response_text(response))
    if isinstance(payload, str):
        payload = json.loads(payload)
    return model_type.model_validate(payload)
```

Do not catch `JSONDecodeError` or `ValidationError`; fail-fast semantics require them to surface with node context.

- [ ] **Step 5: Implement prompts without adding review standards**

The parse prompt must quote the complete `review_rule_text` and explicitly forbid legal knowledge, industry conventions, thresholds, and standards absent from the input. The risk prompt must JSON-serialize only `rule_basis`, `review_goal`, and the RAG Evidence and state that no citation fields may be generated.

- [ ] **Step 6: Run schema tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_contract_review_schemas.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit schemas and prompts**

```bash
git add app/contract_review/__init__.py app/contract_review/schemas.py app/contract_review/prompts.py tests/test_contract_review_schemas.py
git commit -m "feat: define contract review schemas and prompts"
```

---

### Task 3: Implement sequential LangGraph nodes and progress events

**Files:**
- Create: `app/contract_review/state.py`
- Create: `app/contract_review/nodes.py`
- Create: `tests/test_contract_review_graph.py`

**Interfaces:**
- Consumes: `ContractService.search_contract(contract_id, query)` from Task 1.
- Consumes: all schemas and prompt builders from Task 2.
- Produces: `ContractReviewState` with `Annotated[list[ReviewResult], operator.add]`.
- Produces: `ContractReviewNodes.parse_review_rules`, `.review_item`, `.aggregate_results`.
- Produces: `ProgressCallback = Callable[[str, dict[str, Any]], None]`.

- [ ] **Step 1: Write failing node tests with specific fakes**

Create a `FakeLLM` that records prompts and returns complete production-shaped JSON responses. Create a `FakeContractSearch` that records `(contract_id, query)` and returns complete Evidence dictionaries.

Write tests that assert:

- parsing emits exactly one `review_items_parsed` event with JSON-serializable item dictionaries;
- one review call uses the current item's literal `retrieval_query`;
- events occur as `review_item_started`, `evidence_retrieved`, `review_item_completed`;
- the returned state update contains only `[new_result]` and `current_item_index + 1`;
- citation metadata equals the fake RAG values, including page and node type;
- empty Evidence overrides an otherwise optimistic model decision to `needs_review / insufficient / null`;
- aggregation returns literal counts for one high risk, one low risk, one no-obvious-risk, and one needs-review result;
- malformed LLM JSON and search exceptions propagate.

Name the key mutations caught: using the wrong item index, replacing the result list, accepting LLM-generated Evidence, or emitting Python model objects instead of serializable event payloads.

- [ ] **Step 2: Run focused node tests and verify RED**

Run:

```bash
uv run pytest tests/test_contract_review_graph.py -k "node or aggregate or progress" -v
```

Expected: collection FAIL because State and nodes do not exist.

- [ ] **Step 3: Implement the LangGraph State reducer**

Define:

```python
class ContractReviewState(TypedDict):
    contract_id: str
    review_rule_text: str
    review_items: list[ReviewItem]
    current_item_index: int
    review_results: Annotated[list[ReviewResult], operator.add]
    summary: ReviewSummary | None
```

- [ ] **Step 4: Implement `ContractReviewNodes` and stable event payloads**

The constructor accepts parse LLM, risk LLM, a callable or service exposing `search_contract`, and an optional progress callback. `_emit()` must call the callback only with dictionaries containing `model_dump(mode="json")` output.

`parse_review_rules()` invokes the parse LLM, validates `ReviewItemList`, emits all items, and returns items plus index zero.

`review_item()` must:

1. get exactly one current item;
2. emit `review_item_started`;
3. call `search_contract()` once;
4. validate each RAG dictionary as Evidence and emit `evidence_retrieved`;
5. invoke and validate `RiskDecision`;
6. force the insufficient result when Evidence is empty;
7. construct ReviewResult with item fields and Python-owned Evidence;
8. emit `review_item_completed`;
9. return one result increment plus the next index.

Wrap parsing and decision validation errors with stage and item context using exception chaining; do not swallow the original exception.

`aggregate_results()` computes counts with comprehensions and emits `review_summary`.

- [ ] **Step 5: Run node tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_contract_review_graph.py -k "node or aggregate or progress" -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit State and nodes**

```bash
git add app/contract_review/state.py app/contract_review/nodes.py tests/test_contract_review_graph.py
git commit -m "feat: add contract review graph nodes"
```

---

### Task 4: Compile the graph and expose `ContractReviewService.run()`

**Files:**
- Create: `app/contract_review/graph.py`
- Create: `app/contract_review/service.py`
- Modify: `app/contract_review/__init__.py`
- Modify: `tests/test_contract_review_graph.py`
- Create: `tests/test_contract_review_service.py`

**Interfaces:**
- Consumes: `ContractReviewNodes` and `ContractReviewState` from Task 3.
- Produces: `build_contract_review_graph(nodes) -> CompiledStateGraph`.
- Produces: `ContractReviewService.run(contract_id: str, review_rule_text: str) -> dict[str, Any]`.
- Produces: `build_default_contract_review_service(*, progress_callback=None) -> ContractReviewService`.

- [ ] **Step 1: Write a failing real-graph loop test**

Use two parsed ReviewItems, two literal fake RAG result sets, and two risk decisions. Invoke the real compiled graph and assert:

```python
assert [result.item_id for result in final_state["review_results"]] == ["item_1", "item_2"]
assert final_state["current_item_index"] == 2
assert final_state["summary"].total_items == 2
assert search.queries == ["付款期限", "违约责任"]
```

Name the mutation caught: a missing reducer or incorrect conditional edge either drops the first result or fails to visit the second item.

- [ ] **Step 2: Run the graph-loop test and verify RED**

Run:

```bash
uv run pytest tests/test_contract_review_graph.py -k real_graph -v
```

Expected: FAIL because `build_contract_review_graph` does not exist.

- [ ] **Step 3: Implement the graph and loop route**

Build:

```python
builder = StateGraph(ContractReviewState)
builder.add_node("parse_review_rules", nodes.parse_review_rules)
builder.add_node("review_item", nodes.review_item)
builder.add_node("aggregate_results", nodes.aggregate_results)
builder.add_edge(START, "parse_review_rules")
builder.add_edge("parse_review_rules", "review_item")
builder.add_conditional_edges(
    "review_item",
    route_after_review,
    {"review_item": "review_item", "aggregate_results": "aggregate_results"},
)
builder.add_edge("aggregate_results", END)
return builder.compile()
```

The route returns `review_item` only while `current_item_index < len(review_items)`.

- [ ] **Step 4: Run the graph-loop test and verify GREEN**

Run:

```bash
uv run pytest tests/test_contract_review_graph.py -k real_graph -v
```

Expected: pass with both results preserved in order.

- [ ] **Step 5: Write failing service preflight tests**

Build a fake contract service with separate counters for `get_contract` and `search_contract`. Assert:

- blank contract ID and rule text fail before graph invocation;
- missing contract raises `ContractNotFoundError`;
- non-ready contract raises `ContractNotReadyError`;
- a ready contract calls `get_contract` once and does not call `search_contract` until the first review node;
- if parse LLM fails, `search_contract` remains uncalled, proving preflight does not trigger RAG;
- success returns only JSON-serializable dictionaries/lists/scalars for items, results, Evidence, and summary.

Name the mutation caught: preflight calling `search_contract()` would increment the RAG counter before parsing and fail the test.

- [ ] **Step 6: Run service tests and verify RED**

Run:

```bash
uv run pytest tests/test_contract_review_service.py -v
```

Expected: collection FAIL because `ContractReviewService` does not exist.

- [ ] **Step 7: Implement service preflight, graph invocation, and serialization**

Use only:

```python
contract = self.contract_service.get_contract(contract_id)
if contract is None:
    raise ContractNotFoundError(contract_id)
if contract.status != "ready":
    raise ContractNotReadyError(contract)
```

Do not call `IndexManager`, `search_contract`, or RAG during preflight. Initialize:

```python
initial_state = {
    "contract_id": contract_id,
    "review_rule_text": review_rule_text,
    "review_items": [],
    "current_item_index": 0,
    "review_results": [],
    "summary": None,
}
```

Invoke the compiled graph and return model-dumped JSON data.

- [ ] **Step 8: Build default strict-schema LLM dependencies**

Construct two `ChatOpenAI` bindings with the project's existing `LLM_MODEL`, `LLM_API_KEY`, and `LLM_BASE_URL`, plus `temperature=0`, timeout 120, `max_retries=0`, and `extra_body={"enable_thinking": False}`. Bind each to:

```python
{
    "type": "json_schema",
    "json_schema": {
        "name": schema_name,
        "strict": True,
        "schema": model_type.model_json_schema(),
    },
}
```

The default service builder may import `app.api.build_default_service` inside the function to reuse existing construction without changing Web startup.

- [ ] **Step 9: Run all contract-review tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_contract_review_schemas.py tests/test_contract_review_graph.py tests/test_contract_review_service.py -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit graph and service**

```bash
git add app/contract_review/__init__.py app/contract_review/graph.py app/contract_review/service.py tests/test_contract_review_graph.py tests/test_contract_review_service.py
git commit -m "feat: run sequential contract risk reviews"
```

---

### Task 5: Add the fixed-input command-line smoke runner

**Files:**
- Create: `scripts/test_contract_review.py`

**Interfaces:**
- Consumes: `build_default_contract_review_service(progress_callback=...)`.
- Produces: a directly executable script with editable `CONTRACT_ID` and `REVIEW_RULE_TEXT` constants.

- [ ] **Step 1: Implement a transport-neutral event formatter**

Define constants using an existing local ready contract and a manual rule text containing a small number of explicit checks. Define:

```python
def print_progress(event: str, payload: dict[str, Any]) -> None:
    headings = {
        "review_items_parsed": "解析出的 ReviewItem",
        "review_item_started": "当前审查项",
        "evidence_retrieved": "RAG 命中的 Evidence",
        "review_item_completed": "本项风险结果",
        "review_summary": "最终汇总",
    }
    print(f"\n=== {headings[event]} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
```

The callback consumes the same event contract a future Web adapter will use; no terminal printing belongs in nodes or the service.

- [ ] **Step 2: Add the executable entry point**

Build the default service, call:

```python
result = service.run(CONTRACT_ID, REVIEW_RULE_TEXT)
```

and print the final returned object only if needed for debugging; avoid duplicating all progress output.

- [ ] **Step 3: Verify the script imports and compiles before external calls**

Run:

```bash
uv run python -m py_compile scripts/test_contract_review.py
```

Expected: exit 0.

- [ ] **Step 4: Commit the command-line runner**

```bash
git add scripts/test_contract_review.py
git commit -m "feat: add contract review smoke script"
```

---

### Task 6: Verify the complete implementation and run the real workflow

**Files:**
- Verify only; modify implementation or tests only if a failure exposes a real defect.

**Interfaces:**
- Verifies all spec requirements and compatibility constraints.

- [ ] **Step 1: Run focused contract-review and RAG tests**

```bash
uv run pytest \
  tests/test_rag_pipeline.py \
  tests/test_app_qa.py \
  tests/test_contract_review_schemas.py \
  tests/test_contract_review_graph.py \
  tests/test_contract_review_service.py \
  -v
```

Expected: zero failures.

- [ ] **Step 2: Run chat and Evaluation regressions**

```bash
uv run pytest \
  tests/test_app_api.py \
  tests/test_evaluation_service.py \
  tests/test_evaluation_metrics.py \
  tests/test_evaluation_recovery.py \
  -v
```

Expected: zero failures and unchanged chat/Evaluation result shapes.

- [ ] **Step 3: Run the complete test suite**

```bash
uv run pytest -q
```

Expected: zero failures.

- [ ] **Step 4: Run static compilation checks**

```bash
uv run python -m compileall -q app scripts/test_contract_review.py
```

Expected: exit 0.

- [ ] **Step 5: Execute the real ready-contract smoke test**

```bash
uv run python scripts/test_contract_review.py
```

Expected: the terminal prints parsed items, the active item, real Evidence, each risk result, and the deterministic summary. If an external LLM, embedding, or rerank endpoint fails, record the precise stage and exception; do not represent the real smoke test as passing.

- [ ] **Step 6: Inspect final scope and diff hygiene**

```bash
git status --short
git diff --check HEAD~4..HEAD
git diff --stat HEAD~4..HEAD
```

Expected: only the approved backend, tests, dependency files, script, and design/plan documentation changed; the unrelated `docs/superpowers/.DS_Store` remains untracked and uncommitted.

- [ ] **Step 7: Commit any verification-only fixes after rerunning affected tests**

If and only if Step 1–5 exposed a defect, commit the tested correction with a narrow message. Otherwise create no empty verification commit.
