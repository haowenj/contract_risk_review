# Contract Review Absence Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic full-contract absence verification branch after exactly two unsuccessful RAG attempts, with auditable keywords, candidates, and `absence_verified` results.

**Architecture:** Refactor the current per-item internal loop into explicit LangGraph nodes, while preserving the existing outer ReviewItem loop and accumulated results reducer. Generate the second retrieval query and high-discrimination scan keywords in one strict LLM response; after the second empty RAG result, scan `merged_content_list.json` with pure Python and either send real candidates to the existing risk LLM or produce a separately prompted absence decision.

**Tech Stack:** Python 3.14, Pydantic v2, LangGraph, LangChain `ChatOpenAI`, pytest, existing MinerU text/table/image searchable-text helpers.

**Spec:** `docs/superpowers/specs/2026-08-20-contract-review-absence-check-design.md`

## Global Constraints

- Work directly on the existing `main` branch as explicitly requested; do not create a worktree.
- Preserve all current uncommitted second-RAG, selector-empty, Rerank Debug, and `.DS_Store`-ignore behavior before starting absence-check changes.
- Execute at most two RAG calls per ReviewItem; `absence_check` must not call embedding, Vector, Rerank, Evidence Selector, MinerU, OCR, VL, or PDF parsing.
- Read only the existing `<contract.storage_dir>/merged_content_list.json` for full-text verification.
- Reuse `table_to_searchable_text()` and `image_to_searchable_text()`; do not invent parallel table/image extraction logic.
- Keep chat, Evaluation, ingestion, text/table/image RAG, and existing Web behavior unchanged.
- Evidence-owned citation fields must always come from RAG results or the real merged-content array index, never from an LLM.
- `absence_check` metadata is program-owned; the LLM must not generate or modify it.
- Do not add Web UI, persistence, reports, background tasks, third retrievals, BM25, new indexes, new embeddings, multi-Agent runtime behavior, or human-in-the-loop.
- Use `uv run --with pytest python -m pytest <test-paths>`; the project does not include pytest as a normal runtime dependency.
- Do not push to a remote unless the user explicitly requests it after implementation.

---

### Task 0: Checkpoint the Approved Second-RAG Baseline

**Files:**
- Modify/commit existing working-tree changes only: `app/contract_review/nodes.py`, `app/contract_review/prompts.py`, `app/contract_review/schemas.py`, `app/contract_review/service.py`, `app/rag_pipeline.py`, `app/service.py`, `retrieval_evaluation.py`, `scripts/test_contract_review.py`, `tests/test_app_qa.py`, `tests/test_contract_review_graph.py`, `tests/test_contract_review_schemas.py`, `tests/test_contract_review_service.py`, `tests/test_rag_pipeline.py`

**Interfaces:**
- Consumes: the already approved and tested second-RAG plus Rerank Debug implementation in the dirty working tree.
- Produces: a clean Git baseline on `main` before absence-check TDD begins.

- [ ] **Step 1: Verify the current baseline before committing**

Run:

```bash
uv run --with pytest python -m pytest -q
uv run python -m compileall -q app scripts tests retrieval_evaluation.py
git diff --check
```

Expected: `179 passed`, `12 subtests passed`, compile exit 0, and no whitespace errors. The existing Starlette/httpx deprecation warning is allowed.

- [ ] **Step 2: Review the exact baseline scope**

Run:

```bash
git status --short
git diff --stat
git diff -- app/contract_review app/rag_pipeline.py app/service.py retrieval_evaluation.py scripts/test_contract_review.py tests/test_app_qa.py tests/test_contract_review_graph.py tests/test_contract_review_schemas.py tests/test_contract_review_service.py tests/test_rag_pipeline.py
```

Expected: only the approved second retrieval, explicit empty selection, Rerank Top3 Debug, and their tests are present; the absence-check plan and spec commits are already separate.

- [ ] **Step 3: Commit the approved baseline**

```bash
git add app/contract_review/nodes.py app/contract_review/prompts.py app/contract_review/schemas.py app/contract_review/service.py app/rag_pipeline.py app/service.py retrieval_evaluation.py scripts/test_contract_review.py tests/test_app_qa.py tests/test_contract_review_graph.py tests/test_contract_review_schemas.py tests/test_contract_review_service.py tests/test_rag_pipeline.py
git commit -m "feat: retry empty contract review retrieval"
```

Expected: one commit containing only the approved working-tree implementation; do not push.

---

### Task 1: Extend Strict Schemas and Prompts

**Files:**
- Modify: `app/contract_review/schemas.py`
- Modify: `app/contract_review/prompts.py`
- Modify: `tests/test_contract_review_schemas.py`
- Create: `tests/test_contract_review_prompts.py`

**Interfaces:**
- Consumes: existing `StrictModel`, `RetrievalQueryRewrite`, `RiskDecision`, `ReviewResult`, and prompt builders.
- Produces: `RetrievalQueryRewrite.keywords`, `EvidenceStatus="absence_verified"`, `AbsenceCheckMetadata`, `ReviewResult.absence_check`, and `build_absence_result_prompt()`.

- [ ] **Step 1: Write failing schema tests for keyword cleanup and absence metadata**

Add tests with literal expectations:

```python
def test_retrieval_query_rewrite_normalizes_and_deduplicates_keywords():
    rewrite = RetrievalQueryRewrite.model_validate({
        "retrieval_query": "检索乙方分包、转包和第三方履约限制",
        "reason": "覆盖同义表达",
        "keywords": [" 分包 ", "转包", "ＦＥＮＢＡＯ", "fenbao", "", "转委托"],
    })
    assert rewrite.keywords == ["分包", "转包", "ＦＥＮＢＡＯ", "转委托"]


def test_retrieval_query_rewrite_requires_at_least_one_usable_keyword():
    with pytest.raises(ValidationError):
        RetrievalQueryRewrite.model_validate({
            "retrieval_query": "第二次查询",
            "reason": "原因",
            "keywords": [" ", "\n"],
        })


def test_absence_verified_result_requires_empty_evidence_and_zero_candidates():
    result = ReviewResult.model_validate({
        "item_id": "item_1",
        "item_name": "分包限制",
        "risk_status": "risk",
        "risk_level": "medium",
        "evidence_status": "absence_verified",
        "finding": "基于当前合同全文解析结果未发现相关条款。",
        "risk_description": "规范要求存在该限制。",
        "suggestion": "建议补充明确条款。",
        "evidence": [],
        "absence_check": {
            "keywords": ["分包", "转包"],
            "candidate_count": 0,
        },
    })
    assert result.absence_check.candidate_count == 0


@pytest.mark.parametrize(
    ("evidence", "absence_check"),
    [
        ([EVIDENCE], {"keywords": ["分包"], "candidate_count": 0}),
        ([], None),
        ([], {"keywords": ["分包"], "candidate_count": 1}),
    ],
)
def test_absence_verified_rejects_inconsistent_audit_data(evidence, absence_check):
    with pytest.raises(ValidationError):
        ReviewResult.model_validate({
            "item_id": "item_1",
            "item_name": "分包限制",
            "risk_status": "risk",
            "risk_level": "medium",
            "evidence_status": "absence_verified",
            "finding": "基于当前合同全文解析结果未发现相关条款。",
            "risk_description": "规范要求存在该限制。",
            "suggestion": "建议补充明确条款。",
            "evidence": evidence,
            "absence_check": absence_check,
        })
```

Production mutation caught: missing keyword cleanup/dedup, missing `absence_verified`, or accepting unaudited absence results.

- [ ] **Step 2: Run schema tests and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_schemas.py -v
```

Expected: FAIL because `keywords`, `AbsenceCheckMetadata`, and `absence_verified` validation do not exist.

- [ ] **Step 3: Implement the schema changes**

Add the following shapes and validators:

```python
import unicodedata
from pydantic import field_validator

EvidenceStatus = Literal["found", "insufficient", "absence_verified"]


def _keyword_key(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _clean_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("keywords must be a list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError("keywords must contain strings")
        keyword = raw.strip()
        key = _keyword_key(keyword)
        if key and key not in seen:
            cleaned.append(keyword)
            seen.add(key)
    if not cleaned:
        raise ValueError("keywords must contain a usable keyword")
    return cleaned


class RetrievalQueryRewrite(StrictModel):
    retrieval_query: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)

    @field_validator("keywords", mode="before")
    @classmethod
    def clean_keywords(cls, value: Any) -> list[str]:
        return _clean_keywords(value)


class AbsenceCheckMetadata(StrictModel):
    keywords: list[str] = Field(min_length=1)
    candidate_count: int = Field(ge=0)

    @field_validator("keywords", mode="before")
    @classmethod
    def clean_keywords(cls, value: Any) -> list[str]:
        return _clean_keywords(value)


class ReviewResult(RiskDecision):
    item_id: str = Field(min_length=1)
    item_name: str = Field(min_length=1)
    evidence: list[Evidence]
    absence_check: AbsenceCheckMetadata | None = None

    @model_validator(mode="after")
    def validate_absence_audit(self) -> ReviewResult:
        if self.evidence_status == "absence_verified":
            if self.evidence:
                raise ValueError("absence_verified requires empty evidence")
            if self.absence_check is None or self.absence_check.candidate_count != 0:
                raise ValueError("absence_verified requires a zero-candidate absence_check")
        return self
```

Ensure ordinary RAG results still accept `absence_check=None`.

- [ ] **Step 4: Write failing prompt tests for business-specific keywords and absence wording**

Create `tests/test_contract_review_prompts.py`:

```python
def test_rewrite_prompt_requires_business_specific_scan_keywords():
    prompt = build_retrieval_query_rewrite_prompt(
        ReviewItem.model_validate(ITEM),
        attempted_queries=[ITEM["retrieval_query"]],
        evidence=[],
        decision=None,
    )
    assert '"keywords"' in prompt
    assert "核心术语或具有业务区分度的短语" in prompt
    assert "不得单独输出“同意”“批准”“许可”“责任”“合同”" in prompt


def test_absence_result_prompt_limits_claim_to_parsed_content():
    prompt = build_absence_result_prompt(
        ReviewItem.model_validate(ITEM),
        keywords=["分包", "转包", "转委托"],
    )
    assert "基于当前合同全文解析结果未发现" in prompt
    assert "合同肯定没有" in prompt
    assert "禁止" in prompt
    assert "absence_verified" in prompt
```

Production mutation caught: generic keywords being encouraged or absolute absence wording being allowed.

- [ ] **Step 5: Run prompt tests and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_prompts.py -v
```

Expected: FAIL because rewrite prompt has no keywords contract and `build_absence_result_prompt` is missing.

- [ ] **Step 6: Update both prompts**

Extend the rewrite JSON protocol to require `keywords`, including the exact restrictions:

```text
keywords 应优先生成能够识别当前审查主题的核心术语或具有业务区分度的短语。
不得单独输出“同意”“批准”“许可”“责任”“合同”等泛化词；只有与当前主题组合成明确短语时才能使用。
```

Add:

```python
def build_absence_result_prompt(
    item: ReviewItem,
    *,
    keywords: list[str],
) -> str:
    return f"""你需要根据风险规范和确定性全文缺失核验结果判断风险。

确定性事实：
1. 已执行两次语义检索，均未获得有效合同 Evidence。
2. 程序已使用以下关键词扫描 merged_content_list.json 的完整可检索内容：
{json.dumps(keywords, ensure_ascii=False, indent=2)}
3. 全文扫描候选数量为 0。

要求：
1. 只依据 rule_basis 和 review_goal 判断缺失是否构成风险及其等级，不得增加规范外标准。
2. finding 必须使用“基于当前合同全文解析结果未发现……”的限定措辞。
3. 禁止使用“合同肯定没有……”等绝对表述。
4. evidence_status 必须为 absence_verified。
5. 不得输出 evidence、absence_check 或任何引用字段；这些字段由程序附加。
6. 顶层只能包含 risk_status、risk_level、evidence_status、finding、risk_description、suggestion。

JSON 协议：
{{
  "risk_status": "risk | no_obvious_risk | needs_review",
  "risk_level": "high | medium | low | null",
  "evidence_status": "absence_verified",
  "finding": "限定为当前完整解析结果的审查发现",
  "risk_description": "严格依据输入规范的风险说明",
  "suggestion": "补充条款或人工核对建议"
}}

rule_basis：
{item.rule_basis}

review_goal：
{item.review_goal}
"""
```

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_schemas.py tests/test_contract_review_prompts.py -v
```

Expected: PASS.

Commit:

```bash
git add app/contract_review/schemas.py app/contract_review/prompts.py tests/test_contract_review_schemas.py tests/test_contract_review_prompts.py
git commit -m "feat: define absence verification schemas"
```

---

### Task 2: Add Deterministic Full-Content Scanner and Loader

**Files:**
- Create: `app/contract_review/absence.py`
- Modify: `app/service.py`
- Create: `tests/test_contract_review_absence.py`
- Modify: `tests/test_app_qa.py`

**Interfaces:**
- Consumes: `table_to_searchable_text(table)`, `image_to_searchable_text(image)`, and `ContractService.repository`.
- Produces: `AbsenceScanResult`, `scan_source_objects(source_objects, keywords, limit=20)`, and `ContractService.load_contract_content_objects(contract_id)`.

- [ ] **Step 1: Write failing scanner tests covering text, table, and image**

Create fixtures with array positions 0 through 4 and assert real positions, exact matches, and reusable searchable text:

```python
def test_scan_source_objects_covers_text_table_and_image_with_real_indices():
    source_objects = [
        {"type": "text", "text": "普通标题", "page_idx": 0},
        {"type": "text", "text": "未经甲方书面同意不得转委托", "page_idx": 1},
        {
            "type": "table",
            "table_caption": ["分包审批"],
            "table_body": "<table><tr><td>第三方履行须书面批准</td></tr></table>",
            "table_footnote": ["禁止转包"],
            "page_idx": 2,
        },
        {
            "type": "image",
            "image_type": "general",
            "structured_data": {
                "content_description": "外包限制告知书",
                "visible_text": "委托第三方履行",
            },
            "page_idx": 3,
        },
        {"type": "aside_text", "text": "分包", "page_idx": 4},
    ]

    result = scan_source_objects(
        source_objects,
        ["转委托", "分包审批", "第三方履行", "禁止转包", "外包限制"],
    )

    assert result.candidate_count == 3
    assert [item["source_object_index"] for item in result.candidates] == [2, 3, 1]
    assert result.candidates[0]["node_type"] == "table"
    assert result.candidates[0]["matched_keywords"] == [
        "分包审批", "第三方履行", "禁止转包"
    ]
    assert result.candidates[2]["node_type"] == "image"
```

- [ ] **Step 2: Write failing normalization, deterministic-match, ordering, and limit tests**

```python
def test_scan_normalizes_nfkc_case_and_whitespace_without_fuzzy_matching():
    objects = [
        {"type": "text", "text": "ＡＢＣ\n第三方   履行", "page_idx": 0},
        {"type": "text", "text": "转委拖", "page_idx": 1},
    ]
    result = scan_source_objects(objects, ["abc", "第三方履行", "转委托"])
    assert result.candidate_count == 1
    assert result.candidates[0]["source_object_index"] == 0
    assert result.candidates[0]["matched_keywords"] == ["abc", "第三方履行"]


def test_scan_reports_total_count_before_twenty_candidate_limit():
    objects = [
        {"type": "text", "text": f"分包限制 {index}", "page_idx": index}
        for index in range(25)
    ]
    result = scan_source_objects(objects, ["分包限制"], limit=20)
    assert result.candidate_count == 25
    assert len(result.candidates) == 20
    assert [item["source_object_index"] for item in result.candidates] == list(range(20))
```

Production mutation caught: fuzzy matching, wrong source index, missing table/image fields, wrong ordering, or post-limit counts.

- [ ] **Step 3: Run scanner tests and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_absence.py -v
```

Expected: collection FAIL because `app.contract_review.absence` does not exist.

- [ ] **Step 4: Implement the deterministic scanner**

Create:

```python
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

from image_searchable_text import image_to_searchable_text
from table_searchable_text import table_to_searchable_text

MAX_ABSENCE_CANDIDATES = 20


@dataclass(frozen=True)
class AbsenceScanResult:
    candidates: list[dict[str, Any]]
    candidate_count: int


def normalize_scan_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _object_text(source_object: dict[str, Any]) -> tuple[str, str] | None:
    node_type = source_object.get("type")
    if node_type == "text":
        text = source_object.get("text")
    elif node_type == "table":
        text = table_to_searchable_text(source_object)
    elif node_type == "image":
        text = image_to_searchable_text(source_object)
    else:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return node_type, text


def scan_source_objects(
    source_objects: list[dict[str, Any]],
    keywords: list[str],
    *,
    limit: int = MAX_ABSENCE_CANDIDATES,
) -> AbsenceScanResult:
    if limit < 1:
        raise ValueError("limit must be positive")
    normalized_keywords = [
        (keyword, normalize_scan_text(keyword)) for keyword in keywords
    ]
    matches: list[dict[str, Any]] = []
    for source_index, source_object in enumerate(source_objects):
        if not isinstance(source_object, dict):
            raise ValueError("merged content objects must be JSON objects")
        extracted = _object_text(source_object)
        if extracted is None:
            continue
        node_type, evidence_text = extracted
        normalized_text = normalize_scan_text(evidence_text)
        matched_keywords = [
            keyword
            for keyword, normalized in normalized_keywords
            if normalized and normalized in normalized_text
        ]
        if not matched_keywords:
            continue
        matches.append({
            "source_object_index": source_index,
            "page_idx": source_object.get("page_idx"),
            "node_type": node_type,
            "matched_keywords": matched_keywords,
            "evidence_text": evidence_text,
            "text": evidence_text,
        })
    matches.sort(key=lambda value: (
        -len(value["matched_keywords"]),
        -max(len(normalize_scan_text(item)) for item in value["matched_keywords"]),
        value["source_object_index"],
    ))
    return AbsenceScanResult(
        candidates=matches[:limit],
        candidate_count=len(matches),
    )
```

- [ ] **Step 5: Write failing ContractService loader tests**

Extend `ContractSearchTest` with a ready contract whose storage directory contains a literal JSON list:

```python
def test_load_contract_content_objects_reads_ready_merged_content(self):
    with TemporaryDirectory() as temp_dir:
        service, _, _ = self._build_service(Path(temp_dir), [])
        storage_dir = Path(temp_dir) / "contract"
        storage_dir.mkdir()
        expected = [{"type": "text", "text": "条款", "page_idx": 0}]
        (storage_dir / "merged_content_list.json").write_text(
            json.dumps(expected, ensure_ascii=False), encoding="utf-8"
        )
        contract = service.repository.create("contract.pdf", storage_dir)
        service.repository.update_status(contract.contract_id, "ready")
        assert service.load_contract_content_objects(contract.contract_id) == expected


def test_load_contract_content_objects_rejects_missing_or_non_ready_contract(self):
    with TemporaryDirectory() as temp_dir:
        service, _, _ = self._build_service(Path(temp_dir), [])
        with self.assertRaises(ContractNotFoundError):
            service.load_contract_content_objects("missing")
        contract = service.repository.create(
            "contract.pdf", Path(temp_dir) / "queued-contract"
        )
        with self.assertRaises(ContractNotReadyError):
            service.load_contract_content_objects(contract.contract_id)


def test_load_contract_content_objects_rejects_missing_or_invalid_content_file(self):
    invalid_payloads = [
        "not-json",
        json.dumps({"type": "text"}),
        json.dumps([{"type": "text", "text": "条款"}, "invalid-member"]),
    ]
    with TemporaryDirectory() as temp_dir:
        service, _, _ = self._build_service(Path(temp_dir), [])
        missing_dir = Path(temp_dir) / "missing-file"
        missing_dir.mkdir()
        missing = service.repository.create("missing.pdf", missing_dir)
        service.repository.update_status(missing.contract_id, "ready")
        with self.assertRaises(FileNotFoundError):
            service.load_contract_content_objects(missing.contract_id)

        for index, payload in enumerate(invalid_payloads):
            storage_dir = Path(temp_dir) / f"invalid-{index}"
            storage_dir.mkdir()
            (storage_dir / "merged_content_list.json").write_text(
                payload, encoding="utf-8"
            )
            contract = service.repository.create(f"invalid-{index}.pdf", storage_dir)
            service.repository.update_status(contract.contract_id, "ready")
            with self.subTest(payload=payload), self.assertRaises(
                (json.JSONDecodeError, ValueError)
            ):
                service.load_contract_content_objects(contract.contract_id)
```

- [ ] **Step 6: Run loader tests and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_app_qa.py::ContractSearchTest -v
```

Expected: FAIL because `load_contract_content_objects` is missing.

- [ ] **Step 7: Implement the loader without invoking RAG**

Add to `ContractService`:

```python
def load_contract_content_objects(
    self,
    contract_id: str,
) -> list[dict[str, Any]]:
    contract = self.repository.get(contract_id)
    if contract is None:
        raise ContractNotFoundError(contract_id)
    if contract.status != "ready":
        raise ContractNotReadyError(contract)
    source_path = Path(contract.storage_dir) / "merged_content_list.json"
    with source_path.open("r", encoding="utf-8") as source_file:
        payload = json.load(source_file)
    if not isinstance(payload, list):
        raise ValueError("merged_content_list.json must contain a JSON array")
    if any(not isinstance(value, dict) for value in payload):
        raise ValueError("merged content objects must be JSON objects")
    return payload
```

Import `json`; do not touch the index manager or RAG pipeline.

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_absence.py tests/test_app_qa.py::ContractSearchTest -v
```

Expected: PASS.

Commit:

```bash
git add app/contract_review/absence.py app/service.py tests/test_contract_review_absence.py tests/test_app_qa.py
git commit -m "feat: scan parsed contract content for clauses"
```

---

### Task 3: Refactor the Existing Two-Attempt Review into Explicit Graph Nodes

**Files:**
- Modify: `app/contract_review/state.py`
- Modify: `app/contract_review/nodes.py`
- Modify: `app/contract_review/graph.py`
- Modify: `app/contract_review/service.py`
- Modify: `tests/test_contract_review_graph.py`
- Modify: `tests/test_contract_review_service.py`

**Interfaces:**
- Consumes: the existing `search_contract(contract_id, query, *, debug_callback=None)`, `RetrievalQueryRewrite.keywords`, and `review_results` reducer.
- Produces: explicit nodes `prepare_review_item`, `retrieve_evidence`, `rewrite_query`, `risk_decision`, `insufficient_result`, and `finalize_review_item`, plus state-only route functions.

- [ ] **Step 1: Rewrite graph tests first around explicit state transitions**

Add focused tests that directly exercise each node and one compiled-graph sequence:

```python
def test_prepare_review_item_resets_per_item_state():
    update = nodes.prepare_review_item(initial_state(
        review_items=[ReviewItem.model_validate(ITEMS_PAYLOAD["review_items"][0])],
        retrieval_attempt=2,
        current_retrieval_query="旧查询",
        retrieved_evidence=[Evidence.model_validate(EVIDENCE)],
        absence_keywords=["旧关键词"],
        absence_candidates=[Evidence.model_validate(EVIDENCE)],
        absence_candidate_count=1,
        current_decision=RiskDecision.model_validate(RISK_DECISION),
    ))
    assert update == {
        "retrieval_attempt": 1,
        "current_retrieval_query": "合同约定的付款期限是多久",
        "retrieved_evidence": [],
        "absence_keywords": [],
        "absence_candidates": [],
        "absence_candidate_count": None,
        "current_decision": None,
    }


def test_route_after_retrieve_never_creates_a_third_rag_attempt():
    assert route_after_retrieve(initial_state(
        retrieval_attempt=1, retrieved_evidence=[]
    )) == "rewrite_query"
    assert route_after_retrieve(initial_state(
        retrieval_attempt=2, retrieved_evidence=[]
    )) == "insufficient_result"
    assert route_after_retrieve(initial_state(
        retrieval_attempt=2,
        retrieved_evidence=[Evidence.model_validate(EVIDENCE)],
    )) == "risk_decision"
```

Update the existing compiled graph test so it asserts exactly two `search_contract` calls for an empty/empty item, no third query, one accumulated result, and continuation to the next ReviewItem.

- [ ] **Step 2: Run graph tests and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_graph.py tests/test_contract_review_service.py -v
```

Expected: FAIL because the explicit nodes, state fields, and routes do not exist.

- [ ] **Step 3: Extend `ContractReviewState` and service initial state**

Use:

```python
class ContractReviewState(TypedDict):
    contract_id: str
    review_rule_text: str
    review_items: list[ReviewItem]
    current_item_index: int
    review_results: Annotated[list[ReviewResult], operator.add]
    summary: ReviewSummary | None
    retrieval_attempt: int
    current_retrieval_query: str
    retrieved_evidence: list[Evidence]
    absence_keywords: list[str]
    absence_candidates: list[Evidence]
    absence_candidate_count: int | None
    current_decision: RiskDecision | None
```

Initialize all fields in `ContractReviewService.run()` so graph invocations never depend on missing keys.

- [ ] **Step 4: Split the current `review_item()` loop into explicit nodes**

Implement these responsibilities without absence scanning yet:

```python
def prepare_review_item(self, state):
    item = state["review_items"][state["current_item_index"]]
    self._emit("review_item_started", {
        "current_item_index": state["current_item_index"],
        "item": item.model_dump(mode="json"),
    })
    return {
        "retrieval_attempt": 1,
        "current_retrieval_query": item.retrieval_query,
        "retrieved_evidence": [],
        "absence_keywords": [],
        "absence_candidates": [],
        "absence_candidate_count": None,
        "current_decision": None,
    }


def retrieve_evidence(self, state):
    item = state["review_items"][state["current_item_index"]]
    rerank_top3_debug = []
    raw_evidence = self.contract_service.search_contract(
        state["contract_id"],
        state["current_retrieval_query"],
        debug_callback=rerank_top3_debug.extend,
    )
    current = [Evidence.model_validate(value) for value in raw_evidence]
    merged = list(state["retrieved_evidence"])
    seen = {value.source_object_index for value in merged}
    for value in current:
        if value.source_object_index not in seen:
            merged.append(value)
            seen.add(value.source_object_index)
    self._emit("evidence_retrieved", {
        "item_id": item.id,
        "attempt": state["retrieval_attempt"],
        "retrieval_query": state["current_retrieval_query"],
        "evidence": [value.model_dump(mode="json") for value in current],
    })
    if not current:
        self._emit("empty_evidence_rerank_debug", {
            "item_id": item.id,
            "attempt": state["retrieval_attempt"],
            "retrieval_query": state["current_retrieval_query"],
            "rerank_top3": rerank_top3_debug,
        })
    return {"retrieved_evidence": merged}


def rewrite_query(self, state):
    item = state["review_items"][state["current_item_index"]]
    response = self.query_rewrite_llm.invoke(
        build_retrieval_query_rewrite_prompt(
            item,
            attempted_queries=[state["current_retrieval_query"]],
            evidence=state["retrieved_evidence"],
            decision=state["current_decision"],
        )
    )
    rewrite = parse_llm_response(response, RetrievalQueryRewrite)
    if rewrite.retrieval_query == state["current_retrieval_query"]:
        raise ValueError("rewritten retrieval_query must differ from attempted query")
    self._emit("retrieval_query_rewritten", {
        "item_id": item.id,
        "next_attempt": 2,
        "previous_query": state["current_retrieval_query"],
        "retrieval_query": rewrite.retrieval_query,
        "reason": rewrite.reason,
    })
    return {
        "retrieval_attempt": 2,
        "current_retrieval_query": rewrite.retrieval_query,
        "absence_keywords": rewrite.keywords,
    }


def risk_decision(self, state):
    item = state["review_items"][state["current_item_index"]]
    evidence = state["absence_candidates"] or state["retrieved_evidence"]
    response = self.review_llm.invoke(build_review_item_prompt(item, evidence))
    return {"current_decision": parse_llm_response(response, RiskDecision)}


def insufficient_result(self, state):
    return {"current_decision": RiskDecision(
        risk_status="needs_review",
        risk_level=None,
        evidence_status="insufficient",
        finding="两次检索均未获得足以支持判断的合同证据。",
        risk_description="证据不足，无法可靠判断该审查项是否存在风险。",
        suggestion="请人工核对合同全文及相关附件后再作判断。",
    )}


def finalize_review_item(self, state):
    item = state["review_items"][state["current_item_index"]]
    evidence = state["absence_candidates"] or state["retrieved_evidence"]
    result = ReviewResult(
        item_id=item.id,
        item_name=item.name,
        evidence=evidence,
        absence_check=None,
        **state["current_decision"].model_dump(),
    )
    self._emit("review_item_completed", {"result": result.model_dump(mode="json")})
    return {
        "review_results": [result],
        "current_item_index": state["current_item_index"] + 1,
    }
```

Keep the stage-specific `RuntimeError` wrapping and original exception causes for every node.

- [ ] **Step 5: Implement state-only route functions and temporary two-RAG graph**

Use:

```python
def route_after_retrieve(state):
    if state["retrieved_evidence"]:
        return "risk_decision"
    if state["retrieval_attempt"] < 2:
        return "rewrite_query"
    return "insufficient_result"


def route_after_risk_decision(state):
    decision = state["current_decision"]
    if decision.evidence_status == "insufficient" and state["retrieval_attempt"] < 2:
        return "rewrite_query"
    return "finalize_review_item"


def route_after_finalize(state):
    if state["current_item_index"] < len(state["review_items"]):
        return "prepare_review_item"
    return "aggregate_results"
```

Wire `rewrite_query -> retrieve_evidence`, `insufficient_result -> finalize_review_item`, and preserve outer item looping. This task intentionally preserves the existing final `needs_review` behavior; Task 5 replaces only the second-empty route.

- [ ] **Step 6: Run graph/service tests and commit**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_graph.py tests/test_contract_review_service.py -v
```

Expected: PASS with exact two-attempt behavior and accumulated ReviewResults.

Commit:

```bash
git add app/contract_review/state.py app/contract_review/nodes.py app/contract_review/graph.py app/contract_review/service.py tests/test_contract_review_graph.py tests/test_contract_review_service.py
git commit -m "refactor: expose contract review graph routing"
```

---

### Task 4: Add the Absence Check Branch with Real Candidates

**Files:**
- Modify: `app/contract_review/nodes.py`
- Modify: `app/contract_review/graph.py`
- Modify: `tests/test_contract_review_graph.py`
- Modify: `tests/test_contract_review_service.py`

**Interfaces:**
- Consumes: `scan_source_objects()`, `ContractService.load_contract_content_objects()`, stored `absence_keywords`, and explicit Task 3 routes.
- Produces: `ContractReviewNodes.absence_check()`, `route_after_absence_check()`, and candidate-to-risk-decision flow.

- [ ] **Step 1: Write a failing graph test for an absence candidate**

First extend the existing graph-test fake without changing its RAG behavior:

```python
class FakeContractService:
    def __init__(self, *evidence_sets, source_objects=None):
        self.evidence_sets = list(evidence_sets)
        self.source_objects = [] if source_objects is None else source_objects
        self.searches = []
        self.content_loads = []

    def search_contract(self, contract_id, query, *, debug_callback=None):
        self.searches.append((contract_id, query))
        evidence = self.evidence_sets.pop(0)
        if isinstance(evidence, Exception):
            raise evidence
        if not evidence and debug_callback is not None:
            debug_callback(RERANK_TOP3_DEBUG)
        return evidence

    def load_contract_content_objects(self, contract_id):
        self.content_loads.append(contract_id)
        if isinstance(self.source_objects, Exception):
            raise self.source_objects
        return self.source_objects
```

Then use an empty first and second RAG response, a rewrite containing keywords, and merged content containing a matching text object:

```python
def test_graph_scans_after_two_empty_rag_results_and_reviews_real_candidate():
    contract_service = FakeContractService(
        [],
        [],
        source_objects=[
            {"type": "text", "text": "标题", "page_idx": 0},
            {
                "type": "text",
                "text": "未经甲方书面同意，乙方不得委托第三方履行合同义务。",
                "page_idx": 4,
            },
        ],
    )
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(ONE_ITEM_PAYLOAD),
        query_rewrite_llm=FakeLLM({
            **QUERY_REWRITE,
            "keywords": ["分包", "转包", "委托第三方"],
        }),
        review_llm=FakeLLM(RISK_DECISION),
        contract_service=contract_service,
    )
    final_state = build_contract_review_graph(nodes).invoke(initial_state())
    result = final_state["review_results"][0]
    assert len(contract_service.searches) == 2
    assert contract_service.content_loads == ["contract-1"]
    assert [item.source_object_index for item in result.evidence] == [1]
    assert result.evidence[0].matched_keywords == ["委托第三方"]
    assert result.absence_check.model_dump() == {
        "keywords": ["分包", "转包", "委托第三方"],
        "candidate_count": 1,
    }
```

Also assert the review LLM prompt contains the candidate text but not the Rerank Top3 Debug text.

- [ ] **Step 2: Run the candidate test and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_graph.py -k absence_candidate -v
```

Expected: FAIL because the second-empty route still selects `insufficient_result`.

- [ ] **Step 3: Implement `absence_check()` and progress events**

Use:

```python
def absence_check(self, state):
    item = state["review_items"][state["current_item_index"]]
    self._emit("absence_check_started", {
        "item_id": item.id,
        "retrieval_attempt": state["retrieval_attempt"],
    })
    self._emit("absence_keywords_generated", {
        "item_id": item.id,
        "keywords": state["absence_keywords"],
    })
    source_objects = self.contract_service.load_contract_content_objects(
        state["contract_id"]
    )
    scan = scan_source_objects(source_objects, state["absence_keywords"])
    candidates = [Evidence.model_validate(value) for value in scan.candidates]
    self._emit("absence_candidates_found", {
        "item_id": item.id,
        "candidate_count": scan.candidate_count,
        "candidates": [value.model_dump(mode="json") for value in candidates],
    })
    return {
        "absence_candidates": candidates,
        "absence_candidate_count": scan.candidate_count,
    }
```

Wrap failures as `RuntimeError(f"absence_check {item.id} failed")` with `from exc`.

- [ ] **Step 4: Replace the second-empty route and add the candidate route**

Change:

```python
def route_after_retrieve(state):
    if state["retrieved_evidence"]:
        return "risk_decision"
    if state["retrieval_attempt"] < 2:
        return "rewrite_query"
    return "absence_check"


def route_after_absence_check(state):
    if state["absence_candidates"]:
        return "risk_decision"
    return "insufficient_result"
```

Register `absence_check` and its conditional edge. In this task, the no-candidate branch deliberately retains the prior `insufficient_result`; Task 5 first adds a failing `absence_verified` test and only then replaces that branch with `absence_result`.

- [ ] **Step 5: Attach program-owned audit metadata in finalization**

Use the `None` versus integer distinction:

```python
absence_check = None
if state["absence_candidate_count"] is not None:
    absence_check = AbsenceCheckMetadata(
        keywords=state["absence_keywords"],
        candidate_count=state["absence_candidate_count"],
    )
```

Pass that value to `ReviewResult`. Do not include `absence_check` in any risk LLM prompt.

- [ ] **Step 6: Run candidate-path tests and commit**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_graph.py -k 'absence_candidate or uses_current_query or real_graph' -v
```

Expected: PASS; exactly two RAG calls and one content load.

Commit:

```bash
git add app/contract_review/nodes.py app/contract_review/graph.py tests/test_contract_review_graph.py tests/test_contract_review_service.py
git commit -m "feat: route empty reviews through full-text scan"
```

---

### Task 5: Produce Audited `absence_verified` Results When No Candidate Exists

**Files:**
- Modify: `app/contract_review/nodes.py`
- Modify: `app/contract_review/graph.py`
- Modify: `tests/test_contract_review_graph.py`
- Modify: `tests/test_contract_review_service.py`

**Interfaces:**
- Consumes: `build_absence_result_prompt()`, zero-candidate Task 4 state, and existing `review_llm` structured binding.
- Produces: `ContractReviewNodes.absence_result()` and `absence_confirmed` progress event.

- [ ] **Step 1: Write a failing zero-candidate end-to-end graph test**

```python
ABSENCE_DECISION = {
    "risk_status": "risk",
    "risk_level": "medium",
    "evidence_status": "absence_verified",
    "finding": "基于当前合同全文解析结果，未发现明确限制乙方分包或转包的条款。",
    "risk_description": "两次语义检索及全文关键词核验均未发现对应约定。",
    "suggestion": "建议补充未经书面同意不得分包或转包的明确条款。",
}


def test_graph_returns_audited_absence_verified_after_two_empty_rag_results():
    events = []
    contract_service = FakeContractService([], [], source_objects=[
        {"type": "text", "text": "付款与验收条款", "page_idx": 1},
    ])
    nodes = ContractReviewNodes(
        parse_llm=FakeLLM(ONE_ITEM_PAYLOAD),
        query_rewrite_llm=FakeLLM({
            **QUERY_REWRITE,
            "keywords": ["分包", "转包", "转委托", "委托第三方"],
        }),
        review_llm=FakeLLM(ABSENCE_DECISION),
        contract_service=contract_service,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )
    final_state = build_contract_review_graph(nodes).invoke(initial_state())
    result = final_state["review_results"][0]
    assert len(contract_service.searches) == 2
    assert result.risk_status == "risk"
    assert result.risk_level == "medium"
    assert result.evidence_status == "absence_verified"
    assert result.evidence == []
    assert result.absence_check.model_dump() == {
        "keywords": ["分包", "转包", "转委托", "委托第三方"],
        "candidate_count": 0,
    }
    assert "合同肯定没有" not in result.finding
    assert [event for event, _ in events].count("absence_confirmed") == 1
```

Production mutation caught: third RAG, Python-fixed risk level, missing audit metadata, absolute wording, or zero candidates falling back to `needs_review`.

- [ ] **Step 2: Run the zero-candidate test and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_graph.py -k audited_absence_verified -v
```

Expected: FAIL because `absence_result()` is not implemented.

- [ ] **Step 3: Implement `absence_result()` with strict postconditions**

```python
def absence_result(self, state):
    item = state["review_items"][state["current_item_index"]]
    try:
        response = self.review_llm.invoke(build_absence_result_prompt(
            item,
            keywords=state["absence_keywords"],
        ))
        decision = parse_llm_response(response, RiskDecision)
        if decision.evidence_status != "absence_verified":
            raise ValueError("absence_result must return absence_verified")
        forbidden = "合同肯定没有"
        if forbidden in decision.finding or forbidden in decision.risk_description:
            raise ValueError("absence_result used an absolute absence claim")
        if "基于当前合同全文解析结果" not in decision.finding:
            raise ValueError("absence_result finding lacks parsed-content scope")
        self._emit("absence_confirmed", {
            "item_id": item.id,
            "keywords": state["absence_keywords"],
            "candidate_count": 0,
            "decision": decision.model_dump(mode="json"),
        })
        return {"current_decision": decision}
    except Exception as exc:
        raise RuntimeError(f"absence_result {item.id} failed") from exc
```

The risk level remains entirely LLM-produced and schema-validated.

- [ ] **Step 4: Wire no-candidate route to finalization**

Change the no-candidate branch, register `absence_result`, and add its finalization edge:

```python
def route_after_absence_check(state):
    if state["absence_candidates"]:
        return "risk_decision"
    return "absence_result"


builder.add_node("absence_result", nodes.absence_result)
builder.add_edge("absence_result", "finalize_review_item")
```

Ensure `finalize_review_item()` attaches `AbsenceCheckMetadata` and empty Evidence.

- [ ] **Step 5: Add failure tests for malformed absence decisions and unreadable merged content**

Assert:

```python
with pytest.raises(RuntimeError, match="absence_result item_1 failed") as error:
    build_contract_review_graph(ContractReviewNodes(
        parse_llm=FakeLLM(ONE_ITEM_PAYLOAD),
        query_rewrite_llm=FakeLLM({
            **QUERY_REWRITE,
            "keywords": ["分包", "转包"],
        }),
        review_llm=FakeLLM(INSUFFICIENT_DECISION),
        contract_service=FakeContractService([], [], source_objects=[]),
    )).invoke(initial_state())
assert isinstance(error.value.__cause__, ValueError)

class MissingContentContractService(FakeContractService):
    def load_contract_content_objects(self, contract_id):
        raise FileNotFoundError("merged_content_list.json")


with pytest.raises(RuntimeError, match="absence_check item_1 failed") as error:
    build_contract_review_graph(ContractReviewNodes(
        parse_llm=FakeLLM(ONE_ITEM_PAYLOAD),
        query_rewrite_llm=FakeLLM({
            **QUERY_REWRITE,
            "keywords": ["分包", "转包"],
        }),
        review_llm=FakeLLM(ABSENCE_DECISION),
        contract_service=MissingContentContractService([], []),
    )).invoke(initial_state())
assert isinstance(error.value.__cause__, FileNotFoundError)
```

These failures must not create `absence_verified` results.

- [ ] **Step 6: Run all contract-review tests and commit**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_schemas.py tests/test_contract_review_prompts.py tests/test_contract_review_absence.py tests/test_contract_review_graph.py tests/test_contract_review_service.py -v
```

Expected: PASS.

Commit:

```bash
git add app/contract_review/nodes.py app/contract_review/graph.py tests/test_contract_review_graph.py tests/test_contract_review_service.py
git commit -m "feat: verify missing contract clauses"
```

---

### Task 6: Expose Progress Output and JSON Results

**Files:**
- Modify: `scripts/test_contract_review.py`
- Modify: `tests/test_contract_review_service.py`
- Create: `tests/test_contract_review_script.py`

**Interfaces:**
- Consumes: four new progress events and `ReviewResult.absence_check`.
- Produces: clear terminal headings and JSON-serializable audit metadata.

- [ ] **Step 1: Write failing service serialization and CLI output tests**

Service assertion:

```python
assert result["review_results"][0]["absence_check"] == {
    "keywords": ["分包", "转包", "转委托", "委托第三方"],
    "candidate_count": 0,
}
```

Script output test:

```python
def test_print_progress_labels_absence_events(capsys):
    print_progress("absence_check_started", {"item_id": "item_2"})
    print_progress("absence_keywords_generated", {"keywords": ["分包"]})
    print_progress("absence_candidates_found", {
        "candidate_count": 0,
        "candidates": [],
    })
    print_progress("absence_confirmed", {
        "candidate_count": 0,
        "decision": ABSENCE_DECISION,
    })
    output = capsys.readouterr().out
    assert "=== 开始全文缺失核验 ===" in output
    assert "=== 全文扫描关键词 ===" in output
    assert "=== 全文扫描候选 ===" in output
    assert "=== 缺失核验结果 ===" in output
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_service.py tests/test_contract_review_script.py -v
```

Expected: FAIL because the headings and serialized metadata assertion are not implemented.

- [ ] **Step 3: Add the four terminal headings**

Extend `EVENT_HEADINGS`:

```python
"absence_check_started": "开始全文缺失核验",
"absence_keywords_generated": "全文扫描关键词",
"absence_candidates_found": "全文扫描候选",
"absence_confirmed": "缺失核验结果",
```

Do not print or persist the entire merged content list.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
uv run --with pytest python -m pytest tests/test_contract_review_service.py tests/test_contract_review_script.py -v
```

Expected: PASS.

Commit:

```bash
git add scripts/test_contract_review.py tests/test_contract_review_service.py tests/test_contract_review_script.py
git commit -m "test: expose absence review progress"
```

---

### Task 7: Full Regression and Real “分包转包限制” Verification

**Files:**
- Verify only; modify a file only if a failing test exposes a requirement defect.

**Interfaces:**
- Consumes: the completed absence-check graph.
- Produces: fresh automated and real-chain evidence that the feature works without changing other application behavior.

- [ ] **Step 1: Run the complete automated suite**

Run:

```bash
uv run --with pytest python -m pytest -q
```

Expected: all tests and subtests pass; only the pre-existing Starlette/httpx deprecation warning is allowed.

- [ ] **Step 2: Run syntax and diff verification**

Run:

```bash
uv run python -m compileall -q app scripts tests retrieval_evaluation.py
git diff --check
git status --short
```

Expected: compile exit 0, no whitespace errors, and no unexpected generated files.

- [ ] **Step 3: Run the real contract-review smoke script**

Run:

```bash
uv run python scripts/test_contract_review.py
```

For the existing ready contract and “分包转包限制” item, verify the terminal prints this exact logical sequence without a third RAG:

```text
RAG attempt 1: Evidence=[]
Evidence 为空，Rerank Top3 Debug
证据不足，改写检索问题
RAG attempt 2: Evidence=[]
Evidence 为空，Rerank Top3 Debug
开始全文缺失核验
全文扫描关键词
全文扫描候选
本项风险结果
```

If candidates exist, verify each candidate has a real `source_object_index`, `page_idx`, `node_type`, `matched_keywords`, and text, and that the risk LLM receives those candidates. If no candidates exist, verify `absence_confirmed`, `evidence_status=absence_verified`, `evidence=[]`, and `absence_check.candidate_count=0`.

- [ ] **Step 4: Confirm preserved application boundaries**

Run focused regressions explicitly:

```bash
uv run --with pytest python -m pytest tests/test_app_qa.py tests/test_rag_pipeline.py tests/test_retrieval_evaluation.py tests/test_evaluation_service.py tests/test_contract_pipeline.py tests/test_image_ingestion.py -q
```

Expected: PASS; chat, Evaluation, ingestion, and text/table/image RAG are unchanged.

- [ ] **Step 5: Confirm no verification-only commit or push is created**

Run:

```bash
git status --short
git log -1 --oneline
```

Expected: no unexpected working-tree files and the latest commit is the final feature/test commit from Tasks 5 or 6. If verification exposed a defect, return to the relevant task, add a failing regression test, implement the minimal correction, rerun all of that task's tests, and commit only the named production and test files from that task. Do not create an empty verification commit and do not push.
