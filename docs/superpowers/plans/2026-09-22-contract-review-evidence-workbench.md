# Contract Review Evidence Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Project instructions prohibit subagents unless the user explicitly requests an exception. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic Chinese contract-risk rule importer and evidence workbench that retrieves contract facts, prepares research tasks, and leaves all risk conclusions to humans.

**Architecture:** Add an isolated `app/evidence_review/` package. Versioned PDF/TXT/MD rule sets feed the existing contract retrieval pipeline; machine evidence remains immutable while human decisions are stored separately.

**Tech Stack:** Python 3.14, FastAPI, Jinja2, SQLite, Pydantic v2, LangChain `ChatOpenAI`, MinerU, existing BM25/vector retrieval, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-contract-review-evidence-workbench-design.md`

## Global Constraints

- Preserve existing automatic review, ingestion, chat, and Evaluation behavior.
- Never branch on the sample company, headings, numbering, 42-item count, or known classifications.
- Treat rule and contract files as untrusted text; never obey document instructions or automatically browse.
- Never ask an LLM for risk status, risk level, risk description, or modification advice.
- Evidence metadata comes only from retrieval; `not_found` never means a clause is absent.
- External/internal research is prepared, not executed; summaries count only human decisions.
- Default rule-upload limit is 20 MiB; the proprietary PDF stays outside Git.

## Review Focus

- Invalid, oversized, disguised, empty, or MinerU-failed uploads leave no active partial rule set (Task 2).
- Numbering may be Chinese, decimal, bracketed, bulleted, absent, or deeply nested (Task 3).
- One item failure does not fail the run (Task 6).
- Missing identifiers and pending research block conclusive decisions (Tasks 5 and 8).
- Stale browser updates return HTTP 409 instead of overwriting decisions (Task 8).

## File Structure

```text
app/evidence_review/{__init__,schemas,prompts,repository,rule_import,service,web_service}.py
app/templates/{rule_sets,rule_set_detail,evidence_review}.html
```

---

### Task 1: Define Strict Data Contracts

**Files:** Create `app/evidence_review/__init__.py`, `app/evidence_review/schemas.py`; test `tests/test_evidence_review_schemas.py`.

**Interfaces:** Produce `RuleParseResult`, `RuleItem`, `RuleSection`, `EvidencePackage`, `ResearchPackage`, and `HumanDecision`.

- [ ] Write failing tests for unknown numbering, duplicate IDs, Evidence-index validation, and pending-research decisions:

```python
def test_pending_research_disallows_risk():
    with pytest.raises(ValidationError, match="pending research"):
        HumanDecision(
            human_review_status="completed", decision="risk", risk_level="high",
            opinion="存在风险", research_status="pending", research_notes="",
            updated_at="2026-09-22T08:00:00+00:00",
        )
```

- [ ] Run `uv run pytest tests/test_evidence_review_schemas.py -v`; expect import failure.
- [ ] Implement `EvidenceScope=contract|internal_material|external_query|hybrid`, `DecisionMode=automatic_structure_check|threshold_required|expert_review|query_and_compare`, `ItemKind=review_check|process_control`, and strict models with `extra="forbid"`.
- [ ] Validate unique IDs, non-empty rules, valid fact Evidence indices, null non-risk levels, and non-empty completed opinions.
- [ ] Run the focused test; expect PASS.
- [ ] Commit: `git commit -m "feat: define evidence review schemas"`.

---

### Task 2: Store Rule Sets and Extract Source Text

**Files:** Create `app/evidence_review/repository.py`, `app/evidence_review/rule_import.py`; modify `app/config.py`; test `tests/test_evidence_rule_repository.py`, `tests/test_evidence_rule_files.py`, `tests/test_app_config.py`.

**Interfaces:** Produce rule-set lifecycle methods, `validate_rule_upload()`, and `RuleDocumentExtractor.extract()`.

- [ ] Write failing lifecycle tests for `queued → processing → draft → active`, illegal transitions, and immutable used versions.
- [ ] Write upload tests:

```python
def test_pdf_requires_magic_header():
    with pytest.raises(ValueError, match="PDF 文件头"):
        validate_rule_upload("rules.pdf", b"not-pdf", max_bytes=1024)
```

Cover empty, oversized, unsupported, UTF-8 BOM, path traversal, and MinerU failure.
- [ ] Run `uv run pytest tests/test_evidence_rule_repository.py tests/test_evidence_rule_files.py -v`; expect missing interfaces.
- [ ] Add `rule_sets_dir` and `max_rule_upload_bytes=20971520` to `Settings`, loaded from `APP_RULE_SETS_DIR` and `MAX_RULE_UPLOAD_BYTES`.
- [ ] Create `review_rule_sets` with generated ID, version, source metadata/SHA-256, `queued|processing|draft|active|failed`, parsed JSON, timestamps, and safe error text.
- [ ] Store sources only at `<rule_sets_dir>/<rule_set_id>/source.<suffix>`.
- [ ] Decode TXT/MD with `utf-8-sig`. For PDF, verify `%PDF-`, call existing `run_parse()`, and map non-empty MinerU blocks to one-based pages.
- [ ] Run focused/config tests; expect PASS.
- [ ] Commit: `git commit -m "feat: store and extract risk rule files"`.

---

### Task 3: Parse Generic Rule Documents

**Files:** Create `app/evidence_review/prompts.py`; modify `app/evidence_review/rule_import.py`; test `tests/test_evidence_rule_prompts.py`, `tests/test_evidence_rule_import.py`.

**Interfaces:** Produce `build_rule_parse_prompt()` and `RuleSetImportService.import_rule_set()`.

- [ ] Write prompt tests requiring “不得假定固定章节、编号格式、层级或项目数量” and “不得补充法律知识、行业阈值或公司制度”.
- [ ] Parameterize synthetic inputs for `一、`, `1.2`, `（三）`, `①`, bullets, no number, one level, and three levels. Assert dynamic counts and original labels.
- [ ] Test semantic `process_control`, parent/leaf separation, duplicate IDs, invalid JSON, empty results, and safe failures.
- [ ] Run `uv run pytest tests/test_evidence_rule_prompts.py tests/test_evidence_rule_import.py -v`; expect failure.
- [ ] Bind `ChatOpenAI` to `RuleParseResult.model_json_schema()` with strict JSON Schema; locally validate and compute summaries in Python.
- [ ] Assert application code and prompt contain none of the sample company, expected headings, expected count, or classification lookup.
- [ ] Run focused tests; expect PASS.
- [ ] Commit: `git commit -m "feat: parse generic contract risk rules"`.

---

### Task 4: Add Rule-Set Web Pages

**Files:** Modify `app/api.py`, `app/templates/index.html`; create `app/templates/rule_sets.html`, `app/templates/rule_set_detail.html`; test `tests/test_app_rule_sets.py`.

**Interfaces:** Add `GET/POST /rule-sets`, `GET /rule-sets/{id}`, `POST /rule-sets/{id}/activate`, `GET /api/rule-sets/{id}`.

- [ ] Write failing tests for upload scheduling, draft preview, activation, invalid upload, unknown ID, and safe failed output.
- [ ] Run `uv run pytest tests/test_app_rule_sets.py -v`; expect 404.
- [ ] Add optional repository/import-service injection to `create_app()` without changing existing callers.
- [ ] Implement upload/background import/polling/preview/activation with whitelisted payloads.
- [ ] Render hierarchy, original number/text/pages, classification, fact needs, research needs, and warnings; add dashboard navigation.
- [ ] Run `uv run pytest tests/test_app_rule_sets.py tests/test_app_page.py tests/test_app_review_page.py -v`; expect PASS.
- [ ] Commit: `git commit -m "feat: add risk rule set management pages"`.

---

### Task 5: Extract Facts and Prepare Research Tasks

**Files:** Modify `app/evidence_review/prompts.py`; create `app/evidence_review/service.py`; test `tests/test_evidence_review_prompts.py`, `tests/test_evidence_review_service.py`.

**Interfaces:** Produce `EvidenceReviewService.extract_item(contract_id, item)` and `.run(contract_id, items, progress_callback)`.

- [ ] Write failing test proving output has Evidence/facts/research but no risk fields:

```python
payload = service.extract_item("c1", hybrid_company_rule()).model_dump()
assert payload["research_package"]["research_status"] == "pending"
assert "risk_status" not in payload and "risk_level" not in payload
```

- [ ] Test `not_found`, `source_missing`, external-only with no Evidence, process-control with no search call, missing identifiers, and sensitive-field exclusion.
- [ ] Run focused tests; expect missing service.
- [ ] Implement a strict fact prompt that extracts only Evidence-supported facts with Evidence-array indices and explicitly forbids conclusions and Evidence metadata.
- [ ] Call each retrieval query, validate/deduplicate Evidence by `source_object_index`, validate fact indices, filter bank/phone/ID/address data, and assemble research in program code.
- [ ] Catch each item independently and return `extraction_failed` with public text `该审查项取证失败，请人工处理。`.
- [ ] Run focused tests plus `tests/test_contract_service_retrieval.py`, `tests/test_contract_review_graph.py`, and `tests/test_contract_review_service.py`; expect PASS.
- [ ] Commit: `git commit -m "feat: extract contract evidence for manual review"`.

---

### Task 6: Persist and Orchestrate Evidence Runs

**Files:** Modify `app/evidence_review/repository.py`; create `app/evidence_review/web_service.py`; test `tests/test_evidence_review_db.py`, `tests/test_evidence_review_web_service.py`.

**Interfaces:** Produce run/item persistence and `EvidenceReviewWebService.create_run()`, `.execute_run()`, `.get_run_payload()`, `.recover_interrupted_runs()`.

- [ ] Test that machine `ready` leaves human status `pending`, rule snapshots are immutable, item keys are unique, and interrupted runs recover safely.
- [ ] Fake outputs `found`, `extraction_failed`, and research-pending; assert run reaches ready with all three and progress `3/3`.
- [ ] Run focused tests; expect missing tables/services.
- [ ] Create `evidence_review_runs` (`queued|processing|ready|failed`, plus `pending|in_progress|completed`) and `evidence_review_items` keyed by `(run_id, rule_item_id)`.
- [ ] Validate contract ready/rule active, persist all item results atomically, and keep item failures non-fatal.
- [ ] Recover queued/processing with `服务重启导致人工取证任务中断，请重新创建任务。`.
- [ ] Run focused tests; expect PASS.
- [ ] Commit: `git commit -m "feat: persist evidence review runs"`.

---

### Task 7: Add the Evidence Workbench Page

**Files:** Modify `app/api.py`, `app/templates/index.html`; create `app/templates/evidence_review.html`; test `tests/test_app_evidence_review_page.py`.

**Interfaces:** Add `GET /contracts/{id}/evidence-review`, `POST .../runs`, and `GET /api/.../runs/{run_id}`.

- [ ] Write failing tests for run creation, polling, cross-contract 404, non-ready 409, and Evidence/research display without model conclusions.
- [ ] Run focused test; expect 404.
- [ ] Wire default/injected web service and interrupted-run recovery into `create_app()`.
- [ ] Implement whitelisted APIs and filters for found/not-found/source-missing/failed/research/human status.
- [ ] Render rule text, Evidence pages/object indices, facts, missing sources, query targets/topics/comparison points, and the fixed not-found warning.
- [ ] Make “人工取证” primary and label old review “实验性自动判断”.
- [ ] Run focused tests plus `tests/test_app_page.py`, `tests/test_app_review_page.py`, `tests/test_app_api.py`; expect PASS.
- [ ] Commit: `git commit -m "feat: add contract evidence review workbench"`.

---

### Task 8: Save Human Decisions Safely

**Files:** Modify `app/evidence_review/repository.py`, `app/evidence_review/web_service.py`, `app/api.py`, `app/templates/evidence_review.html`; test `tests/test_evidence_review_decisions.py`, `tests/test_app_evidence_review_page.py`.

**Interfaces:** Produce `save_human_decision(run_id, item_id, decision, expected_updated_at)` and `POST .../items/{item_id}/decision`.

- [ ] Test pending research blocks risk/no-risk, completed requires opinion, non-risk level is null, and risk level is optional.
- [ ] Test stale timestamps raise `DecisionConflictError` and route returns 409.
- [ ] Run focused tests; expect missing decision persistence.
- [ ] Create `evidence_review_decisions` keyed by `(run_id, rule_item_id)`; atomically compare timestamps, upsert, and recalculate run human status.
- [ ] Add form fields for decision, level, opinion, research notes, and hidden timestamp; preserve input on 400 and redirect to item anchor on success.
- [ ] Show only human-derived risk/no-risk/cannot-determine counts.
- [ ] Run decision/page tests; expect PASS.
- [ ] Commit: `git commit -m "feat: add human decisions to evidence review"`.

---

### Task 9: Security, Regression, and Local Acceptance

**Files:** Modify evidence services/tests; create `tests/fixtures/rules/generic_numbering.md`, `nested_rules.md`, `bulleted_rules.md`.

**Interfaces:** Produce verified privacy filtering, safe errors, regression evidence, and a non-committed real-PDF acceptance result.

- [ ] Add prompt-injection fixtures plus bank account, phone, ID, and address data; assert no network call and no sensitive research target.
- [ ] Inject errors containing API keys, absolute paths, and raw model responses; assert web/API output contains only public messages and whitelisted fields.
- [ ] Run focused security tests; expect PASS.
- [ ] Run `uv run pytest -q`; expect zero failures.
- [ ] Run `uv run python -m compileall -q app tests` and `git diff --check`; expect exit 0.
- [ ] Locally import `/Users/wenjuhao/Downloads/投标或拟签约民营及海外工程项目合同风险评估负面清单（找回副本）.pdf`; verify 4 pages, 2 top-level stages, 42 leaf/process items, and original numbers including `一-5.3` and `二-6.6`.
- [ ] Confirm the proprietary PDF is absent from Git and sample-specific terms/counts do not exist in application logic.
- [ ] Commit: `git commit -m "test: harden evidence review workflow"`.

## Final Verification

```bash
uv run pytest -q
uv run python -m compileall -q app tests
git diff --check
git status --short
```

Expected: all tests pass, compileall and diff check exit 0, the working tree has no unintended changes, and local proprietary-PDF acceptance passes without committing the file.
