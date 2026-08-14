# MinerU Raw Parse Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add an independent CLI script that sends one PDF to MinerU, preserves the returned raw *_content_list.json, and prints page/object/type statistics.

**Architecture:** mineru_raw_parse.py owns only the MinerU HTTP protocol, ZIP member selection, raw-byte output, and statistics display. It uses httpx.Client with the asynchronous task protocol from pdf_trans, but imports neither pdf_trans nor LlamaIndex. Tests use httpx.MockTransport, so no live MinerU service is needed.

**Tech Stack:** Python 3.14, httpx, python-dotenv, argparse, json, zipfile, unittest.

**Spec:** docs/superpowers/specs/2026-08-14-mineru-raw-parse-design.md

## Global Constraints

- The script is independent of /Users/wenjuhao/code/python/pdf_trans.
- The script must not import or use LlamaIndex, Document, Index, Retriever, cleaning, cross-page merging, classification, translation, formula processing, or rendering.
- MinerU requests must include return_content_list=true and response_format_zip=true.
- The output JSON must contain the exact bytes of the selected *_content_list.json ZIP member; do not deserialize and reserialize before saving.
- The default MinerU URL is http://127.0.0.1:7100; PDF_TRANS_MINERU_URL, PDF_TRANS_MINERU_BACKEND, and PDF_TRANS_MINERU_SERVER_URL are supported, with CLI arguments taking precedence.
- Automated tests must use httpx.MockTransport and make no network calls.

### Task 1: Add the direct dependency and write a failing flow test

**Files:**
- Modify: pyproject.toml dependency list
- Modify: uv.lock via uv add httpx
- Create: tests/test_mineru_raw_parse.py

**Interfaces:** The test will call mineru_raw_parse.run_parse(pdf_path, output_path, client=client, svr_url=..., backend=..., server_url=..., poll_interval=0). The production implementation will expose the same function and main(argv=None).

- [ ] Step 1: Add httpx as a direct dependency

Run:

~~~bash
PROJECT_DIR='/Users/wenjuhao/code/python/contract_risk_review '
uv --directory "$PROJECT_DIR" add httpx
~~~

Expected: pyproject.toml and uv.lock contain the direct dependency.

- [ ] Step 2: Write the failing test

Create tests/test_mineru_raw_parse.py with a temporary PDF and a ZIP whose single contract_content_list.json member contains deliberately non-pretty bytes such as b'[{"type":"text","page_idx":0},\n {"type":"table","page_idx":1}, {"page_idx":1}]'. Use httpx.MockTransport to return 202 from POST /tasks, pending then completed from the status URL, and the ZIP from the result URL. Call run_parse, assert output bytes equal the original bytes, and assert stdout contains 页数: 2, 解析对象总数: 3, text: 1, table: 1, and <missing>: 1.

The mock POST response must be:

~~~python
httpx.Response(202, json={
    "task_id": "task-1",
    "status_url": "http://mineru.test/status/task-1",
    "result_url": "http://mineru.test/result/task-1",
})
~~~

- [ ] Step 3: Verify the red state

Run:

~~~bash
PROJECT_DIR='/Users/wenjuhao/code/python/contract_risk_review '
PYTHONPATH="$PROJECT_DIR" uv --directory "$PROJECT_DIR" run python -m unittest tests.test_mineru_raw_parse -v
~~~

Expected: FAIL with ModuleNotFoundError: No module named 'mineru_raw_parse'.

### Task 2: Implement the minimal standalone MinerU script

**Files:**
- Create: mineru_raw_parse.py

**Interfaces:**
- run_parse(pdf_path: Path, output_path: Path, *, svr_url: str, backend: str, server_url: str | None, client: httpx.Client | None = None, poll_interval: float = 2.0) -> None: upload, poll, download, preserve raw JSON, and print stats.
- main(argv: list[str] | None = None) -> None: load .env, resolve CLI/environment settings, derive output, and call run_parse.

- [ ] Step 1: Add constants and input validation

Use these exact defaults and form fields:

~~~python
DEFAULT_SVR_URL = "http://127.0.0.1:7100"
DEFAULT_BACKEND = "hybrid-engine"
POLL_INTERVAL_SECONDS = 2.0
TASK_TIMEOUT_SECONDS = 30 * 60
BASE_PARSE_FORM = {
    "parse_method": "auto",
    "effort": "medium",
    "formula_enable": "true",
    "table_enable": "true",
    "image_analysis": "false",
    "return_md": "false",
    "return_middle_json": "false",
    "return_model_output": "false",
    "return_content_list": "true",
    "return_images": "true",
    "response_format_zip": "true",
}
~~~

Resolve the PDF with Path.expanduser().resolve(), require a regular file, and require a case-insensitive .pdf suffix before opening an HTTP client.

- [ ] Step 2: Implement submit, poll, and download

Use the provided httpx.Client or create and close one locally. Submit multipart data with files={"files": (pdf_path.name, pdf_file, "application/pdf")} and form data equal to BASE_PARSE_FORM plus backend and optional server_url.rstrip("/"). Require HTTP 202 and string task_id, status_url, and result_url. Poll pending/processing until completed, raise on failed, unknown status, invalid JSON, non-200 responses, or the 30-minute deadline. Download only after completion and require HTTP 200 with application/zip content type.

- [ ] Step 3: Preserve the raw content-list bytes

Open the downloaded bytes with zipfile.ZipFile(io.BytesIO(...)), select exactly one member whose filename ends in _content_list.json, and read its bytes. Reject zero or multiple matches. Create the output parent and call output_path.write_bytes(raw_content). Do not call json.dump or json.dumps for the output file.

- [ ] Step 4: Print statistics and implement CLI parsing

After writing the bytes, parse the same raw_content with json.loads and require a list. Use:

~~~python
page_count = len({
    item["page_idx"]
    for item in content_list
    if isinstance(item, dict) and item.get("page_idx") is not None
})
object_count = len(content_list)
type_counts = Counter(
    item.get("type", "<missing>") if isinstance(item, dict) else "<non-object>"
    for item in content_list
)
~~~

Print 页数: N, 解析对象总数: N, then sorted type counts. In main(argv=None), call load_dotenv(), accept the PDF positional argument plus --output, --svr-url, --backend, and --server-url, prefer CLI values over PDF_TRANS_MINERU_* environment values, and default output to <input_stem>_mineru_raw.json. Keep the if __name__ == "__main__": main() entry point.

### Task 3: Verify boundaries and commit only task files

**Files:**
- Verify: mineru_raw_parse.py
- Verify: tests/test_mineru_raw_parse.py
- Verify: pyproject.toml, uv.lock

- [ ] Step 1: Verify the focused test is green

Run:

~~~bash
PROJECT_DIR='/Users/wenjuhao/code/python/contract_risk_review '
PYTHONPATH="$PROJECT_DIR" uv --directory "$PROJECT_DIR" run python -m unittest tests.test_mineru_raw_parse -v
~~~

Expected: one test passes with no network access.

- [ ] Step 2: Run all tests, compile, and lock checks

Run:

~~~bash
PROJECT_DIR='/Users/wenjuhao/code/python/contract_risk_review '
PYTHONPATH="$PROJECT_DIR" uv --directory "$PROJECT_DIR" run python -m unittest discover -s tests -v
PYTHONPATH="$PROJECT_DIR" uv --directory "$PROJECT_DIR" run python -m py_compile mineru_raw_parse.py
uv --directory "$PROJECT_DIR" lock --check
~~~

Expected: all tests pass, compilation exits 0, and the lock check succeeds.

- [ ] Step 3: Verify prohibited scope is absent

Run:

~~~bash
PROJECT_DIR='/Users/wenjuhao/code/python/contract_risk_review '
if rg -n 'llama_index|Document|VectorStoreIndex|Retriever|translate|render|clean|cross_page' "$PROJECT_DIR/mineru_raw_parse.py"; then exit 1; fi
~~~

Expected: no matches.

- [ ] Step 4: Commit only the task files

Run:

~~~bash
PROJECT_DIR='/Users/wenjuhao/code/python/contract_risk_review '
git -C "$PROJECT_DIR" add -- mineru_raw_parse.py tests/test_mineru_raw_parse.py pyproject.toml uv.lock
git -C "$PROJECT_DIR" commit --only mineru_raw_parse.py tests/test_mineru_raw_parse.py pyproject.toml uv.lock -m "feat: add minimal MinerU raw parse test"
~~~

Expected: the commit contains only the script, its test, and dependency metadata; unrelated .idea and environment files remain outside this commit.
