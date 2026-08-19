# Contract Image RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让合同中的银行账户、法人身份证和普通图片被结构化识别、按需 OCR 校验、构造成 Image TextNode，并进入现有 RAG 和 Evaluation 链路且保留原始图片引用。

**Architecture:** MinerU 解析阶段保存 content list 引用的原始图片；独立 Vision Service 用一次请求完成分类和结构化提取，bank/id 再调用 MinerU 单图 OCR 并由纯代码校验。识别结果原位补充到 merged_content_list.json，图片再复用现有 retrieval_context、TextNode、向量召回、重排、Evidence Selector、Answer 和 Evaluation。

**Tech Stack:** Python 3.14、httpx、LangChain ChatOpenAI、Pydantic、LlamaIndex TextNode/VectorStoreIndex、FastAPI/Jinja2、unittest/pytest。

**Spec:** docs/superpowers/specs/2026-08-19-image-rag-design.md

## Global Constraints

- 不引入 CLIP、多模态 embedding、新向量库、图片相似度搜索、PaddleOCR 或复杂 Agent。
- VL 分类和字段提取必须在每张图片的一次请求中完成；Schema 修复不得触发第二次模型调用。
- 只为 bank_account 和 identity_card 调用 MinerU OCR；general 不调用 OCR。
- OCR 只校验，不覆盖 VL structured_data；明确冲突必须保留并提示人工核验。
- 图片识别结果直接写回 merged_content_list.json，不创建 image_understanding.json。
- 不改变 merged 对象顺序和数组长度；source_object_index 始终是 merged 数组下标。
- 单图缺失、VL 失败、Schema 失败或 OCR 失败不得导致整份合同失败。
- text/table 的 node.text、embedding、retrieval_context、RAG、聊天和 Evaluation 行为必须保持回归通过。
- img_path 始终保留为合同目录内的原始相对路径；不得把绝对路径写入 Node 或 Evidence。
- ZIP 和 img_path 处理必须拒绝绝对路径、父目录穿越和写出合同目录。
- 当前工作树已有用户未提交改动。执行每个提交步骤前先运行 git diff --cached --name-only 和 git diff -- <paths>；不得暂存或提交与本计划无关的旧改动。若目标文件已有无法安全拆分的旧改动，保留工作树修改并跳过该任务的提交，不得覆盖、stash 或重置用户改动。
- 禁止使用 GitHub CLI gh；只使用本地 git，且本计划不涉及远程平台操作。

---

## File map

### New production files

- app/image_schemas.py：严格的三类图片数据模型、response_format 和响应解析异常。
- app/image_understanding.py：独立 ImageUnderstandingService，多模态请求和本地严格校验。
- app/image_ocr.py：MinerUImageOCRService adapter。
- app/image_ingestion.py：图片路径解析、逐图 enrichment、状态降级和 merged JSON 原子写入。
- image_verification.py：纯代码 OCR 候选提取、归一化和字段一致性校验。
- image_searchable_text.py：从 enriched image 对象生成确定性 node.text。
- app/evidence_serialization.py：聊天和 Evaluation 共用的 Node/Evidence serializer。

### Modified production files

- mineru_raw_parse.py：保存引用图片并增加可复用的单图 OCR 调用。
- app/config.py：IMAGE_VISION_MODEL 和 IMAGE_VISION_TIMEOUT_SECONDS。
- app/pipeline.py：merge 后执行 image enrichment，并把 enriched objects 传给 context 和 Node。
- retrieval_context_preprocess.py：增加 image retrieval_context 分支。
- mineru_to_nodes.py：增加 Image TextNode 分支。
- app/qa.py：使用共享 serializer。
- app/evaluation_metrics.py：使用共享 serializer。
- app/templates/chat.html：展示 image Evidence。
- app/templates/evaluation.html：在现有 Vector/Rerank/Selected Evidence 中展示 image 字段。

### Tests

- tests/test_mineru_raw_parse.py
- tests/test_app_config.py
- tests/test_image_schemas.py
- tests/test_image_understanding.py
- tests/test_image_ocr.py
- tests/test_image_verification.py
- tests/test_image_searchable_text.py
- tests/test_image_ingestion.py
- tests/test_retrieval_context_preprocess.py
- tests/test_mineru_to_nodes.py
- tests/test_contract_pipeline.py
- tests/test_app_qa.py
- tests/test_evaluation_metrics.py
- tests/test_app_evaluation_page.py
- tests/test_app_page.py

---

### Task 1: Persist MinerU image assets and expose single-image OCR

**Files:**
- Modify: mineru_raw_parse.py:45-196
- Create: app/image_ocr.py
- Modify: tests/test_mineru_raw_parse.py
- Create: tests/test_image_ocr.py

**Interfaces:**
- Keeps run_parse(pdf_path, output_path, *, svr_url, backend, server_url, client=None, poll_interval=POLL_INTERVAL_SECONDS) -> None unchanged.
- Produces run_image_ocr(image_path, *, svr_url, backend, server_url, client=None, poll_interval=POLL_INTERVAL_SECONDS) -> str.
- Produces MinerUImageOCRService.extract_text(image_path: Path) -> str.
- Saves referenced images beneath output_path.parent using the exact relative img_path.

- [ ] **Step 1: Extend the parse archive test to require image extraction**

Add an image object and image bytes to the existing mocked ZIP:

~~~python
raw_content = json.dumps(
    [
        {"type": "text", "text": "正文", "page_idx": 0},
        {
            "type": "image",
            "img_path": "images/account.jpg",
            "page_idx": 0,
        },
    ],
    ensure_ascii=False,
).encode()

with zipfile.ZipFile(archive_buffer, "w") as archive:
    archive.writestr("contract/contract_content_list.json", raw_content)
    archive.writestr("contract/images/account.jpg", b"jpeg-bytes")

mineru_raw_parse.run_parse(
    pdf_path,
    output_path,
    svr_url="http://mineru.test",
    backend="hybrid-engine",
    server_url=None,
    client=client,
    poll_interval=0,
)
assert output_path.read_bytes() == raw_content
assert (root / "images" / "account.jpg").read_bytes() == b"jpeg-bytes"
~~~

Also add direct safety tests for a ZIP whose referenced image resolves through .. and for a content list that references a missing image. The traversal case must not write outside output_path.parent; the missing image case must still save the content list.

- [ ] **Step 2: Run the parse tests to verify RED**

Run: uv run --with pytest pytest tests/test_mineru_raw_parse.py -q

Expected: the image extraction assertion fails because run_parse currently saves only JSON.

- [ ] **Step 3: Refactor ZIP reading and save only referenced images**

Replace the JSON-only helper with focused helpers:

~~~python
def _read_parse_archive(
    archive_bytes: bytes,
) -> tuple[str, bytes, list[dict[str, object]], dict[str, bytes]]:
    """Return content member name, raw JSON, objects, and safe member bytes."""


def _referenced_image_member(
    content_member: str,
    img_path: str,
) -> str:
    """Resolve img_path relative to the content-list member directory."""


def _write_referenced_images(
    archive_members: Mapping[str, bytes],
    *,
    content_member: str,
    objects: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Write only valid image references below output_dir."""
~~~

Use PurePosixPath for ZIP member validation. Reject absolute paths and any path containing .. before matching. Resolve the output target and require target.is_relative_to(output_dir.resolve()). Never use ZipFile.extractall().

After validating the content list, run_parse writes raw JSON exactly as before, then writes available referenced images. Missing members log a warning with img_path and continue.

- [ ] **Step 4: Add failing tests for run_image_ocr and its adapter**

Use the same MockTransport task lifecycle as the existing parse test. Return a ZIP containing:

~~~json
[
  {"type": "text", "text": "户名：甲公司"},
  {"type": "text", "text": "账号：110914414810101"}
]
~~~

Assert:

~~~python
text = run_image_ocr(
    image_path,
    svr_url="http://mineru.test",
    backend="hybrid-engine",
    server_url=None,
    client=client,
    poll_interval=0,
)
assert text == "户名：甲公司\n账号：110914414810101"
assert b'name="parse_method"' in submitted_body
assert b"ocr" in submitted_body
~~~

For the adapter:

~~~python
service = MinerUImageOCRService(
    svr_url="http://mineru.test",
    backend="hybrid-engine",
    server_url=None,
    runner=lambda *_args, **_kwargs: "OCR text",
)
assert service.extract_text(image_path) == "OCR text"
~~~

- [ ] **Step 5: Run OCR tests to verify RED**

Run: uv run --with pytest pytest tests/test_image_ocr.py -q

Expected: FAIL because run_image_ocr and MinerUImageOCRService do not exist.

- [ ] **Step 6: Extract shared task transport and implement OCR**

Keep public run_parse behavior stable while extracting:

~~~python
def _run_task(
    input_path: Path,
    *,
    form: Mapping[str, str],
    svr_url: str,
    backend: str,
    server_url: str | None,
    client: httpx.Client | None,
    poll_interval: float,
) -> bytes:
    """Submit, poll, and download one MinerU ZIP result."""
~~~

Implement run_image_ocr with the exact OCR form from the spec. Validate that image_path is an existing ordinary file, but do not require .pdf. Parse the returned content list and join non-empty text fields in document order. If no text exists, return an empty string so the ingestion layer can mark ocr_status=empty.

Implement the thin adapter:

~~~python
class OCRRunner(Protocol):
    def __call__(
        self,
        image_path: Path,
        *,
        svr_url: str,
        backend: str,
        server_url: str | None,
    ) -> str:
        raise NotImplementedError


class MinerUImageOCRService:
    def __init__(
        self,
        *,
        svr_url: str,
        backend: str,
        server_url: str | None,
        runner: OCRRunner = run_image_ocr,
    ) -> None:
        self._svr_url = svr_url
        self._backend = backend
        self._server_url = server_url
        self._runner = runner

    def extract_text(self, image_path: Path) -> str:
        return self._runner(
            image_path,
            svr_url=self._svr_url,
            backend=self._backend,
            server_url=self._server_url,
        )
~~~

- [ ] **Step 7: Run MinerU and OCR tests to verify GREEN**

Run: uv run --with pytest pytest tests/test_mineru_raw_parse.py tests/test_image_ocr.py -q

Expected: PASS.

- [ ] **Step 8: Commit the isolated MinerU work**

First inspect staged scope:

~~~bash
git diff -- mineru_raw_parse.py app/image_ocr.py tests/test_mineru_raw_parse.py tests/test_image_ocr.py
git diff --cached --name-only
~~~

If those paths contain only this task's work:

~~~bash
git add mineru_raw_parse.py app/image_ocr.py tests/test_mineru_raw_parse.py tests/test_image_ocr.py
git commit -m "feat: persist mineru images and support image ocr"
~~~

Otherwise leave the overlapping paths uncommitted and report them; do not stage unrelated hunks.

### Task 2: Define strict image schemas and the Vision Service

**Files:**
- Create: app/image_schemas.py
- Create: app/image_understanding.py
- Modify: app/config.py
- Create: tests/test_image_schemas.py
- Create: tests/test_image_understanding.py
- Modify: tests/test_app_config.py

**Interfaces:**
- Produces ImageExtraction, BankAccountExtraction, IdentityCardExtraction, GeneralImageExtraction.
- Produces IMAGE_RESPONSE_FORMAT for ChatOpenAI.bind(response_format=IMAGE_RESPONSE_FORMAT).
- Produces ImageSchemaError and ImageClassificationError.
- Produces ImageUnderstandingService.classify_and_extract(image_path: Path) -> ImageExtraction.
- Adds Settings.image_vision_model: str and Settings.image_vision_timeout_seconds: float.

- [ ] **Step 1: Write strict Schema tests**

Create tests that validate all three branches:

~~~python
def test_bank_account_schema_accepts_nullable_fields_and_forbids_extras():
    extraction = validate_image_extraction(
        {
            "image_type": "bank_account",
            "data": {
                "account_name": "甲公司",
                "account_number": "110914414810101",
                "bank_name": "中国甲银行",
                "bank_branch": None,
            },
        }
    )
    assert extraction.image_type == "bank_account"
    assert extraction.data.account_number == "110914414810101"

    with pytest.raises(ImageSchemaError):
        validate_image_extraction(
            {
                "image_type": "bank_account",
                "data": {
                    "account_name": "甲公司",
                    "account_number": "1",
                    "bank_name": "银行",
                    "bank_branch": None,
                    "guessed_field": "forbidden",
                },
            }
        )


def test_unknown_type_raises_classification_error():
    with pytest.raises(ImageClassificationError):
        validate_image_extraction(
            {"image_type": "chart", "data": {}}
        )
~~~

Add equivalent acceptance tests for identity_card and general, plus branch mismatch and missing required fields.

- [ ] **Step 2: Run Schema tests to verify RED**

Run: uv run --with pytest pytest tests/test_image_schemas.py -q

Expected: FAIL because app.image_schemas does not exist.

- [ ] **Step 3: Implement strict models and response format**

Use Pydantic models with extra="forbid" and nullable required fields:

~~~python
class BankAccountData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_name: str | None
    account_number: str | None
    bank_name: str | None
    bank_branch: str | None


class BankAccountExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_type: Literal["bank_account"]
    data: BankAccountData
~~~

Define matching identity/general models and:

~~~python
ImageExtraction = Annotated[
    BankAccountExtraction
    | IdentityCardExtraction
    | GeneralImageExtraction,
    Field(discriminator="image_type"),
]

def validate_image_extraction(payload: Any) -> ImageExtraction:
    if not isinstance(payload, dict):
        raise ImageSchemaError("image response must be an object")
    if payload.get("image_type") not in {
        "bank_account",
        "identity_card",
        "general",
    }:
        raise ImageClassificationError(
            f"unsupported image_type: {payload.get('image_type')!r}"
        )
    try:
        return IMAGE_EXTRACTION_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ImageSchemaError("invalid image extraction schema") from exc
~~~

Check image_type before TypeAdapter validation so unsupported values raise ImageClassificationError; all other validation failures raise ImageSchemaError. IMAGE_RESPONSE_FORMAT must be a strict json_schema envelope with additionalProperties=false and the three branch data Schemas.

- [ ] **Step 4: Write Vision Service tests**

Use an injected fake LLM:

~~~python
class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(
            content=json.dumps(self.payload, ensure_ascii=False)
        )


def test_classify_and_extract_sends_one_image_request(tmp_path):
    image_path = tmp_path / "account.jpg"
    image_path.write_bytes(b"jpeg")
    llm = FakeLLM(
        {
            "image_type": "bank_account",
            "data": {
                "account_name": "甲公司",
                "account_number": "110914414810101",
                "bank_name": "甲银行",
                "bank_branch": None,
            },
        }
    )
    service = ImageUnderstandingService(
        model_name="test-vl",
        llm=llm,
    )

    result = service.classify_and_extract(image_path)

    assert result.image_type == "bank_account"
    assert len(llm.messages) == 1
    content = llm.messages[0][0].content
    assert any(block.get("type") == "image_url" for block in content)
    assert "base64," in json.dumps(content)
~~~

Also test parsed content from response.additional_kwargs["parsed"], malformed JSON, unsupported extension, and missing file.

Add a constructor test for the non-injected path:

~~~python
def test_default_llm_binds_strict_image_response_format(monkeypatch):
    factory = Mock()
    model = factory.return_value
    bound = model.bind.return_value
    monkeypatch.setattr(
        "app.image_understanding.ChatOpenAI",
        factory,
    )

    service = ImageUnderstandingService(
        model_name="test-vl",
        api_key="test-key",
        base_url="https://llm.test/v1",
        timeout_seconds=120,
    )

    factory.assert_called_once_with(
        model="test-vl",
        api_key="test-key",
        base_url="https://llm.test/v1",
        temperature=0,
        timeout=120,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    model.bind.assert_called_once_with(
        response_format=IMAGE_RESPONSE_FORMAT
    )
    assert service._llm is bound
    assert IMAGE_RESPONSE_FORMAT["json_schema"]["strict"] is True
~~~

- [ ] **Step 5: Run Vision tests to verify RED**

Run: uv run --with pytest pytest tests/test_image_understanding.py -q

Expected: FAIL because ImageUnderstandingService does not exist.

- [ ] **Step 6: Implement ImageUnderstandingService**

Use a constructor that is fully injectable:

~~~python
class ImageUnderstandingService:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        llm: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._llm = llm or ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            timeout=timeout_seconds,
            max_retries=0,
            extra_body={"enable_thinking": False},
        ).bind(response_format=IMAGE_RESPONSE_FORMAT)

    def classify_and_extract(self, image_path: Path) -> ImageExtraction:
        resolved = image_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        mime_type = _mime_type(resolved)
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": IMAGE_EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}"
                    },
                },
            ]
        )
        response = self._llm.invoke([message])
        return validate_image_extraction(_response_payload(response))
~~~

Read bytes once, map .jpg/.jpeg/.png/.webp to MIME, construct a HumanMessage with a text instruction and one image_url data URL, invoke exactly once, parse additional_kwargs.parsed or JSON text, then call validate_image_extraction(). Do not catch transport errors here; the ingestion service assigns per-image statuses.

- [ ] **Step 7: Add config tests and implementation**

Extend tests/test_app_config.py:

~~~python
def test_load_settings_reads_image_vision_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "qwen-current-plus")
    monkeypatch.setenv("IMAGE_VISION_MODEL", "qwen-test-vl")
    monkeypatch.setenv("IMAGE_VISION_TIMEOUT_SECONDS", "180")

    settings = load_settings(tmp_path)

    assert settings.image_vision_model == "qwen-test-vl"
    assert settings.image_vision_timeout_seconds == 180.0
~~~

Add default-valued dataclass fields at the end so existing direct Settings constructor calls stay source-compatible:

~~~python
image_vision_model: str = "qwen3.7-plus"
image_vision_timeout_seconds: float = 120.0
~~~

load_settings uses IMAGE_VISION_MODEL, then LLM_MODEL, then qwen3.7-plus.

- [ ] **Step 8: Run Schema, Vision, and config tests**

Run: uv run --with pytest pytest tests/test_image_schemas.py tests/test_image_understanding.py tests/test_app_config.py -q

Expected: PASS.

- [ ] **Step 9: Commit the Schema and Vision layer**

~~~bash
git diff -- app/image_schemas.py app/image_understanding.py app/config.py tests/test_image_schemas.py tests/test_image_understanding.py tests/test_app_config.py
git diff --cached --name-only
git add app/image_schemas.py app/image_understanding.py app/config.py tests/test_image_schemas.py tests/test_image_understanding.py tests/test_app_config.py
git commit -m "feat: add structured contract image understanding"
~~~

If app/config.py or its test contains prior user edits that cannot be isolated, do not stage those files; report the deferred commit scope.

### Task 3: Implement deterministic OCR verification and searchable text

**Files:**
- Create: image_verification.py
- Create: image_searchable_text.py
- Create: tests/test_image_verification.py
- Create: tests/test_image_searchable_text.py

**Interfaces:**
- Produces VerificationResult(status: VerificationStatus, details: dict[str, Any]).
- Produces verify_image_data(image_type: str, structured_data: Mapping[str, Any], ocr_text: str | None) -> VerificationResult.
- Produces image_to_searchable_text(image: Mapping[str, Any]) -> str | None.

- [ ] **Step 1: Write verification tests for exact matches, conflicts, and insufficiency**

~~~python
def test_bank_account_number_exact_match_after_separator_normalization():
    result = verify_image_data(
        "bank_account",
        {
            "account_name": "甲有限公司",
            "account_number": "1109 1441-4810 101",
            "bank_name": "中国甲银行",
            "bank_branch": None,
        },
        "户名：甲有限公司\n开户银行：中国甲银行\n账号：110914414810101",
    )
    assert result.status == "verified"
    assert result.details["account_number"]["status"] == "verified"


def test_one_digit_account_difference_is_conflict():
    result = verify_image_data(
        "bank_account",
        {
            "account_name": "甲有限公司",
            "account_number": "110914414810101",
            "bank_name": "中国甲银行",
            "bank_branch": None,
        },
        "户名：甲有限公司\n开户银行：中国甲银行\n账号：110914414810107",
    )
    assert result.status == "conflict"
    assert result.details["account_number"]["status"] == "conflict"


def test_identity_without_ocr_candidate_is_insufficient():
    result = verify_image_data(
        "identity_card",
        {
            "name": "张三",
            "id_number": "11010119900101123X",
            "valid_from": None,
            "valid_to": None,
        },
        "模糊文字",
    )
    assert result.status == "insufficient"
~~~

Add tests for uppercase X, validity exact comparison, label-bound name/bank conflict, and general returning not_required without reading OCR.

- [ ] **Step 2: Run verification tests to verify RED**

Run: uv run --with pytest pytest tests/test_image_verification.py -q

Expected: FAIL because image_verification does not exist.

- [ ] **Step 3: Implement normalization, candidate extraction, and aggregate status**

Define:

~~~python
VerificationStatus = Literal[
    "verified", "conflict", "insufficient", "not_required"
]

@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    details: dict[str, Any]
~~~

Use unicodedata.normalize("NFKC", value), exact alphanumeric normalization for account/id, and explicit regex candidates. A missing exact match is conflict only when a credible field-specific candidate exists; otherwise it is insufficient. Overall conflict wins, then type-specific required verified fields, else insufficient.

- [ ] **Step 4: Write searchable text tests**

~~~python
def test_bank_image_searchable_text_is_deterministic_and_omits_path():
    image = {
        "img_path": "images/secret.jpg",
        "image_type": "bank_account",
        "structured_data": {
            "account_name": "甲有限公司",
            "account_number": "110914414810101",
            "bank_name": "中国甲银行",
            "bank_branch": None,
        },
        "verification_status": "verified",
    }
    text = image_to_searchable_text(image)
    assert text == (
        "银行账户信息。\n"
        "户名：甲有限公司。\n"
        "开户银行：中国甲银行。\n"
        "银行账号：110914414810101。\n"
        "OCR校验：已核验。"
    )
    assert "secret.jpg" not in text


def test_conflict_is_visible_in_actual_evidence_text():
    text = image_to_searchable_text(
        {
            "image_type": "identity_card",
            "structured_data": {
                "name": "张三",
                "id_number": "11010119900101123X",
                "valid_from": None,
                "valid_to": None,
            },
            "verification_status": "conflict",
        }
    )
    assert "关键字段存在冲突，需要人工核验" in text


def test_empty_general_result_has_no_searchable_text():
    assert image_to_searchable_text(
        {
            "image_type": "general",
            "structured_data": {
                "visible_text": None,
                "content_description": "",
            },
            "verification_status": "not_required",
        }
    ) is None
~~~

- [ ] **Step 5: Run searchable text tests to verify RED**

Run: uv run --with pytest pytest tests/test_image_searchable_text.py -q

Expected: FAIL because image_searchable_text does not exist.

- [ ] **Step 6: Implement all three deterministic formatters**

Implement image_to_searchable_text(image) with fixed field order, whitespace normalization, full-width Chinese punctuation, omitted empty rows, and verification suffix only for bank/id. Do not serialize JSON and do not include img_path, OCR raw text, model name, or error strings.

- [ ] **Step 7: Run pure image processing tests**

Run: uv run --with pytest pytest tests/test_image_verification.py tests/test_image_searchable_text.py -q

Expected: PASS.

- [ ] **Step 8: Commit pure deterministic processing**

~~~bash
git add image_verification.py image_searchable_text.py tests/test_image_verification.py tests/test_image_searchable_text.py
git commit -m "feat: verify and format contract image data"
~~~

### Task 4: Enrich merged image objects with per-image degradation

**Files:**
- Create: app/image_ingestion.py
- Create: tests/test_image_ingestion.py

**Interfaces:**
- Consumes ImageUnderstandingService.classify_and_extract(), MinerUImageOCRService.extract_text(), and verify_image_data().
- Produces ContractImageIngestionService.enrich_images(objects: list[dict[str, Any]], *, storage_dir: Path) -> list[dict[str, Any]].
- Produces write_json_atomic(path: Path, payload: Any) -> None.
- Preserves object order, array length, and all original fields.

- [ ] **Step 1: Write success and source-index preservation tests**

~~~python
def test_enriches_bank_image_in_place_without_changing_indices(tmp_path):
    image_path = tmp_path / "images" / "account.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"jpeg")
    objects = [
        {"type": "text", "text": "开户信息"},
        {
            "type": "image",
            "img_path": "images/account.jpg",
            "bbox": [1, 2, 3, 4],
            "page_idx": 2,
        },
    ]
    vision = FakeVision(bank_extraction())
    ocr = FakeOCR("户名：甲公司\n开户银行：甲银行\n账号：110914414810101")
    service = ContractImageIngestionService(
        vision_service=vision,
        ocr_service=ocr,
    )

    enriched = service.enrich_images(objects, storage_dir=tmp_path)

    assert len(enriched) == 2
    assert enriched[0] == objects[0]
    image = enriched[1]
    assert image["img_path"] == "images/account.jpg"
    assert image["image_type"] == "bank_account"
    assert image["structured_data"]["account_number"] == "110914414810101"
    assert image["ocr_status"] == "ready"
    assert image["verification_status"] == "verified"
~~~

Assert the service logs source_object_index=1 but does not store that index as an invented MinerU field; Node construction derives it from enumeration.

- [ ] **Step 2: Add degradation tests**

Cover:

~~~python
def test_missing_image_is_recorded_and_other_images_continue(tmp_path):
    existing = tmp_path / "images" / "general.jpg"
    existing.parent.mkdir()
    existing.write_bytes(b"jpeg")
    objects = [
        {"type": "image", "img_path": "images/missing.jpg"},
        {"type": "image", "img_path": "images/general.jpg"},
    ]
    service = ContractImageIngestionService(
        vision_service=SequenceVision([general_extraction()]),
        ocr_service=FakeOCR("must not be called"),
    )
    enriched = service.enrich_images(objects, storage_dir=tmp_path)
    assert enriched[0]["image_processing_status"] == "missing_image"
    assert enriched[1]["image_processing_status"] == "ready"


def test_schema_error_is_recorded_without_ocr(tmp_path):
    image_path = write_test_image(tmp_path, "images/bad.jpg")
    vision = RaisingVision(ImageSchemaError("invalid schema"))
    ocr = FakeOCR("must not be called")
    service = ContractImageIngestionService(
        vision_service=vision,
        ocr_service=ocr,
    )
    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]
    assert image["image_processing_status"] == "schema_invalid"
    assert image["ocr_status"] == "not_started"
    assert ocr.calls == []


def test_general_never_calls_ocr(tmp_path):
    image_path = write_test_image(tmp_path, "images/general.jpg")
    ocr = FakeOCR("must not be called")
    service = ContractImageIngestionService(
        vision_service=FakeVision(general_extraction()),
        ocr_service=ocr,
    )
    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]
    assert image["ocr_status"] == "not_required"
    assert image["verification_status"] == "not_required"
    assert ocr.calls == []


def test_ocr_failure_keeps_bank_structured_data(tmp_path):
    image_path = write_test_image(tmp_path, "images/account.jpg")
    service = ContractImageIngestionService(
        vision_service=FakeVision(bank_extraction()),
        ocr_service=RaisingOCR(RuntimeError("ocr unavailable")),
    )
    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]
    assert image["image_processing_status"] == "ready"
    assert image["structured_data"]["account_number"] == "110914414810101"
    assert image["ocr_status"] == "failed"
    assert image["verification_status"] == "insufficient"


def test_conflict_keeps_vl_value_and_ocr_text(tmp_path):
    image_path = write_test_image(tmp_path, "images/account.jpg")
    service = ContractImageIngestionService(
        vision_service=FakeVision(bank_extraction()),
        ocr_service=FakeOCR(
            "户名：甲公司\n开户银行：甲银行\n账号：110914414810107"
        ),
    )
    image = service.enrich_images(
        [{"type": "image", "img_path": image_path.relative_to(tmp_path).as_posix()}],
        storage_dir=tmp_path,
    )[0]
    assert image["structured_data"]["account_number"] == "110914414810101"
    assert image["ocr_text"].endswith("107")
    assert image["verification_status"] == "conflict"
~~~

Also test absolute img_path and ../escape are marked missing_image/path error without file reads outside storage_dir.

- [ ] **Step 3: Run ingestion tests to verify RED**

Run: uv run --with pytest pytest tests/test_image_ingestion.py -q

Expected: FAIL because ContractImageIngestionService does not exist.

- [ ] **Step 4: Implement safe resolution and per-image orchestration**

Use:

~~~python
IMAGE_DERIVED_DEFAULTS = {
    "image_processing_status": "ready",
    "image_type": None,
    "structured_data": None,
    "ocr_status": "not_started",
    "ocr_text": None,
    "verification_status": "insufficient",
    "verification_details": {},
    "image_schema_version": "image-v1",
    "image_model": None,
    "image_error": None,
}
~~~

Deep-copy the input list. Enumerate the copy so logs use merged source indices. For each image:

1. Apply defaults.
2. Resolve a safe ordinary file below storage_dir.
3. Call Vision once.
4. model_dump() only the extraction.data into structured_data.
5. For general, set ocr_status and verification_status to not_required.
6. For bank/id, call OCR in a nested try/except; empty output is ocr_status=empty.
7. Verify only when OCR contains text.
8. Catch ImageClassificationError as unclassified, ImageSchemaError as schema_invalid, and other Vision exceptions as vl_failed.

Use logger.exception/warning without logging image bytes, API keys, or absolute credential paths.

- [ ] **Step 5: Implement atomic JSON persistence and its test**

Test that the final file is valid UTF-8 JSON with a trailing newline and that os.replace is used after the temporary file is fully written.

Implement:

~~~python
def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
~~~

Use a fixed sibling temporary name only within the contract's dedicated processing task. On failure, best-effort remove only that exact temporary path.

- [ ] **Step 6: Run ingestion tests to verify GREEN**

Run: uv run --with pytest pytest tests/test_image_ingestion.py -q

Expected: PASS.

- [ ] **Step 7: Commit the enrichment service**

~~~bash
git add app/image_ingestion.py tests/test_image_ingestion.py
git commit -m "feat: enrich mineru image objects"
~~~

### Task 5: Generate image retrieval context and build Image TextNodes

**Files:**
- Modify: retrieval_context_preprocess.py:41-315
- Modify: mineru_to_nodes.py:71-172
- Modify: tests/test_retrieval_context_preprocess.py
- Modify: tests/test_mineru_to_nodes.py

**Interfaces:**
- Extends generate_contexts(objects, *, llm=None, context_generator=None, concurrency=CONTEXT_LLM_CONCURRENCY) to include image indices that have searchable text.
- Keeps existing generate_contexts signature and text/table outputs compatible.
- Extends build_nodes(objects, *, retrieval_contexts=None) to append one Image TextNode for each searchable enriched image.

- [ ] **Step 1: Write the image context test**

~~~python
def test_context_generation_uses_image_text_section_and_nearby_body():
    llm = RecordingLLM(response="位于开户资料章节的银行账户图片")
    objects = [
        {"type": "text", "text": "第五条 开户资料", "text_level": 2},
        {"type": "text", "text": "以下为乙方收款账户：", "page_idx": 4},
        {
            "type": "image",
            "page_idx": 4,
            "image_caption": ["收款账户"],
            "image_footnote": [],
            "image_type": "bank_account",
            "structured_data": {
                "account_name": "乙方公司",
                "account_number": "110914414810101",
                "bank_name": "甲银行",
                "bank_branch": None,
            },
            "verification_status": "verified",
        },
        {"type": "text", "text": "转账时请备注合同编号。", "page_idx": 4},
    ]

    contexts = generate_contexts(objects, llm=llm, concurrency=1)

    assert contexts[2] == "位于开户资料章节的银行账户图片"
    prompt = llm.prompts[-1]
    assert "第五条 开户资料" in prompt
    assert "以下为乙方收款账户" in prompt
    assert "转账时请备注合同编号" in prompt
    assert "110914414810101" in prompt
~~~

Add a test that empty/failed general images produce no context entry and a regression assertion that existing text/table context mappings are unchanged.

- [ ] **Step 2: Run the focused context test to verify RED**

Run: uv run --with pytest pytest tests/test_retrieval_context_preprocess.py -q

Expected: new image context test fails because images are currently ignored.

- [ ] **Step 3: Implement the image context branch**

Add:

~~~python
def _image_objects(
    objects: list[dict],
) -> list[tuple[int, dict, str]]:
    found = []
    for source_index, obj in enumerate(objects):
        if obj.get("type") != "image":
            continue
        searchable_text = image_to_searchable_text(obj)
        if searchable_text:
            found.append((source_index, obj, searchable_text))
    return found

def _nearby_texts(
    objects: list[dict],
    target_index: int,
    *,
    max_each_side: int = 2,
    max_chars: int = 600,
) -> list[str]:
    before = [
        obj["text"].strip()
        for obj in reversed(objects[:target_index])
        if obj.get("type") == "text"
        and isinstance(obj.get("text"), str)
        and obj["text"].strip()
    ][:max_each_side]
    after = [
        obj["text"].strip()
        for obj in objects[target_index + 1 :]
        if obj.get("type") == "text"
        and isinstance(obj.get("text"), str)
        and obj["text"].strip()
    ][:max_each_side]
    ordered = [*reversed(before), *after]
    remaining = max_chars
    limited = []
    for text in ordered:
        if remaining <= 0:
            break
        value = text[:remaining]
        limited.append(value)
        remaining -= len(value)
    return limited

def build_image_context_prompt(
    image_input: str,
    section_path: list[str],
) -> str:
    section_text = " > ".join(section_path) or "未识别到章节标题"
    return (
        "你是合同检索预处理器。请为当前 image 生成简短的 "
        "retrieval_context。\n"
        "只描述图片在合同中的章节定位、主题或用途；附近正文只用于"
        "定位，不得把附近正文事实写成图片识别事实。无法确定时输出"
        "空字符串。只输出 retrieval_context。\n\n"
        f"章节路径：\n{section_text}\n\n"
        f"图片和附近内容：\n{image_input.strip()}"
    )
~~~

Build image_input with labeled searchable text, caption/footnote, and nearby text. Reuse _generate_one_context(), retry count, normalization, concurrency, and section fallback. Merge image mappings after the existing text and table mappings. Do not modify text/table prompt text.

- [ ] **Step 4: Write Image Node tests**

~~~python
def test_build_nodes_creates_image_text_node_with_raw_reference():
    obj = {
        "type": "image",
        "img_path": "images/account.jpg",
        "page_idx": 4,
        "bbox": [1, 2, 3, 4],
        "image_caption": ["收款账户"],
        "image_footnote": [],
        "content": "",
        "sub_type": "image",
        "image_processing_status": "ready",
        "image_type": "bank_account",
        "structured_data": {
            "account_name": "乙方公司",
            "account_number": "110914414810101",
            "bank_name": "甲银行",
            "bank_branch": None,
        },
        "ocr_status": "ready",
        "ocr_text": "账号 110914414810101",
        "verification_status": "verified",
        "verification_details": {
            "account_number": {"status": "verified"}
        },
    }

    node = build_nodes([obj], retrieval_contexts={0: "开户资料章节"})[0]

    assert node.metadata["node_type"] == "image"
    assert node.metadata["source_object_index"] == 0
    assert node.metadata["img_path"] == "images/account.jpg"
    assert node.metadata["structured_data"]["account_number"] == "110914414810101"
    assert node.metadata["verification_status"] == "verified"
    assert "银行账号：110914414810101" in node.text
    embedded = node.get_content(metadata_mode=MetadataMode.EMBED)
    assert "开户资料章节" in embedded
    assert "银行账号：110914414810101" in embedded
    assert "img_path" not in embedded
    assert "structured_data" not in embedded
~~~

Add tests that failed/empty images create no nodes and that the existing text+table fixture still produces the same two nodes in the same branch order.

- [ ] **Step 5: Run Node tests to verify RED**

Run: uv run --with pytest pytest tests/test_mineru_to_nodes.py -q

Expected: Image Node test fails because build_nodes currently handles only text and table.

- [ ] **Step 6: Implement the Image Node branch**

After the existing table loop, enumerate objects and handle type=image. Call image_to_searchable_text(obj); skip None. Use:

~~~python
metadata = {
    "node_type": "image",
    "retrieval_context": retrieval_context,
    "source_object_index": source_index,
    "page_idx": obj.get("page_idx"),
    "bbox": obj.get("bbox"),
    "img_path": obj.get("img_path"),
    "image_type": obj.get("image_type"),
    "structured_data": obj.get("structured_data"),
    "ocr_text": obj.get("ocr_text"),
    "ocr_status": obj.get("ocr_status"),
    "verification_status": obj.get("verification_status"),
    "verification_details": obj.get("verification_details"),
    "image_processing_status": obj.get("image_processing_status"),
    "image_caption": obj.get("image_caption"),
    "image_footnote": obj.get("image_footnote"),
    "content": obj.get("content"),
    "sub_type": obj.get("sub_type"),
}
~~~

Remove only None and empty-string values. Exclude every metadata key except retrieval_context from embedding. Keep text and table branches unchanged.

- [ ] **Step 7: Run context and Node regression tests**

Run: uv run --with pytest pytest tests/test_retrieval_context_preprocess.py tests/test_mineru_to_nodes.py -q

Expected: PASS.

- [ ] **Step 8: Commit context and Node support when safe**

~~~bash
git diff -- retrieval_context_preprocess.py mineru_to_nodes.py tests/test_retrieval_context_preprocess.py tests/test_mineru_to_nodes.py
git diff --cached --name-only
git add retrieval_context_preprocess.py mineru_to_nodes.py tests/test_retrieval_context_preprocess.py tests/test_mineru_to_nodes.py
git commit -m "feat: build retrieval context and nodes for images"
~~~

Do not execute git add if any listed path contains unrelated user hunks that have not been checkpointed.

### Task 6: Insert image enrichment into contract ingestion

**Files:**
- Modify: app/pipeline.py:24-140
- Modify: tests/test_contract_pipeline.py

**Interfaces:**
- Adds optional ContractProcessor(repository, settings, index_manager, *, embedding_model=None, image_ingestion_service: ContractImageIngestionService | None = None).
- Produces build_default_image_ingestion_service(settings: Settings) -> ContractImageIngestionService.
- Runs enrichment after merged JSON load and before generate_contexts().
- Writes enriched objects atomically back to the existing merged path.

- [ ] **Step 1: Write an orchestration-order test**

~~~python
def test_process_enriches_images_before_context_and_nodes():
    merged_objects = [
        {
            "type": "image",
            "img_path": "images/general.jpg",
            "page_idx": 1,
        }
    ]
    enriched_objects = [
        {
            **merged_objects[0],
            "image_processing_status": "ready",
            "image_type": "general",
            "structured_data": {
                "visible_text": "营业执照",
                "content_description": "证照图片",
            },
            "ocr_status": "not_required",
            "verification_status": "not_required",
            "verification_details": {},
        }
    ]
    image_service = Mock()
    image_service.enrich_images.return_value = enriched_objects

    # Patch clean/merge/context/nodes/index as in existing processor tests.
    result = processor.process(contract.contract_id, mode="reuse_existing")

    image_service.enrich_images.assert_called_once()
    assert captured["context_objects"] == enriched_objects
    assert captured["node_objects"] == enriched_objects
    persisted = json.loads(
        (contract_dir / "merged_content_list.json").read_text()
    )
    assert persisted == enriched_objects
    assert result.status == "ready"
~~~

Add a no-image regression test showing no default Vision/OCR service is constructed and existing call order remains parse/clean/merge/context/nodes/index/persist/cache.

- [ ] **Step 2: Run pipeline tests to verify RED**

Run: uv run --with pytest pytest tests/test_contract_pipeline.py -q

Expected: the new constructor argument or enrichment assertion fails.

- [ ] **Step 3: Implement the default service factory**

In app/pipeline.py:

~~~python
def build_default_image_ingestion_service(
    settings: Settings,
) -> ContractImageIngestionService:
    vision = ImageUnderstandingService(
        model_name=settings.image_vision_model,
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        timeout_seconds=settings.image_vision_timeout_seconds,
    )
    ocr = MinerUImageOCRService(
        svr_url=settings.mineru_url,
        backend=settings.mineru_backend,
        server_url=settings.mineru_server_url,
    )
    return ContractImageIngestionService(
        vision_service=vision,
        ocr_service=ocr,
    )
~~~

Keep ImageUnderstandingService and MinerUImageOCRService imports inside build_default_image_ingestion_service so tests can configure environment variables before model modules are imported.

- [ ] **Step 4: Insert enrichment and atomic save**

After loading and validating objects:

~~~python
if any(
    isinstance(obj, dict) and obj.get("type") == "image"
    for obj in objects
):
    image_service = (
        self.image_ingestion_service
        or build_default_image_ingestion_service(self.settings)
    )
    objects = image_service.enrich_images(
        objects,
        storage_dir=Path(contract.storage_dir),
    )
    write_json_atomic(paths["merged"], objects)

contexts = generate_contexts(objects)
save_retrieval_contexts(contexts, paths["context"])
nodes = build_nodes(objects, retrieval_contexts=contexts)
~~~

Do not add a new sidecar path. Per-image failures are already captured inside enriched objects; only unexpected orchestration or atomic-write failure reaches the existing contract-level exception handler.

- [ ] **Step 5: Run pipeline and existing clean/merge tests**

Run: uv run --with pytest pytest tests/test_contract_pipeline.py tests/test_clean_mineru_data.py tests/test_merge_cross_page_paragraphs.py -q

Expected: PASS.

- [ ] **Step 6: Commit orchestration only if dirty-file scope is safe**

app/pipeline.py and tests/test_contract_pipeline.py already have user changes. Inspect them:

~~~bash
git diff -- app/pipeline.py tests/test_contract_pipeline.py
git diff --cached --name-only
~~~

If prior hunks have been separately checkpointed, commit:

~~~bash
git add app/pipeline.py tests/test_contract_pipeline.py
git commit -m "feat: ingest contract images before indexing"
~~~

Otherwise leave both files uncommitted and continue without staging unrelated work.

### Task 7: Share Evidence serialization and preserve image references

**Files:**
- Create: app/evidence_serialization.py
- Modify: app/qa.py:9-46
- Modify: app/evaluation_metrics.py:10-137
- Modify: tests/test_app_qa.py
- Modify: tests/test_evaluation_metrics.py

**Interfaces:**
- Produces serialize_node_result(result: Any) -> dict[str, Any].
- text defaults to node_type=text for old/new nodes without explicit metadata.
- table retains current fields.
- image includes original path, structured data, OCR and verification metadata in every serialized stage.

- [ ] **Step 1: Add serializer-focused tests through both public consumers**

Define an image result helper:

~~~python
def image_result_for(index: int, text: str, score: float = 0.9):
    return SimpleNamespace(
        node=SimpleNamespace(
            node_id=f"image-node-{index}",
            text=text,
            metadata={
                "node_type": "image",
                "source_object_index": index,
                "page_idx": 4,
                "bbox": [1, 2, 3, 4],
                "img_path": "images/account.jpg",
                "image_type": "bank_account",
                "structured_data": {
                    "account_name": "甲公司",
                    "account_number": "110914414810101",
                    "bank_name": "甲银行",
                    "bank_branch": None,
                },
                "ocr_text": "账号 110914414810101",
                "ocr_status": "ready",
                "verification_status": "verified",
                "verification_details": {
                    "account_number": {"status": "verified"}
                },
                "image_processing_status": "ready",
            },
        ),
        score=score,
    )
~~~

In tests/test_evaluation_metrics.py, pass this result through vector_results, reranked_results, and selected_nodes, then assert every list entry preserves node_type, img_path, image_type, structured_data, verification_status, and text.

In tests/test_app_qa.py, call answer_question(FakeIndex([image_result]), "账号是什么？", debug=True, reranker=FakeReranker(), selector_llm=FakeLLM({"evidence_indices": [12]}), answer_llm=FakeLLM({"answer": "账号为110914414810101。"})) and assert evidence plus debug.rerank_top10 preserve the same fields.

Add a text result assertion:

~~~python
assert serialized_text["node_type"] == "text"
~~~

- [ ] **Step 2: Run QA and metric tests to verify RED**

Run: uv run --with pytest pytest tests/test_app_qa.py tests/test_evaluation_metrics.py -q

Expected: image fields and text node_type assertions fail.

- [ ] **Step 3: Implement the shared serializer**

~~~python
COMMON_METADATA_KEYS = (
    "page_idx",
    "start_page_idx",
    "end_page_idx",
    "source_page_indices",
    "source_bboxes",
    "merged_cross_page",
    "retrieval_context",
)

TABLE_METADATA_KEYS = (
    "bbox",
    "table_body",
    "table_caption",
    "table_footnote",
)

IMAGE_METADATA_KEYS = (
    "bbox",
    "img_path",
    "image_type",
    "structured_data",
    "ocr_text",
    "ocr_status",
    "verification_status",
    "verification_details",
    "image_processing_status",
)

def serialize_node_result(result: Any) -> dict[str, Any]:
    node = result.node
    metadata = getattr(node, "metadata", {}) or {}
    node_type = metadata.get("node_type") or "text"
    serialized = {
        "node_type": node_type,
        "source_object_index": metadata.get("source_object_index"),
        "node_id": getattr(node, "node_id", None),
        "text": getattr(node, "text", ""),
    }
    for key in COMMON_METADATA_KEYS:
        if key in metadata and metadata[key] is not None:
            serialized[key] = metadata[key]
    type_keys = (
        TABLE_METADATA_KEYS
        if node_type == "table"
        else IMAGE_METADATA_KEYS
        if node_type == "image"
        else ()
    )
    for key in type_keys:
        if key in metadata and metadata[key] is not None:
            serialized[key] = metadata[key]
    if getattr(result, "score", None) is not None:
        serialized["score"] = result.score
    if metadata.get("retrieval_score") is not None:
        serialized["retrieval_score"] = metadata["retrieval_score"]
    return serialized
~~~

Only add keys whose values are not None. Keep empty lists/dicts when they are meaningful structured fields. Include score and retrieval_score exactly as current serializers do.

Replace app.qa._serialize_result and app.evaluation_metrics._serialize_result with imports and calls to serialize_node_result. Do not modify RAGPipeline, retrieve/rerank, Recall, Selector, or Answer code.

- [ ] **Step 4: Run serialization and RAG regression tests**

Run: uv run --with pytest pytest tests/test_app_qa.py tests/test_evaluation_metrics.py tests/test_rag_pipeline.py tests/test_retrieval_evaluation.py -q

Expected: PASS.

- [ ] **Step 5: Commit serializer changes when safe**

~~~bash
git diff -- app/evidence_serialization.py app/qa.py app/evaluation_metrics.py tests/test_app_qa.py tests/test_evaluation_metrics.py
git diff --cached --name-only
git add app/evidence_serialization.py app/qa.py app/evaluation_metrics.py tests/test_app_qa.py tests/test_evaluation_metrics.py
git commit -m "feat: preserve image metadata in rag evidence"
~~~

If any existing modified file contains unrelated user hunks, stage only after those hunks have a separate checkpoint; otherwise defer the commit.

### Task 8: Display Image Evidence in existing chat and Evaluation pages

**Files:**
- Modify: app/templates/chat.html:232-263
- Modify: app/templates/evaluation.html:239-258
- Modify: tests/test_app_page.py
- Modify: tests/test_app_evaluation_page.py

**Interfaces:**
- Consumes serialized image fields from Task 7.
- Keeps existing routes, Evaluation cases, run storage, Recall metrics, and page layout.
- Adds no image-specific page and no binary image route.

- [ ] **Step 1: Add chat page rendering test**

Extend the fake answer result with image Evidence and assert the response contains:

~~~python
image_evidence = {
    "node_type": "image",
    "source_object_index": 12,
    "page_idx": 4,
    "text": "银行账户信息。\n银行账号：110914414810101。",
    "img_path": "images/account.jpg",
    "image_type": "bank_account",
    "structured_data": {
        "account_name": "甲公司",
        "account_number": "110914414810101",
        "bank_name": "甲银行",
        "bank_branch": None,
    },
    "verification_status": "verified",
}

assert "node_type=image" in response.text
assert "bank_account" in response.text
assert "images/account.jpg" in response.text
assert "110914414810101" in response.text
assert "verified" in response.text
~~~

Cover both normal Selected Evidence and debug Rerank display.

- [ ] **Step 2: Add Evaluation page rendering test**

Put the same image payload into vector_results, reranked_results, and selected_nodes of a saved run result. Assert all three section headings remain and the image metadata appears in the rendered page. Keep expected_source_object_indices=[12] and assert the correct Node ID remains visible.

- [ ] **Step 3: Run page tests to verify RED**

Run: uv run --with pytest pytest tests/test_app_page.py tests/test_app_evaluation_page.py -q

Expected: node_type and image metadata strings are not rendered.

- [ ] **Step 4: Add a compact image metadata block to existing cards**

In each chat evidence/debug card and each Evaluation result loop:

~~~jinja2
<small>
  node_type={{ item.node_type }}
  · source_object_index={{ item.source_object_index }}
  {% if item.page_idx is defined %} · page={{ item.page_idx }}{% endif %}
</small>
{% if item.node_type == "image" %}
  <div class="image-meta">
    <div>image_type={{ item.image_type }}</div>
    <div>verification_status={{ item.verification_status }}</div>
    <div>img_path={{ item.img_path }}</div>
    <pre>{{ item.structured_data | tojson(indent=2) }}</pre>
  </div>
{% endif %}
<div>{{ item.text }}</div>
~~~

Use the variable name already present in each loop (item or node). Do not create a new route, static mount, page, JavaScript bundle, or actual img tag.

- [ ] **Step 5: Run page and API tests**

Run: uv run --with pytest pytest tests/test_app_page.py tests/test_app_evaluation_page.py tests/test_app_api.py -q

Expected: PASS.

- [ ] **Step 6: Commit template work only after reviewing existing hunks**

These templates and page tests already contain user changes:

~~~bash
git diff -- app/templates/chat.html app/templates/evaluation.html tests/test_app_page.py tests/test_app_evaluation_page.py
git diff --cached --name-only
~~~

If prior changes have a safe checkpoint:

~~~bash
git add app/templates/chat.html app/templates/evaluation.html tests/test_app_page.py tests/test_app_evaluation_page.py
git commit -m "feat: show image evidence in rag pages"
~~~

Otherwise leave the changes uncommitted and do not stage unrelated page work.

### Task 9: Verify end-to-end ingestion, regression behavior, and repository hygiene

**Files:**
- Create: tests/test_image_rag_integration.py
- No production behavior should be added in this task.

**Interfaces:**
- Verifies the full enriched image object → context → TextNode → serializer contract.
- Verifies all existing text/table/chat/Evaluation behavior.

- [ ] **Step 1: Add one complete mocked integration test if no prior test spans the whole boundary**

The test must use real pure enrichment, verification, searchable text, context generation with a fake context generator, and build_nodes, while mocking only VL, OCR, embedding/index persistence:

~~~python
def test_image_flows_from_enrichment_to_context_node_and_evidence(tmp_path):
    image_path = tmp_path / "images" / "account.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"jpeg")
    extraction = validate_image_extraction(
        {
            "image_type": "bank_account",
            "data": {
                "account_name": "测试甲公司",
                "account_number": "110914414810101",
                "bank_name": "测试甲银行",
                "bank_branch": None,
            },
        }
    )
    objects = [
        {"type": "text", "text": "第五条 开户资料", "text_level": 2},
        {
            "type": "image",
            "img_path": "images/account.jpg",
            "page_idx": 4,
            "bbox": [1, 2, 3, 4],
            "image_caption": [],
            "image_footnote": [],
            "content": "",
        },
    ]
    ingestion = ContractImageIngestionService(
        vision_service=FakeVision(extraction),
        ocr_service=FakeOCR(
            "户名：测试甲公司\n开户银行：测试甲银行\n"
            "账号：110914414810101"
        ),
    )

    enriched = ingestion.enrich_images(objects, storage_dir=tmp_path)
    contexts = generate_contexts(
        enriched,
        context_generator=lambda _text, _path: "开户资料章节",
        concurrency=1,
    )
    image_node = build_nodes(
        enriched,
        retrieval_contexts=contexts,
    )[1]
    evidence = serialize_node_result(
        SimpleNamespace(node=image_node, score=0.95)
    )

    assert enriched[1]["verification_status"] == "verified"
    assert contexts[1] == "开户资料章节"
    assert "银行账号：110914414810101" in image_node.text
    assert image_node.metadata["img_path"] == "images/account.jpg"
    assert evidence["node_type"] == "image"
    assert evidence["img_path"] == "images/account.jpg"
    assert evidence["structured_data"]["account_name"] == "测试甲公司"
~~~

Define the two local fakes in the same test file:

~~~python
class FakeVision:
    model_name = "test-vl"

    def __init__(self, extraction):
        self.extraction = extraction

    def classify_and_extract(self, _image_path):
        return self.extraction


class FakeOCR:
    def __init__(self, text):
        self.text = text

    def extract_text(self, _image_path):
        return self.text
~~~

Use only the explicit synthetic account values above; do not add real personal data.

- [ ] **Step 2: Run all image-focused tests**

Run:

~~~bash
uv run --with pytest pytest \
  tests/test_mineru_raw_parse.py \
  tests/test_image_ocr.py \
  tests/test_image_schemas.py \
  tests/test_image_understanding.py \
  tests/test_image_verification.py \
  tests/test_image_searchable_text.py \
  tests/test_image_ingestion.py \
  tests/test_retrieval_context_preprocess.py \
  tests/test_mineru_to_nodes.py \
  tests/test_contract_pipeline.py \
  tests/test_app_qa.py \
  tests/test_evaluation_metrics.py \
  tests/test_app_page.py \
  tests/test_app_evaluation_page.py -q
~~~

Expected: PASS with no real network calls.

- [ ] **Step 3: Run the full test suite**

Run: uv run --with pytest pytest tests -q

Expected: all tests PASS. Any pre-existing environment-dependent failure must be reported with its exact test name and output; do not claim completion while a new regression remains.

- [ ] **Step 4: Run static and diff verification**

Run:

~~~bash
./.venv/bin/python -m compileall -q .
git diff --check
git status --short
git diff --stat
~~~

Expected: compileall exit 0, git diff --check produces no output, and status shows only intentional implementation changes plus the user's known pre-existing changes.

- [ ] **Step 5: Perform local-service smoke checks only when safe fixtures exist**

Check MinerU without uploading contract data:

~~~bash
curl -fsS --max-time 5 http://127.0.0.1:7100/health
~~~

If the workspace contains an explicitly designated non-sensitive test contract with bank/id/general images, run from_scratch ingestion through the existing Web/API and record the resulting source_object_index, image_type, verification_status, Vector rank, Rerank rank, Selected Evidence, and Answer. If no such safe fixture exists, report the real-model smoke test as not run; do not reuse real identity or bank images from unrelated contracts and do not manufacture a success claim.

- [ ] **Step 6: Create the final implementation commit only for intentional unstaged work**

Review every remaining hunk:

~~~bash
git diff
git diff --cached --name-only
~~~

If all image-RAG hunks are isolated from the user's earlier changes, stage the exact intended files and commit:

~~~bash
git commit -m "test: verify contract image rag integration"
~~~

If they are not safely separable, do not commit them. Report the verified working-tree state and the files whose commit was deferred.

---

## Completion checklist

- MinerU images exist at storage_dir/img_path for new from_scratch parses.
- Existing raw-only historical contracts degrade missing images unless reparsed from scratch.
- Each image receives at most one VL business request.
- general never receives an OCR request.
- bank/id OCR mismatch never overwrites VL data.
- enriched image fields are visible in merged_content_list.json.
- Failed images carry diagnosable status/error fields and do not stop other images.
- Image Node uses deterministic text and preserves img_path.
- Vector, Rerank, Selector, Answer and Evaluation require no image-specific algorithm branch.
- Chat and Evaluation expose actual LLM evidence text and image metadata.
- Existing text/table tests and full suite pass.
- No CLIP, multimodal embedding, new vector store, PaddleOCR, Agent or binary image route was added.
