# Contract Image RAG Design

## Goal

为现有合同 RAG Web 项目增加第一版图片内容入库能力，使银行账户图片、法人身份证图片和普通图片能够经过结构化识别、必要的 MinerU OCR 校验、检索上下文增强和 `TextNode` 构建，进入现有 Vector → Rerank → Evidence Selector → Answer 链路，并在聊天和 Evaluation 结果中保留原始图片引用。

本次不建设通用多模态 RAG。图片只转换为确定性的文本表示并进入现有文本向量索引，不引入 CLIP、多模态 embedding、新向量库、图片相似度搜索或复杂 Agent。

## Existing behavior and data

### MinerU image object

当前项目中的 MinerU image 对象实际包含以下字段：

```json
{
  "type": "image",
  "img_path": "images/<hash>.jpg",
  "image_caption": [],
  "image_footnote": [],
  "content": "",
  "sub_type": "seal",
  "bbox": [260, 183, 453, 323],
  "page_idx": 32
}
```

其中 `sub_type` 是可选字段；当前本地样本中的 image 均为 `sub_type=seal`。第一版类型体系只有 `bank_account`、`identity_card` 和 `general`，因此印章等其他图片归入 `general`。

清洗只删除 header、page number 和特定无效 text，对 image 对象不做改写。跨页合并只合并满足条件的 text，相邻 image 对象也会原样保留。

RAG 和 Evaluation 使用的 `source_object_index` 是 `merged_content_list.json` 的数组下标。它可能与 `raw_content_list.json` 中的下标不同，因为清洗会删除对象。所有图片识别结果、Node metadata 和评测标注必须使用 merged 下标。

### Image file availability

当前 MinerU 请求已经设置：

```text
return_images=true
response_format_zip=true
```

但客户端下载 ZIP 后只提取 `*_content_list.json`，没有保存 ZIP 中的 `images/` 文件。因此已有合同虽然包含 `img_path`，实际合同目录里没有对应图片。

图片的稳定定位规则是：

```text
absolute_image_path = contract.storage_dir / image_object["img_path"]
```

例如：

```text
data/contracts/<contract_id>/images/<hash>.jpg
```

新的 MinerU 解析必须安全提取 content list 明确引用的图片。已经只保存 JSON 的历史合同无法通过 `reuse_existing` 恢复图片，必须执行 `from_scratch` 重新解析；缺图时仍按单图降级规则继续入库。

### Existing RAG path

现有入库顺序是：

```text
parse → clean → merge → retrieval_context → build_nodes → VectorStoreIndex
```

text 和 table 都构造 LlamaIndex `TextNode`。table 已使用“确定性 searchable text + retrieval_context + 原始 metadata”的方式接入现有文本 RAG。图片沿用这一模式，不建立独立索引。

Rerank 使用 `node.get_content(metadata_mode=MetadataMode.EMBED)`；Evidence Selector 和 Answer Generator 使用 `node.text`。因此 Image Node 的 `node.text` 必须是实际提供给 LLM 的、可审计的确定性图片证据文本。

## Architecture decision

图片识别结果直接补充到 `merged_content_list.json` 中对应的 image 对象，不创建独立 `image_understanding.json`。

原因是现有 Evaluation“查看解析结果”入口直接读取 `merged_content_list.json`。将图片派生结果放入独立文件会让现有页面看不到结果，并形成两套需要关联的数据源。图片处理只原位增加已知字段，不删除或改写 MinerU 原始字段，不改变数组长度和 `source_object_index`。

处理流程调整为：

```text
MinerU ZIP
  → 保存 raw_content_list.json
  → 安全提取 content list 引用的 images/*
  → clean
  → merge
  → 读取 merged objects
  → 逐张 image 执行 VL 分类和结构化提取
  → bank_account / identity_card 额外执行 MinerU OCR
  → 确定性校验 VL 与 OCR
  → 把结果补充到对应 image object
  → 原子写回 merged_content_list.json
  → 为 text/table/image 生成 retrieval_context
  → 为 text/table/image 构造 TextNode
  → 进入现有 VectorStoreIndex
```

模型调用不进入 Node 构建函数。Node 构建只消费已经补充完成的普通 Python/JSON 对象，因此可重复、可单测，也不会在重建 Node 时意外重复调用 VL 或 OCR。

## Services and responsibilities

### `ImageUnderstandingService`

独立 Vision Service 提供：

```python
class ImageUnderstandingService:
    def classify_and_extract(self, image_path: Path) -> ImageExtraction:
        """一次 VL 请求完成图片分类和对应字段提取。"""
```

职责：

- 验证并读取图片文件；
- 根据扩展名或已知文件类型生成 MIME type；
- 构造 base64 data URL 多模态消息；
- 一次请求同时输出 `image_type` 和对应 `data`；
- 解析响应并执行严格的本地 Schema 校验；
- 不负责 OCR、VL/OCR 一致性校验、上下文生成、Node 构建或文件持久化。

模型通过新增的 `IMAGE_VISION_MODEL` 配置选择。未显式配置时回退到项目的 `LLM_MODEL`，继续复用现有 `LLM_API_KEY` 和 `LLM_BASE_URL`。请求使用 `temperature=0`、`enable_thinking=false`、`max_retries=0`，不设置可能截断 JSON 的 token 上限。

每张图片只发起一次业务 VL 请求。分类和字段提取不得拆成两次调用；Schema 失败时也不额外调用模型修复 JSON。

### `MinerUImageOCRService`

OCR Service 提供：

```python
class MinerUImageOCRService:
    def extract_text(self, image_path: Path) -> str:
        """使用现有 MinerU task API 对一张原图执行 OCR。"""
```

它复用现有 MinerU 提交、轮询和 ZIP 下载能力，不引入 PaddleOCR 等依赖。仅 `bank_account` 和 `identity_card` 调用；`general` 不调用。

单图 OCR 使用：

```text
parse_method=ocr
formula_enable=false
table_enable=false
image_analysis=false
return_md=false
return_middle_json=false
return_model_output=false
return_content_list=true
return_images=false
response_format_zip=true
```

MinerU 返回的 content list 按文档顺序提取有效文字并拼接为 `ocr_text`。OCR 输出只用于一致性校验，不覆盖 VL 结构化结果。

### `ContractImageIngestionService`

图片入库编排 Service 提供：

```python
class ContractImageIngestionService:
    def enrich_images(
        self,
        objects: list[dict[str, Any]],
        *,
        storage_dir: Path,
    ) -> list[dict[str, Any]]:
        """处理所有 image，并返回保留原下标的 enriched objects。"""
```

职责：

- 只处理 `type=image` 对象；
- 安全解析 `img_path`；
- 调用 Vision Service；
- 根据分类决定是否调用 MinerU OCR；
- 调用确定性验证函数；
- 生成每张图的处理状态和错误字段；
- 捕获单图异常并继续处理后续对象；
- 保留所有非 image 对象和全部 MinerU 原始字段。

第一版按数组顺序串行处理图片，避免额外的模型/OCR 并发控制和服务压力。并发优化不属于本次范围。

## Structured output schemas

JSON Schema 放在 `app/image_schemas.py`，定义三个独立 data Schema，并组装为一次请求使用的结构化输出 Schema。本地再使用与 Schema 对应的 Pydantic 模型或等价严格校验器校验 `image_type` 和 `data` 的匹配关系。

所有 Schema 都要求顶层只有 `image_type` 和 `data`，`additionalProperties=false`。无法可靠识别的字段必须返回 `null` 或空字符串，不允许猜测。

### Bank account

```json
{
  "image_type": "bank_account",
  "data": {
    "account_name": null,
    "account_number": null,
    "bank_name": null,
    "bank_branch": null
  }
}
```

`bank_branch` 是第一版唯一允许补充的银行字段，不继续扩展无限字段。

### Identity card

```json
{
  "image_type": "identity_card",
  "data": {
    "name": null,
    "id_number": null,
    "valid_from": null,
    "valid_to": null
  }
}
```

只看到正面或反面时，另一面的字段必须为空。日期保持图片可见语义，不根据身份证号码等信息推断。

### General image

```json
{
  "image_type": "general",
  "data": {
    "visible_text": null,
    "content_description": null
  }
}
```

`visible_text` 忠实保留实际可见文字；`content_description` 只描述可见内容，不添加图片中不存在的合同事实。

项目当前文本模型已经使用 `response_format.type=json_schema` 和 `strict=true`。Vision Service 优先沿用同一方式。由于模型服务对 VL strict JSON Schema 的支持可能与文本模型不同，真实实现完成后必须用项目实际 endpoint 做集成冒烟测试；如果 endpoint 不接受该 Schema，视为明确的部署兼容问题，不在运行时静默切换为另一次修复调用。

## Safe image extraction and path handling

MinerU ZIP 解包只保存：

- 唯一的 `*_content_list.json`；
- content list 中 image 对象明确引用的 `img_path` 文件。

不得直接对整个 ZIP 使用不受约束的 `extractall()`。每个成员需要：

- 把反斜杠统一为 POSIX 分隔符；
- 拒绝绝对路径、空路径、`.` 和 `..` 路径段；
- 根据 content list 成员所在目录解析相对图片成员；
- 防止写出 `output_path.parent`；
- 如果同一引用匹配多个 ZIP 成员则拒绝模糊选择；
- 使用二进制写入创建实际的 `<storage_dir>/images/...` 文件。

ZIP 中缺少某个被引用图片时，仍保存 content list 并记录警告。后续图片处理将该对象标记为 `missing_image`，不让单张图片缺失导致合同失败。

图片读取再次验证：

```text
resolved_path = (storage_dir / img_path).resolve()
```

`img_path` 必须是非空相对路径，且 `resolved_path` 必须仍位于 `storage_dir` 内并是普通文件。Node metadata 和 Evidence 始终保存原始相对 `img_path`，不保存机器相关的绝对路径。

## Deterministic OCR verification

一致性校验放在纯函数模块 `image_verification.py`，不得调用模型。

### Normalization

- 使用 Unicode NFKC；
- 去除空格、换行和常见显示分隔符；
- 身份证字母统一为大写；
- 不使用编辑距离、拼音、语义相似度或容错替换；
- 不把一位不同的账号或身份证号强行视为相同。

### Candidate extraction

- 银行账号：从 OCR 文本中提取合理长度的数字序列，允许原文中存在空格或常见分隔符；
- 身份证号：提取 `17` 位数字加数字或 `X` 的候选；
- 户名、姓名、银行名：优先读取带 `户名`、`账户名称`、`姓名`、`开户行`、`银行` 等明确标签的同行或相邻内容；
- 有效期：只比较能够可靠对应 `valid_from`、`valid_to` 的日期候选。

单字段状态为：

```text
verified     VL 值与对应 OCR 候选归一化后完全一致
conflict     OCR 中存在可信的对应候选，但与 VL 值明确不同
insufficient VL 值为空，或 OCR 没有可可靠对应的候选
```

整体状态规则：

- 任一关键字段 `conflict`，整体为 `conflict`；
- bank_account 的 `account_name`、`account_number`、`bank_name` 全部验证后，整体为 `verified`；
- identity_card 的 `name`、`id_number` 验证，且 VL 中非空的有效期字段也一致后，整体为 `verified`；
- 其他 bank/id 情况为 `insufficient`；
- general 没有 OCR 校验，状态为 `not_required`。

验证详情按字段保存 VL 值、匹配状态和必要的 OCR 候选。出现冲突时保留原始 VL `structured_data` 和 OCR 结果，不自动选择任何一方为正确值。

## Enriched image object

图片处理完成后，原位增加以下顶层字段：

```json
{
  "image_processing_status": "ready",
  "image_type": "bank_account",
  "structured_data": {
    "account_name": "XXX有限公司",
    "account_number": "110914414810101",
    "bank_name": "中国XX银行",
    "bank_branch": null
  },
  "ocr_status": "ready",
  "ocr_text": "户名 XXX有限公司\n账号 110914414810101",
  "verification_status": "verified",
  "verification_details": {
    "account_name": {"status": "verified"},
    "account_number": {"status": "verified"},
    "bank_name": {"status": "verified"}
  },
  "image_schema_version": "image-v1",
  "image_model": "configured-model-name",
  "image_error": null
}
```

这些字段与 MinerU 原始字段并存。处理完所有图片后，对 `merged_content_list.json` 执行一次原子替换写入，避免页面读取到半写入 JSON。

状态字段使用固定枚举：

```text
image_processing_status:
  ready | missing_image | vl_failed | schema_invalid | unclassified | empty_result

ocr_status:
  not_required | not_started | ready | failed | empty
```

`general` 的 `ocr_status` 为 `not_required`。图片在 VL 阶段失败、尚未进入 OCR 时为 `not_started`。bank/id 即使 OCR 失败，只要 VL 结果合法，`image_processing_status` 仍为 `ready`，具体不确定性由 `ocr_status` 和 `verification_status` 表达。

失败对象也写入可排查状态，例如：

```json
{
  "image_processing_status": "vl_failed",
  "image_type": null,
  "structured_data": null,
  "ocr_status": "not_started",
  "ocr_text": null,
  "verification_status": "insufficient",
  "verification_details": {},
  "image_schema_version": "image-v1",
  "image_model": "configured-model-name",
  "image_error": "vision request timeout"
}
```

`reuse_existing` 仍从 raw JSON 重新 clean 和 merge，因此每次下游重处理都会先得到不含旧派生字段的对象，再重新执行图片理解，不会静默沿用旧模型结果。

## Searchable text

`image_searchable_text.py` 根据 `image_type`、`structured_data` 和验证状态生成确定性文本。不得将 `img_path`、JSON 原文或不受约束的自由总结直接作为 embedding 内容。

### Bank account

```text
银行账户信息。
户名：XXX。
开户银行：XXX。
开户支行：XXX。
银行账号：110914414810101。
OCR校验：已核验。
```

### Identity card

```text
法人/身份证信息。
姓名：XXX。
身份证号码：XXX。
有效期限：XXX 至 XXX。
OCR校验：信息不足，尚未确认。
```

### General image

```text
图片内容：XXX。
可见文字：XXX。
```

空字段对应的整行省略，不生成 `None`、空标签或猜测值。bank/id 冲突时增加：

```text
OCR校验：关键字段存在冲突，需要人工核验。
```

如果 VL 未产生合法结果，或者 general 的两个字段都为空，则没有可安全索引的事实，不创建该图片的 Image Node。

## Retrieval context

图片复用现有 retrieval context 模块和 `retrieval_context.json`，不建立另一套上下文文件。

新增 image 分支：

- 使用现有 `_section_path_before_index()` 获取图片前方的章节标题路径；
- 使用图片的 deterministic searchable text 作为当前内容；
- 加入 MinerU `image_caption` 和 `image_footnote`；
- 选取图片前后各最多两个有效 text 对象，优先同页，并限制总字符数；
- 附近正文只用于帮助判断图片在合同中的位置、主题或用途；
- Prompt 明确禁止把附近正文中的主体、号码、日期或义务写成图片本身识别出的事实；
- 生成失败时沿用当前章节路径 fallback。

只为能够生成合法 searchable text 的图片生成 retrieval context。现有 text 和 table 的筛选、Prompt、并发、重试和 fallback 行为不改变。

## Image TextNode

继续使用 LlamaIndex `TextNode`：

```text
node.text = image searchable_text
```

metadata 至少包含：

```text
node_type=image
image_type
source_object_index
page_idx
bbox
img_path
structured_data
ocr_text
ocr_status
verification_status
verification_details
retrieval_context
image_processing_status
image_caption
image_footnote
content
sub_type
```

所有原始字段和结构化字段都排除出 embedding metadata；只有 `retrieval_context` 参与 metadata embedding。因此实际 embedding 内容维持现有顺序语义：

```text
retrieval_context

searchable_text
```

text Node 的原文、metadata 和 embedding 行为不变。table Node 的 `table_to_searchable_text()`、metadata 和 embedding 行为不变。Image Node 使用与其他对象相同的 `source_object_index` Node ID 规则，不创建独立 VectorStoreIndex。

## RAG and Evidence

`RAGPipeline` 的 retrieve → rerank → select → answer 算法不增加图片分支：

- Vector Retriever 根据 Image Node 的 embedding 文本召回；
- Reranker 继续读取 `MetadataMode.EMBED`；
- Evidence Selector 继续读取 `node.text`；
- Answer Generator 继续只读取选中的 `node.text`。

因此 `node.text` 就是实际提供给 Evidence Selector 和 Answer Generator 的图片证据文本。冲突和未验证状态必须写入该文本，不能只藏在 metadata 中。

聊天和 Evaluation 当前各自包含相似的 result serializer。新增一个共享的 evidence serializer，统一输出：

```text
node_type=text|table|image
source_object_index
node_id
text
page metadata
score
retrieval_score
retrieval_context
```

对旧索引中没有 `node_type` 的节点，serializer 默认输出 `node_type=text`，保证旧索引兼容。table 继续输出原有表格字段。image 额外输出：

```text
bbox
img_path
image_type
structured_data
ocr_text
ocr_status
verification_status
verification_details
image_processing_status
```

`img_path` 始终是合同目录内的原始相对引用。第一版不增加图片二进制 HTTP 路由；后续页面展示原图时，可以在服务端对该相对路径执行同样的安全解析。

## Evaluation and Web UI

Evaluation case 已经允许保存任意数字 `source_object_index`，Recall 和 rank 也按该字段计算，因此数据模型和评测算法不需要为图片单独修改。

现有 Evaluation 页面继续展示同一组：

```text
Vector Top10
Rerank Top10
Selected Evidence
Final Answer
```

三组 Node 卡片都增加 `node_type` 展示。image 卡片额外展示：

- `source_object_index`；
- page；
- `image_type`；
- `structured_data`；
- `verification_status`；
- `img_path`；
- `text`，即实际提供给 LLM 的证据。

聊天的 Selected Evidence 和 debug Rerank 卡片使用同一 serializer 和字段展示。

现有“查看解析结果”入口无需改变数据源。由于识别结果已经写回 `merged_content_list.json`，页面现有 JSON 展示会直接显示图片处理状态、结构化数据、OCR 和校验详情。不新增图片专用评测页面或图片解析结果页面。

## Error handling and degradation

### Per-image degradations

| Condition | Stored state | Node behavior |
|---|---|---|
| `img_path` 为空、越界或文件不存在 | `missing_image` + error | 跳过该 Image Node |
| VL 调用失败 | `vl_failed` + error | 跳过该 Image Node |
| VL 响应不是合法 JSON/Schema | `schema_invalid` + error | 跳过该 Image Node |
| image type 不在三类中或 data 与类型不匹配 | `unclassified` + error | 跳过该 Image Node |
| general 两个字段都为空 | `empty_result` | 跳过该 Image Node |
| MinerU OCR 调用失败 | `ocr_status=failed` | 保留 VL Node，`verification_status=insufficient` |
| OCR 无有效文字 | `ocr_status=empty` | 保留 VL Node，`verification_status=insufficient` |
| VL/OCR 明确冲突 | `verification_status=conflict` | 保留 Node，证据文本提示人工核验 |

单图异常必须记录包含 `contract_id`、`source_object_index`、`img_path` 和阶段的日志，但日志不得输出 API key 或其他凭据。

### Contract-level failures

以下仍属于合同级失败：

- MinerU ZIP 不合法或找不到唯一 content list；
- merged JSON 无法读取或不是数组；
- enriched merged JSON 无法原子持久化；
- VectorStoreIndex 构建或持久化失败。

这些情况无法保证整个索引的一致性，因此沿用当前合同 `failed` 状态处理。

## Expected file changes

### New files

- `app/image_schemas.py`：三类结构化 Schema 和严格结果类型；
- `app/image_understanding.py`：独立 VL Service；
- `app/image_ocr.py`：MinerU 单图 OCR Service adapter；
- `app/image_ingestion.py`：逐图编排、状态记录和 enriched object 构造；
- `image_verification.py`：纯代码归一化、候选提取和一致性校验；
- `image_searchable_text.py`：三类确定性检索文本；
- `app/evidence_serialization.py`：聊天和 Evaluation 共用的 Node/Evidence serializer。

### Modified files

- `mineru_raw_parse.py`：安全提取图片，复用 task transport 支持单图 OCR；
- `app/config.py`：新增 Vision 模型及超时配置；
- `app/pipeline.py`：在 merge 与 context 之间执行图片 enrichment，并原子写回 merged JSON；
- `retrieval_context_preprocess.py`：新增 image context 分支，保持 text/table 行为不变；
- `mineru_to_nodes.py`：新增 Image TextNode 分支；
- `app/qa.py`：使用共享 Evidence serializer；
- `app/evaluation_metrics.py`：使用共享 serializer，保留所有现有指标；
- `app/templates/chat.html`：展示 image Evidence 字段；
- `app/templates/evaluation.html`：在现有三阶段结果卡片中展示 image 字段。

`app/rag_pipeline.py`、Evaluation 数据库模型、Recall 算法和 IndexManager 不需要图片专用改造。

## Testing strategy

### MinerU and file tests

- ZIP 同时保存 content list 和被引用图片；
- 支持 content list 位于 ZIP 子目录；
- 拒绝绝对路径、`..`、写出合同目录和多重模糊匹配；
- 缺少单张引用图片时保存 JSON 并记录降级；
- 单图 OCR 使用 `parse_method=ocr` 和精简返回参数；
- OCR content list 按顺序拼接有效文本。

### Vision and schema tests

- 每张图片只调用一次 VL；
- 请求包含正确 MIME、base64 图片和结构化输出设置；
- 三类合法响应可解析；
- null/空字段可接受；
- 额外字段、缺字段、错误类型和分支不匹配被拒绝；
- VL 异常和 Schema 异常转换为单图状态，不向外抛出终止整个合同。

### Verification tests

- 账号完全一致为 verified；
- 空格和常见分隔符不影响一致性；
- 一位数字不同为 conflict；
- OCR 没有可信账号候选为 insufficient；
- 身份证号大小写 `X` 和格式归一化；
- 姓名、银行名只有明确标签对应时才判 conflict；
- 有效期无法可靠对应时不猜测；
- general 不调用 OCR，状态为 not_required。

### Enrichment and persistence tests

- image 原始字段全部保留；
- enriched list 长度和对象顺序不变；
- source_object_index 使用 merged 下标；
- 成功、缺图、VL 失败、OCR 失败、空 OCR 和 conflict 都写入 merged JSON；
- 多张图片中一张失败不影响后续图片；
- enriched merged JSON 使用原子替换；
- `reuse_existing` 从 raw 重新生成对象，不沿用旧派生结果。

### Searchable text and context tests

- bank、identity、general 输出稳定字段顺序；
- 空字段整行省略；
- searchable text 不包含 `img_path` 或 JSON 原文；
- conflict 明确写入 LLM 证据文本；
- image context 包含章节路径和受限的附近正文；
- nearby text 不会改变现有 text/table context 结果；
- context 失败继续使用现有章节 fallback。

### Node and RAG tests

- Image Node 使用 `TextNode`；
- metadata 保留原图、结构化数据、OCR 和验证状态；
- embedding 只包含 retrieval context 和 searchable text；
- 原始 metadata 不进入 embedding；
- text Node 原文、metadata 和 embedding 回归不变；
- table Node searchable text、metadata 和 embedding 回归不变；
- Image Node 可完整经过 Vector、Rerank、Selector 和 Answer 测试替身。

### Evidence, Evaluation, and page tests

- serializer 对 text/table/image 输出明确 `node_type`；
- image 在 Vector、Rerank、Selected Evidence 三组结果中保留全部要求字段；
- 聊天正常 Evidence 和 debug 使用相同字段；
- Evaluation 仍能用 image 的数字 `source_object_index` 标注 expected evidence；
- Vector/Rerank Recall 和排名算法不变；
- Evaluation 页面显示 image type、结构化数据、验证状态、路径和证据文本；
- 现有解析结果页面能直接看到 enriched image JSON；
- 最终 Answer 保持现有展示和失败文案。

### Real integration smoke test

使用项目实际 VL endpoint 和本机 MinerU 分别验证：

- 银行账户图片；
- 身份证正面；
- 身份证反面；
- 普通图片；
- 印章图片；
- 人为构造的账号单字符冲突图片。

自动化测试不依赖外部模型，全部使用测试替身。真实冒烟测试只在有明确测试图片和服务可用时执行，不把真实身份证号、银行卡号或 API key 写入仓库。

## Non-goals

- CLIP 或任何多模态 embedding；
- 图片向量数据库或图片相似度搜索；
- 复杂图表、工程图或 P&ID 理解；
- 手写内容专项识别；
- 长图片拆分和多图联合推理；
- 通用 OCR 服务重构；
- PaddleOCR 或其他 OCR 依赖；
- 图片 Agent；
- 图片二进制 Web 展示路由；
- 对历史缺图合同从 PDF 重新裁剪图片作为隐式回退。
