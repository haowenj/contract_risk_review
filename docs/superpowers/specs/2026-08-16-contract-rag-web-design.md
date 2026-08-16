# 合同 RAG 最小 FastAPI Web 应用设计

## 目标

将当前零散的合同 RAG 实验脚本整理为一个最小可用的 FastAPI 应用，支持合同上传、后台入库、状态查询、已持久化索引复用、单问题问答和 RAG 调试信息展示。

第一版只接受 PDF。现有 MinerU、清洗、跨页合并、retrieval_context、Node、Embedding、Retriever、qwen3-rerank、Evidence Selector 和 Answer Generator 的算法行为保持不变；应用层只负责编排、持久化、状态管理和 Web 适配。

## 约束与非目标

- 不通过 subprocess 调用 Python 脚本；后台任务直接调用现有 Python 函数。
- 不引入 Qdrant、Redis、Celery、LangGraph、Vue、React 或其他额外基础设施。
- 使用 FastAPI `BackgroundTasks` 完成第一版入库任务。
- 使用 SQLite 保存合同基本信息和处理状态。
- 每份合同使用独立 `contract_id` 和独立目录。
- 使用 LlamaIndex `storage_context.persist()` 保存索引，使用 `load_index_from_storage()` 恢复；问答阶段禁止重新向量化。
- 每个问题独立检索，不实现多轮上下文理解。
- 页面由 FastAPI 后端渲染 HTML；不做前后端分离，不建立独立前端工程。

## 现有代码复用边界

现有模块的职责保持如下：

- `mineru_raw_parse.run_parse()`：调用 MinerU 服务并保存原始 content list。
- `clean_mineru_data.clean_content_list_file()`：执行确定性的清洗。
- `merge_cross_page_paragraphs.merge_content_list_file()`：执行跨页段落合并并可保存日志。
- `retrieval_context_preprocess.generate_contexts()` 与 `save_retrieval_contexts()`：生成并保存 retrieval context。
- `mineru_to_nodes.build_nodes()`：把已处理的 JSON 和 context 构造成带元数据的 `TextNode`。
- `retrieval_evaluation`：保留现有检索、重排、证据选择和答案生成逻辑；新增的应用入口只组合这些函数并序列化结果。

需要的最小兼容性整理是把当前脚本中的环境配置/模型构造收敛为可调用的工厂或应用初始化依赖，同时保留现有测试和脚本入口可用。不得改变现有检索参数、提示词、重排映射、Evidence Selector 回退策略或 Answer Generator 的失败答案。

## 目录与数据模型

默认应用数据根目录为项目根目录下的 `data/`，可用环境变量覆盖。每份合同的目录布局为：

```text
data/
├── contracts.db
└── contracts/
    └── <contract_id>/
        ├── source.pdf
        ├── raw_content_list.json
        ├── cleaned_content_list.json
        ├── merged_content_list.json
        ├── merge_log.json
        ├── retrieval_context.json
        └── index/
            ├── docstore.json
            ├── index_store.json
            └── vector_store.json
```

SQLite `contracts` 表至少包含：

- `contract_id`：随机 UUID 字符串，主键。
- `filename`：用户上传时的原文件名，仅用于展示。
- `storage_dir`：该合同目录的绝对路径或应用根目录下的稳定路径。
- `status`：`queued`、`processing`、`ready`、`failed` 之一。
- `error_message`：失败原因，成功时为空。
- `created_at`、`updated_at`：ISO 8601 字符串。

数据库模块使用 Python 标准库 `sqlite3`，在初始化时创建表，并提供创建、状态更新、单条查询和列表查询函数。状态更新必须在后台任务成功持久化索引后才写为 `ready`；任何阶段异常都写为 `failed` 并保存可展示的错误消息。

## 入库流程

`POST /api/contracts` 接收 `UploadFile`：

1. 校验扩展名为 `.pdf`，否则返回 HTTP 400。
2. 创建 UUID `contract_id` 和独立目录，保存 `source.pdf`。
3. 写入 SQLite，状态为 `queued`。
4. 通过 `BackgroundTasks.add_task()` 调用同步的入库服务，并立即返回合同摘要。

后台入库服务按以下顺序直接调用函数：

```text
source.pdf
  → run_parse(..., raw_content_list.json)
  → clean_content_list_file(..., cleaned_content_list.json)
  → merge_content_list_file(..., merged_content_list.json, merge_log.json)
  → load merged JSON
  → generate_contexts(...)
  → save_retrieval_contexts(..., retrieval_context.json)
  → build_nodes(...)
  → VectorStoreIndex(nodes, embed_model=embedding_model)
  → index.storage_context.persist(persist_dir=index_dir)
  → SQLite status=ready
```

后台任务开始时更新为 `processing`。异常由服务捕获，记录日志并更新为 `failed`，不会把不完整索引标记为 `ready`。

## IndexManager

`IndexManager` 是进程内最小缓存，内部维护 `dict[str, VectorStoreIndex]`，并以线程锁保护加载和写入。

接口：

```python
class IndexManager:
    def get(self, contract: ContractRecord) -> VectorStoreIndex:
        """返回缓存索引；未命中时从 contract.storage_dir/index 恢复。"""

    def put(self, contract_id: str, index: VectorStoreIndex) -> None:
        """写入当前进程缓存。"""

    def clear(self, contract_id: str | None = None) -> None:
        """清理单份或全部缓存，测试和未来维护使用。"""
```

加载路径必须使用：

```python
storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
load_index_from_storage(storage_context, embed_model=embedding_model)
```

索引缓存只缓存已加载对象，不把缓存作为唯一数据源；服务重启后仍可以从磁盘恢复。只有 `ready` 合同允许问答，缺少索引文件时返回服务错误而不重新向量化。

## 问答流程

`POST /api/contracts/{contract_id}/chat` 接收：

```json
{
  "question": "合同的付款方式是什么？",
  "debug": false
}
```

服务校验合同存在且状态为 `ready`，通过 `IndexManager.get()` 获得索引，然后调用一个新增的单问题应用入口。该入口内部等价于当前评估流程：

1. `index.as_retriever(similarity_top_k=TOP_K)`。
2. 对向量检索 Top10 调用当前 DashScope `qwen3-rerank`。
3. 调用当前 Evidence Selector，沿用选择失败时回退到 Rerank Top3 的逻辑。
4. 只将 Selected Evidence 传给当前 Answer Generator。
5. 返回答案、Selected Evidence 和可选调试结构。

不传入历史消息；每次请求仅使用当前问题。

正常响应结构：

```json
{
  "contract_id": "...",
  "question": "...",
  "answer": "...",
  "evidence": [
    {
      "source_object_index": 111,
      "node_id": "...",
      "text": "...",
      "page_idx": 12,
      "start_page_idx": 12,
      "end_page_idx": 13,
      "score": 0.95,
      "retrieval_score": 0.72
    }
  ],
  "debug": null
}
```

当 `debug=true` 时，`debug` 包含：

- `rerank_top10`：按最终重排顺序的最多 10 条候选及分数、来源索引、原文和页码元数据。
- `selected_evidence`：最终选择的证据条目。
- `final_answer`：最终答案字符串。

证据序列化只读取现有 Node/Result 的文本、source object index、页码和分数，不改变算法对象本身。

## HTTP API 与服务端渲染页面

API：

- `POST /api/contracts`：上传 PDF，返回 HTTP 202 和合同摘要。
- `GET /api/contracts`：返回合同列表及状态。
- `GET /api/contracts/{contract_id}`：返回合同详情及状态/错误。
- `POST /api/contracts/{contract_id}/chat`：返回单问题 JSON 答案和证据。

页面：

- `GET /`：服务端读取合同列表并渲染完整 HTML。
- 页面左侧显示上传表单、合同列表、状态和失败消息。
- 页面右侧显示当前合同的提问表单、答案和证据。
- 上传表单提交到后端上传路由后重新渲染页面。
- 提问表单提交到后端页面路由；服务端加载索引、执行问答并重新渲染答案。
- 调试开关是页面中的普通复选框；选中后在渲染结果中显示 Rerank Top10、Selected Evidence 和 Final Answer。
- API 与页面共用同一个 service 层，页面路由不复制 RAG 逻辑。

模板可以使用 FastAPI/Jinja2 的标准模板支持，保持依赖和结构最小；不创建独立 JS 构建流程。

## 配置与依赖

继续使用 `.env` 中现有 LLM 和 MinerU 配置，并增加：

- `APP_DATA_DIR`：默认 `data`。
- `APP_DATABASE_PATH`：默认 `<APP_DATA_DIR>/contracts.db`。
- `APP_CONTRACTS_DIR`：默认 `<APP_DATA_DIR>/contracts`。

新增运行依赖：`fastapi`、`uvicorn`、`python-multipart`、`jinja2`。不新增数据库 ORM 或任务队列。

## 错误处理

- 非 PDF 上传：HTTP 400。
- 不存在的合同：HTTP 404。
- 合同尚未 `ready`：问答返回 HTTP 409，并附当前状态。
- 后台入库失败：合同状态为 `failed`，页面和状态 API 展示 `error_message`。
- IndexManager 恢复失败：问答返回 HTTP 500，并记录日志；不尝试重新向量化。
- Rerank、Evidence Selector、Answer Generator 的现有降级策略保持不变。

## 测试策略

- 新增配置/SQLite 测试：建表、创建合同、状态更新、列表排序和错误信息。
- 新增 IndexManager 测试：缓存命中不加载磁盘、缓存未命中调用 `load_index_from_storage()`、`clear()` 行为。
- 新增入库编排测试：用可控的函数依赖验证每个处理阶段按顺序调用，成功才写 `ready`，异常写 `failed`。
- 新增问答序列化测试：验证现有 Retriever → Rerank → Selector → Answer 结果被映射为答案、证据和 debug 结构。
- 新增 FastAPI API 测试：上传校验、列表/详情、ready 状态问答和非 ready 拒绝；后台任务在测试中使用可替换服务执行。
- 新增服务端渲染测试：根路径包含上传表单、合同列表、提问表单和调试输出。
- 现有测试全部保持通过；无法访问真实 MinerU/LLM 时使用测试替身，不改变现有算法测试。
