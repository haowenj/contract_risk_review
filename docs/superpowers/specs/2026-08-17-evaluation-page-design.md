# Contract Evaluation Page Design

## 目标

在现有合同入库和聊天页面之上增加合同级 Evaluation 页面。评测使用已经持久化的向量索引，不重新解析 PDF、不重新构建向量；聊天和评测必须复用同一个不带评测语义的单题 RAG Pipeline。评测集允许按合同保存自定义问题和正确的数字节点 ID，支持单题运行和全量运行，并保存每次运行的完整检索结果，为后续自动汇总指标和版本对比提供数据基础。

本次使用的数字节点 ID 是现有脚本中的 `source_object_index`，不是 LlamaIndex 的 UUID `node_id`。

## 范围与非目标

本次包含：

- 将聊天使用的 Vector Retrieval、Rerank、Evidence Selector、Answer Generator 组合成公共应用层 `RAGPipeline`。
- 聊天入口和 Evaluation 入口分别调用同一个 `RAGPipeline`，但 Evaluation 的 gold、Recall 和版本校验留在 Evaluation Service。
- 按合同持久化评测问题、正确 `source_object_index` 列表和排序。
- 在已入库合同旁边增加 Evaluation 入口。
- Evaluation 页面支持编辑、保存、新增、删除评测问题。
- 支持每道题单独测试以及所有已保存问题一起测试。
- 保存 Vector Top10、Rerank Top10、分数、排名、Recall@5/10、Selected Evidence 和最终回答。
- 页面展示实际命中的数字节点 ID 以及 Vector/Rerank 的 Recall@5 和 Recall@10。

本次不包含：

- 自动生成或修改正确答案/正确节点 ID。
- 不同评测集版本或不同 RAG pipeline 之间的对比页面。
- 自动汇总所有题目的平均 Recall、成功率等更高层指标。
- 新增向量数据库、任务队列或前端构建工程。

## 设计取舍

### 方案 A：保留脚本引擎，新增应用层公共单题 Pipeline（采用）

保留 `retrieval_evaluation.py` 中已经验证过的算法、Prompt、降级策略和命令行入口；新增 `app/rag_pipeline.py` 作为真正公共的单题 Pipeline。聊天和评测分别调用它，Recall 和 gold node 只存在于 Evaluation Service。优点是改动集中、脚本仍可运行、两种入口天然共用 retrieve → rerank → selector → answer 算法；新增的持久化和页面逻辑不会污染底层检索实现。

### 方案 B：把脚本全部移动到 `app/` 下

可以让应用结构更集中，但会造成较大的文件移动和导入变化，容易破坏已有评测脚本和测试，当前收益不足以抵消风险。

### 方案 C：每份合同使用 JSON 文件保存评测集和运行结果

实现较快，但查询最近运行、按问题关联结果和后续版本对比都需要遍历文件；运行结果和合同状态也不在同一持久化边界内。因此评测集和运行元数据使用现有 SQLite，完整结果以 JSON 字段保存。

## 公共单题 RAG Pipeline

新增 `app/rag_pipeline.py`，负责组合既有 `retrieval_evaluation` 函数，不重新实现算法，也不接收评测集、正确节点或 Recall 参数。

公共入口提供以下语义：

```python
class RAGPipeline:
    def run(
        self,
        index: Any,
        question: str,
        *,
        reranker: Any | None = None,
        selector_llm: Any | None = None,
        answer_llm: Any | None = None,
    ) -> dict[str, Any]:
        """执行单题 retrieve → rerank → selector → answer。"""
```

`run()` 内部固定按以下顺序调用现有逻辑：

```text
index.as_retriever(similarity_top_k=10)
  → vector retrieval
  → DashScope rerank
  → Evidence Selector
  → Answer Generator
```

必要时对 `retrieval_evaluation.py` 做小范围拆分：提供不带 gold 的单题 retrieve/rerank 原语，保留现有 `run_evaluation()` 作为脚本评测适配器。`RAGPipeline.run()` 直接组合这些原语、`select_evidence()` 和 `generate_answer()`，不调用 `run_evaluation()`，从结构上避免 Chat `answer()` 依赖 Evaluation 语义。测试替身可以通过可选的 reranker、selector LLM 和 answer LLM 注入。

当前的 `app.qa.answer_question()` 保留为兼容入口，但改为委托 `RAGPipeline.run()` 并只负责聊天结果序列化；它不构造空 gold，也不计算 Recall。`ContractService.ask()` 负责检查合同存在、状态为 `ready`、通过 `IndexManager.get()` 加载已持久化索引，然后调用公共 Pipeline。Evaluation Service 也加载同一个索引，逐题调用公共 Pipeline，再在 Pipeline 结果之上计算 Recall、排名和命中节点。全量评测在一次运行中只获取一次 IndexManager 结果，所有问题共享该索引对象。

## 评测集持久化

在现有 SQLite 中增加合同索引版本字段和两类评测表。表结构初始化继续由 `ContractRepository` 的数据库初始化流程完成，不增加 ORM。

### 合同索引版本

`contracts` 增加 `index_version` 字段，成功持久化一份索引后生成新的 UUID 版本并与 `ready` 状态一并写入。合同重新解析或重新建索引成功后必须生成新的 `index_version`，不能沿用旧值。旧评测集不会自动迁移到新版本。

`evaluation_cases` 和 `evaluation_runs` 都保存创建时的 `index_version`。只有 case 的 `index_version` 与合同当前 `index_version` 一致时才允许运行；不一致时页面标记“评测集对应旧索引，请重新保存/重新标注”，防止 `source_object_index` 变化后人工标注静默失效。

`IndexManager` 的缓存键也必须包含 `index_version`，或在版本切换时可靠清除旧缓存，不能因为仍使用同一个 `contract_id` 而返回旧索引。

### `evaluation_cases`

每一行是合同下的一道评测题：

- `case_id`：整数主键。
- `contract_id`：关联 `contracts.contract_id`，合同删除时级联删除。
- `index_version`：保存评测集时绑定的合同索引版本，非空。
- `question`：非空问题文本。
- `expected_source_object_indices`：JSON 数组，只允许整数。
- `sort_order`：页面展示和全量执行顺序。
- `created_at`、`updated_at`：UTC ISO 时间。

保存配置时以表单当前顺序整体更新该合同的评测集；删除页面行即删除对应 case。问题为空或节点 ID 不是整数时拒绝保存并在页面显示错误，不写入部分配置。

如果合同没有任何已保存评测题，页面首次展示脚本中现有的 `EVALUATION_QUERIES` 作为可编辑的默认表单内容；这些默认内容必须点击“保存评测集”后才成为持久化数据。

### `evaluation_runs` 与 `evaluation_run_items`

`evaluation_runs` 保存一次单题或全量运行的元数据：

- `run_id`：UUID 字符串主键。
- `contract_id`：被评测合同。
- `scope`：`single` 或 `all`。
- `status`：`queued`、`processing`、`ready`、`failed`。
- `index_version`：本次运行使用的索引版本。
- `pipeline_version`：当前固定写入 `rag-v1`，用于后续版本对比。
- `config_snapshot`：本次运行实际使用的配置 JSON，至少包含 `vector_top_k`、`rerank_top_k`、rerank 模型、answer 模型、selector 模型和 `pipeline_version`。
- `created_at`、`started_at`、`completed_at`。
- `error_message`：失败时保存可展示错误。

创建 run 时保存本次执行的 case 快照，避免运行期间用户修改评测集导致结果与输入不一致。

`evaluation_run_items` 每行对应一次 run 中的一道题，保存：

- `run_id`、`case_id`。
- `question_snapshot`。
- `expected_source_object_indices_snapshot`。
- `result_json`：完整序列化结果。

`result_json` 至少包含：

```json
{
  "query": "合同分几期付款，每期比例是多少？",
  "expected_source_object_indices": [111, 112, 113, 114],
  "vector_results": [],
  "reranked_results": [],
  "selected_nodes": [],
  "vector_scores": {},
  "vector_ranks": {},
  "rerank_ranks": {},
  "vector_recall_at_5": 0.5,
  "vector_recall_at_10": 1.0,
  "rerank_recall_at_5": 0.75,
  "rerank_recall_at_10": 1.0,
  "llm_summary": {
    "answer": "...",
    "evidence_indices": [111, 112]
  }
}
```

`config_snapshot` 示例：

```json
{
  "vector_top_k": 10,
  "rerank_top_k": 10,
  "rerank_model": "qwen3-rerank",
  "selector_model": "qwen3.7-plus",
  "answer_model": "qwen3.7-plus",
  "pipeline_version": "rag-v1"
}
```

候选节点序列化时保留 `source_object_index`、真实 `node_id`、原文、页码元数据、向量分数、重排分数和检索上下文。运行结果不只保存命中的 ID，保证后续可以复盘完整检索链路。

## 页面与路由

### 合同列表页

`/` 的 `ready` 合同行新增 `Evaluation` 按钮，链接到：

```text
/contracts/{contract_id}/evaluation
```

不为 `queued`、`processing`、`failed` 合同提供可用评测入口。

### Evaluation 页面上方

页面显示合同名称和索引状态，并渲染可编辑的评测集表格。每行包含：

- 问题文本框；
- 正确 Node ID 文本框，使用逗号、空格或换行分隔的数字；
- 保存后的 case ID；
- “测试本题”按钮；
- 删除按钮。

页面提供“新增问题”“保存评测集”“测试全部问题”按钮。新增加但尚未保存的行不能单独测试，页面提示先保存评测集。

### Evaluation 页面下方

默认显示最近一次运行结果；如果没有运行结果，显示引导信息。每道题的结果卡展示：

- 问题与正确 Node ID；
- Vector Top10 实际返回的数字 Node ID；
- Rerank Top10 实际返回的数字 Node ID；
- Vector Recall@5、Vector Recall@10；
- Rerank Recall@5、Rerank Recall@10；
- 候选节点明细：排名、数字 Node ID、真实 `node_id`、分数、页码、文本；
- Selected Evidence 和最终回答。

页面使用后端渲染，允许少量原生 JavaScript 添加/删除表单行以及轮询运行状态，不建立独立前端工程。

### 运行方式

单题和全量测试都创建 `evaluation_run`，通过 FastAPI `BackgroundTasks` 执行，避免全量评测的远程 Rerank/LLM 调用阻塞页面请求。POST 路由创建 run 后重定向回 Evaluation 页面，页面通过 JSON 状态接口轮询 `queued`/`processing`/`ready`/`failed`，运行完成后加载并展示保存的结果。

应用启动时必须先调用评测 Repository 的恢复逻辑：将所有遗留的 `queued` 或 `processing` run 标记为 `failed`，写入“服务重启导致评测任务中断”的错误信息和完成时间。这样上一次进程中断的任务不会永久停留在 processing；新的后台任务只在恢复完成后接受。

计划使用以下服务端页面路由和状态接口：

```text
GET  /contracts/{contract_id}/evaluation
POST /contracts/{contract_id}/evaluation/config
POST /contracts/{contract_id}/evaluation/cases/{case_id}/run
POST /contracts/{contract_id}/evaluation/run-all
GET  /api/contracts/{contract_id}/evaluation/runs/{run_id}
```

页面路由只负责表单解析、权限/状态检查、创建后台任务和模板渲染；RAG 计算和结果保存放在应用 Service，不复制到路由函数。

## 错误处理

- 不存在的合同返回 404。
- 合同未处于 `ready` 状态时不能进入或运行评测，并展示当前状态。
- 评测问题为空、正确 ID 不是整数或格式非法时，配置保存失败并保留用户输入。
- 评测 case 的 `index_version` 与合同当前版本不一致时禁止运行，页面明确提示需要重新保存/标注。
- 索引目录不存在时，运行标记为 `failed`，不重新向量化。
- Rerank、Evidence Selector、Answer Generator 继续使用现有降级策略；如果整个运行发生未处理异常，run 和对应 item 保存失败信息。
- 单题失败不影响此前已完成的其他 run；全量运行按已保存 case 快照执行，失败状态可在页面展示。

## 测试策略

新增测试覆盖：

- `RAGPipeline`：聊天和评测都调用同一个不带 gold 的 retrieve → rerank → selector → answer 单题 Pipeline；聊天不构造 Evaluation 语义，Evaluation 在 Pipeline 结果之外计算 Recall。
- 索引版本：重新建索引后生成新版本、旧 case 被识别为 stale、IndexManager 不返回旧缓存。
- 评测集 Repository：创建、更新、排序、删除、非法 ID 拒绝和合同关联。
- Evaluation run Repository：创建 run、保存 `config_snapshot`、状态、case 快照和完整 JSON 结果、读取最近结果；启动恢复会将 queued/processing 标记为 failed。
- Evaluation Service：ready 合同加载 IndexManager、单题/全量共用索引、结果完整保存、异常标记 failed。
- FastAPI 路由：ready 合同出现入口、评测页面渲染默认问题、配置保存、单题运行、全量运行、状态接口和非 ready 拒绝。
- 模板：显示实际命中的数字 Node ID、四类 Recall 值、候选节点和最终结果。
- 保持现有 retrieval evaluation、聊天、入库和 API 测试通过。

无法访问真实 MinerU、Rerank 或 LLM 时，使用现有测试替身注入，不改变实际算法和线上配置。
