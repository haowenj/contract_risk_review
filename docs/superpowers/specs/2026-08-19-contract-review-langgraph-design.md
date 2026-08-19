# Contract Review LangGraph Backend Design

## Goal

在现有合同 RAG Web 项目中增加第一版合同风险审查后端。输入一个已经完成解析和索引、状态为 `ready` 的 `contract_id`，以及一段自然语言 `review_rule_text`，系统使用 LangGraph 将规则拆成审查项，逐项复用现有 Vector → Rerank → Evidence Selector 链路获取真实合同证据，结合风险规范与证据生成结构化风险判断，最后用纯 Python 汇总统计。

第一版只提供可测试、可从命令行执行的同步后端流程。不增加 Web 页面、数据库审查运行记录、后台任务、并行审查、持久化 checkpointer、人工介入、复杂 Agent、ReAct、工具调用 Agent、二次自主检索或正式报告生成。现有 text、table、image RAG，以及聊天、Evaluation 和入库流程的外部行为保持不变。

## Existing System

当前 `RAGPipeline.run()` 顺序执行：

```text
Vector Retrieval
  → Rerank
  → Evidence Selector
  → Answer Generator
```

其中检索、重排、证据选择和最终回答已经分别由 `retrieval_evaluation.retrieve_and_rerank()`、`select_evidence()`、`filter_nodes_by_indices()` 和 `generate_answer()` 实现。`app.evidence_serialization.serialize_node_result()` 已能把 text、table 和 image 节点序列化，并保留 `source_object_index`、`page_idx`、`node_type` 及各类型的原始元数据。

`ContractService.ask()` 会验证合同存在且为 `ready`，通过 `IndexManager` 加载索引，再调用包含 Answer Generator 的完整 RAG 流程。风险审查不能复用 `ask()` 的答案作为判断依据，因为这会丢失直接使用真实 Evidence 的边界。

当前运行环境能够导入 LangGraph 1.2.11，但项目尚未把 `langgraph` 声明为直接依赖。

## Architecture Decision

采用最小扩展现有 RAG 的方案：

1. 给 `RAGPipeline` 增加薄的 `retrieve_evidence()` 方法，只运行 Vector → Rerank → Evidence Selector，返回原始阶段结果。
2. 让现有 `RAGPipeline.run()` 复用 `retrieve_evidence()`，然后按原逻辑调用 Answer Generator，以避免产生两套检索实现。
3. 给 `ContractService` 增加 `search_contract()`，复用现有合同状态校验、索引加载和 Evidence 序列化能力。
4. 新建 `app/contract_review/`，将风险审查的状态、Schema、Prompt、节点、图和服务与现有 Web/RAG 代码隔离。
5. 使用同步 LangGraph `StateGraph` 串行循环所有审查项；不配置 checkpointer。

不采用审查节点直接调用 `retrieval_evaluation` 的方案，因为这会让新子系统耦合 RAG 底层细节。不重构为通用多阶段 RAG 框架，因为这会扩大聊天和 Evaluation 的回归范围。

## File Responsibilities

### `app/contract_review/state.py`

定义 `ContractReviewState` TypedDict，至少包含：

```text
contract_id
review_rule_text
review_items
current_item_index
review_results
summary
```

`review_results` 使用 LangGraph reducer 累积。`review_item` 每次只返回一个新结果，不能传回整个历史列表，避免覆盖或重复。

### `app/contract_review/schemas.py`

定义严格 Pydantic 模型：

- `ReviewItem`
- `ReviewItemList`
- `Evidence`
- `RiskDecision`
- `ReviewResult`
- `ReviewSummary`

所有 LLM 边界模型禁止额外字段。`ReviewItem` 的字符串字段去除首尾空白且不能为空，规则列表中的 `id` 必须唯一。

`RiskDecision` 只包含模型有权判断的字段：`risk_status`、`risk_level`、`evidence_status`、`finding`、`risk_description` 和 `suggestion`。模型不输出 Evidence 引用。程序把当前 `ReviewItem` 的 ID、名称以及 RAG 返回的 Evidence 附加后，再构造并校验完整 `ReviewResult`。

跨字段规则为：

- `evidence_status=insufficient` 时，必须是 `risk_status=needs_review` 且 `risk_level=null`；
- `risk_status=risk` 时，`risk_level` 必须是 `high`、`medium` 或 `low`；
- `risk_status=no_obvious_risk` 或 `needs_review` 时，`risk_level=null`。

如果 RAG 返回空 Evidence，程序不接受模型给出的乐观判断，而是强制构造 `needs_review / insufficient / null` 结果。风险描述必须表达证据不足，不能表达“合同没有约定”这一未经证据支持的结论。

### `app/contract_review/prompts.py`

提供两个纯 Prompt 构造函数。

规则解析 Prompt 明确要求：

- 只拆分、整理输入规范中明确存在的审查标准；
- 不增加法律常识、行业惯例、模型知识或输入中不存在的阈值；
- `rule_basis` 必须可追溯到原始规范；
- `retrieval_query` 只用于在合同中找事实，不把风险结论写进查询；
- 输出严格 JSON Schema 对象，顶层字段为 `review_items`。

风险判断 Prompt 只提交当前项的 `rule_basis`、`review_goal` 和序列化后的真实合同 Evidence，并要求：

- 只根据输入的规范和证据判断；
- 不增加规范标准；
- 不生成、猜测或改写证据引用；
- 证据不足时使用 `needs_review / insufficient`；
- 不把“没检索到”解释成“合同没有约定”。

### `app/contract_review/nodes.py`

实现三个节点：

- `parse_review_rules`
- `review_item`
- `aggregate_results`

节点通过构造函数注入规则解析 LLM、风险判断 LLM、合同检索接口和可选进度回调，方便测试并避免模块级可变状态。

`parse_review_rules` 调用绑定严格 `response_format=json_schema` 的 LLM，解析并本地校验 `ReviewItemList`，初始化 `current_item_index=0`。空规则文本、空审查项列表、重复 ID、非法 JSON 或 Schema 不匹配均抛出明确异常。

`review_item` 只读取 `review_items[current_item_index]`。它调用 `search_contract(contract_id, retrieval_query)` 获取 Evidence，将规则、目标和 Evidence 交给风险判断 LLM，校验 `RiskDecision`，由程序组装最终 `ReviewResult`，返回单元素 `review_results` 增量并将索引加一。

`aggregate_results` 不调用 LLM，只统计：

- `total_items`
- `risk_count`
- `high_risk_count`
- `medium_risk_count`
- `low_risk_count`
- `no_obvious_risk_count`
- `needs_review_count`

### `app/contract_review/graph.py`

构建以下同步图：

```text
START
  ↓
parse_review_rules
  ↓
review_item
  ↓
current_item_index < len(review_items) ?
  ├─ yes → review_item
  └─ no  → aggregate_results → END
```

规则解析必须至少生成一个审查项，因此无需从解析节点直接跳过循环。条件路由函数只读取 State，不执行副作用。

### `app/contract_review/service.py`

`ContractReviewService` 是外部入口，提供：

```python
service.run(contract_id, review_rule_text)
```

服务在进入图前验证两个输入非空，并通过合同检索服务提前确认合同存在且为 `ready`。随后以干净的初始 State 调用编译后的 LangGraph，并返回包含审查项、逐项结果和汇总的普通字典。

服务支持注入依赖用于测试，也提供默认构建函数，复用项目当前环境变量、数据库、embedding model、`IndexManager`、`ContractService` 和 LLM 配置。规则解析与风险判断使用 `temperature=0`、`max_retries=0`、`enable_thinking=false` 和严格 JSON Schema。

可选进度回调接收结构化事件。服务本身不强制写终端，避免以后被 API 或其他后端代码复用。

### `scripts/test_contract_review.py`

脚本顶部保存：

- 一个本地现有 `ready` 合同 ID；
- 一段可手动修改的 `review_rule_text`。

脚本构建默认 `ContractReviewService`，注册终端进度打印回调并调用 `run()`。输出顺序清晰展示：

1. 解析出的全部 ReviewItem；
2. 当前正在审查的项目；
3. RAG 命中的 Evidence；
4. 当前项目的风险结果；
5. 最终汇总。

## Detailed Data Flow

### Rule Parsing

输入：

```text
review_rule_text
```

输出：

```json
{
  "review_items": [
    {
      "id": "item_1",
      "name": "付款期限",
      "rule_basis": "付款期限不得超过90日",
      "review_goal": "判断合同付款期限是否超过90日",
      "retrieval_query": "合同约定的付款期限是多久"
    }
  ]
}
```

严格 Schema 只能约束输出形状，不能从技术上证明模型没有补充规范。因此 Prompt 会明确禁止补充标准；最终 `rule_basis` 和审查项会通过命令行输出供调用方核对。第一版不增加第二个 LLM 或文本蕴含模型做规则溯源判定。

### Evidence Retrieval

对每个 ReviewItem 执行：

```text
retrieval_query
  → index.as_retriever(similarity_top_k=10)
  → existing reranker
  → existing Evidence Selector
  → selected_nodes
  → existing evidence serializer
```

不调用 Answer Generator。序列化结果直接成为本项 `ReviewResult.evidence` 的来源。模型看到 Evidence 的文本和上下文，但无权输出引用元数据。

### Risk Decision

风险模型输入：

```text
rule_basis
review_goal
serialized Evidence
```

风险模型输出严格 `RiskDecision` JSON。程序进行本地 Pydantic 校验，然后：

1. 从当前 ReviewItem 复制 `item_id` 和 `item_name`；
2. 从已校验 RiskDecision 复制判断字段；
3. 从 RAG 结果直接附加 Evidence；
4. 构造并再次校验完整 ReviewResult。

这条边界确保 `source_object_index`、`page_idx` 和 `node_type` 直接来自 RAG 结果，LLM 无法生成或修改这些引用。

### Aggregation

汇总函数只遍历最终 `review_results`。`risk_count` 等于 `risk_status=risk` 的项数；高中低计数只统计风险项；无明显风险和需人工复核按对应状态统计。总数必须等于结果列表长度。

## Failure Semantics

以下情况立即终止整次审查并抛出明确异常：

- `contract_id` 为空、不存在或合同不是 `ready`；
- `review_rule_text` 为空；
- 规则解析 LLM 调用失败、返回非法 JSON 或不符合 Schema；
- 风险判断 LLM 调用失败、返回非法 JSON、不符合 Schema 或违反跨字段规则；
- 索引文件不存在；
- Vector、Rerank 或 Evidence Selector 链路抛出异常。

第一版不自动重试、不调用 LLM 修复 JSON、不进行二次检索，也不把基础设施或模型失败转换成 `needs_review`。

只有检索链路正常完成但 Evidence 为空，或风险模型在已有 Evidence 下明确判断证据仍不足，才返回 `needs_review / insufficient`。

## Compatibility

现有 `RAGPipeline.run()` 的返回字段保持不变：`query`、`vector_results`、`reranked_results`、`selected_indices`、`selected_nodes` 和 `llm_summary`。聊天仍调用 `answer_question()`，Evaluation 仍调用 `RAGPipeline.run()`，它们继续生成 Answer。

`search_contract()` 是新增接口，不修改 `ask()`。Evidence 继续使用统一序列化器，因此 text、table 和当前分支已有的 image Evidence 字段保持原状。

项目增加 `langgraph` 直接依赖并更新锁文件，不引入其他新框架。

## Testing

测试遵循 TDD，先观察新增行为测试失败，再写最小实现。

### RAG boundary tests

- `retrieve_evidence()` 执行 Vector、Rerank、Evidence Selector 并返回所选节点；
- `retrieve_evidence()` 不调用 Answer LLM；
- `run()` 仍返回与当前相同的完整结果并生成答案；
- `ContractService.search_contract()` 验证合同状态并序列化 text、table、image Evidence。

### Schema and node tests

- 规则解析拒绝空列表、重复 ID、空字段和额外字段；
- 风险输出拒绝非法枚举、缺失字段、额外字段和矛盾状态组合；
- 单个 `review_item` 使用当前项的 `retrieval_query`；
- RAG Evidence 引用原样进入 ReviewResult；
- 空 Evidence 强制产生 `needs_review / insufficient / null`；
- LLM 或检索异常向上抛出并终止流程。

### Graph and aggregation tests

- 使用假的 LLM 和假的 RAG 接口运行真实编译图；
- 多项规则按顺序审查且 `review_results` 只追加不覆盖；
- 汇总正确计算总数、风险数、高中低风险数、无明显风险数和需人工复核数。

### Verification

完成后运行：

1. 新增合同审查测试；
2. 现有 RAG、聊天和 Evaluation 相关测试；
3. 完整测试集；
4. 使用本地现有 `ready` 合同执行 `scripts/test_contract_review.py` 冒烟测试。

真实命令行冒烟测试需要当前环境的 LLM、embedding 和 rerank 服务可用。如果外部服务不可用，必须如实报告具体失败阶段，不能用单元测试结果替代真实执行结果。
