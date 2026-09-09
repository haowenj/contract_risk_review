# 正式合同审查 BM25 混合检索设计

## 目标

将已验证的持久化 BM25 索引接入合同审查主流程，与现有向量索引分别召回、按 `node_id` 去重后统一进入现有 rerank 和 Evidence 流程；移除正式审查中的“查询改写 + 关键词全文匹配”检索兜底，不改变解析、Evidence 数据格式和后续风险审查接口。

## 当前链路与改造边界

当前正式审查路径为：

```text
ContractReviewNodes.retrieve_evidence
  -> ContractService.search_contract
  -> RAGPipeline.retrieve_evidence
  -> vector Top K -> rerank -> Evidence Selector -> selected Evidence
  -> review LLM / finalize
```

证据不足时，当前图会执行第二次 LLM 查询改写，改写同时生成全文扫描关键词，再扫描 `merged_content_list.json` 并进入 `absence_result`。这套分支是本次要退出正式业务流程的检索兜底。

本次不修改：

- MinerU 解析、清洗、跨页合并、`retrieval_context` 生成和 Node 构建；
- 解析阶段已存在的向量索引和 BM25 索引生成、持久化位置及中文 `jieba.lcut_for_search` 分词方式；
- 当前 rerank、Evidence Selector、Evidence 序列化格式、风险判断和审查结果页面；
- 历史已保存结果中 `absence_check` 字段的读取和展示兼容性。

## 正式检索设计

### 解析阶段

继续使用现有 `ContractProcessor.process` 流程。在 `build_nodes` 返回同一批 Node 后按顺序执行：

1. 构建并持久化向量索引；
2. 只有向量索引成功后，使用同一批 Node 构建并持久化 BM25 索引；
3. 使用同一个 `index_version` 将两套索引放入 `IndexManager` 缓存。

查询阶段只从 `IndexManager` 加载已经持久化的向量索引和 BM25 索引，不在查询中实例化或重建 BM25。BM25 索引缺失时明确失败并提示重新解析，禁止静默退回纯向量。

### 召回与 Evidence

正式审查的单次查询执行以下步骤：

```text
Query
  ├─ vector index Top K
  └─ persisted BM25 index Top K
       ↓
按 node_id 去重，保留候选 Node 与各自原始分数
       ↓
现有 rerank
       ↓
现有 Evidence Selector / Evidence 序列化
       ↓
后续合同风险审查
```

向量分数与 BM25 分数不做加权、归一化或直接排序比较。候选合并只负责集合合并；重复 Node 只保留一个候选，优先保留向量召回中的对象，BM25 独有对象保留 BM25 原始分数。rerank 仍负责最终相关性排序。

当混合候选或最终 Evidence 为空时，直接使用现有 `insufficient_result` 生成 `needs_review / insufficient` 结果，不再发起第二次检索，不再调用查询改写 LLM，不再扫描全文关键词，也不再生成新的 `absence_verified` 结果。

### 日志与进度

每次正式混合检索记录结构化日志，至少包含：

- `vector_count`；
- `bm25_count`；
- `merged_candidate_count`；
- `reranked_count`；
- 最终 Evidence 的 `node_id`、数量；
- 向量召回、BM25 召回、去重、rerank、Evidence 选择和总耗时。

现有审查进度事件继续保留；`evidence_retrieved` 可以增加上述诊断字段，但不改变前端结果数据格式。

## 失败行为

- 向量索引加载失败：审查任务失败，错误保留向量索引阶段上下文；
- BM25 索引加载失败：审查任务失败，错误明确指出 BM25 索引不可用并提示重新解析；
- 向量召回、BM25 召回、去重或 rerank 失败：审查任务失败并保留异常原因；
- 混合召回没有有效 Evidence：进入现有证据不足结果，不视为合同缺失，也不执行关键词兜底。

## 测试与验收

新增或更新测试覆盖：

- 两个索引在查询时都被加载，BM25 不被临时构建；
- 向量和 BM25 候选按 `node_id` 去重，原始分数不互相计算，合并候选统一进入 rerank；
- 正式合同审查只执行一次混合检索；空 Evidence 直接得到 `insufficient`，查询改写和全文扫描不被调用；
- 默认审查服务不再构建查询改写 LLM；
- 现有解析、索引持久化、Evidence 序列化、审查页面及历史 `absence_check` 展示测试继续通过；
- 使用仓库中已有真实合同的持久化向量/BM25 索引执行一次混合召回 smoke test，确认中文关键词可以召回对应 Node。

