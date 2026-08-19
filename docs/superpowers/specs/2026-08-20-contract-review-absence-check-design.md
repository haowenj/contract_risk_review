# 合同风险审查缺失条款核验设计

## 1. 背景与目标

当前合同风险审查流程对每个 `ReviewItem` 最多执行两次 RAG。第一次没有 Evidence 时，LLM 改写 `retrieval_query`，然后执行第二次 RAG；如果第二次仍没有 Evidence，当前实现返回 `needs_review / insufficient`。

本次增加 `absence_check`。它用于处理审查规范明确要求合同必须包含或必须限制某类事项，但两次语义检索均未找到有效 Evidence 的场景。`absence_check` 不执行第三次 RAG，而是读取合同已有的 `merged_content_list.json`，通过确定性 Python 关键词扫描核验完整解析内容。

本次改造只覆盖缺失条款核验，不增加 BM25、新索引、重新 embedding、重新解析、OCR/VL 调用、多 Agent、Web 页面、数据库持久化或人工中断流程。

## 2. 总体流程

合同审查图改为显式的单项审查节点：

```text
parse_review_rules
  ↓
prepare_review_item
  ↓
retrieve_evidence
  ├─ 有累计 Evidence → risk_decision
  ├─ 无 Evidence、attempt=1 → rewrite_query
  │                           ↓
  │                     retrieve_evidence
  └─ 无 Evidence、attempt=2 → absence_check
                                ├─ 有候选 → risk_decision
                                └─ 无候选 → absence_result
                                               ↓
                                      finalize_review_item
                                               ↓
                                  下一项 / aggregate_results
```

`risk_decision` 返回 `evidence_status=insufficient` 且尚未执行第二次 RAG 时，仍保留现有改写并重试一次的行为。任何路径最多执行两次 RAG。

## 3. LangGraph State

在现有 `ContractReviewState` 上增加以下单项审查字段：

```text
retrieval_attempt: int
current_retrieval_query: str
retrieved_evidence: list[Evidence]
absence_keywords: list[str]
absence_candidates: list[Evidence]
absence_candidate_count: int
current_decision: RiskDecision | None
```

字段含义：

- `retrieval_attempt`：当前 RAG 次数，取值为 1 或 2。
- `current_retrieval_query`：当前实际执行的检索问题。
- `retrieved_evidence`：当前 ReviewItem 累积的 RAG Evidence，按 `source_object_index` 去重。
- `absence_keywords`：第一次改写查询时同时生成、供必要时全文扫描的关键词。
- `absence_candidates`：确定性全文扫描得到的真实候选 Evidence。
- `absence_candidate_count`：应用候选返回上限之前命中的唯一 source object 总数。
- `current_decision`：当前 ReviewItem 尚未封装为 `ReviewResult` 的风险判断。

`review_results` 继续使用 reducer 累积。每个 ReviewItem 开始时重置新增的单项字段，不改变合同级字段和最终返回结构。

## 4. 查询改写与扫描关键词

扩展现有严格 Schema `RetrievalQueryRewrite`：

```json
{
  "retrieval_query": "第二次语义检索问题",
  "reason": "改写原因",
  "keywords": ["分包", "转包", "转委托"]
}
```

第二次查询和扫描关键词在第一次 RAG 为空或第一轮风险判断为 `insufficient` 后由同一次 LLM 调用生成，不增加关键词专用 LLM 调用。

关键词要求：

- 只能围绕当前 `rule_basis` 和 `review_goal` 扩展。
- 允许同义词、常见合同及法律表达、措辞变体。
- 优先生成能够识别当前审查主题的核心术语或具有业务区分度的短语。
- 不得单独输出“同意”“批准”“许可”“责任”“合同”等缺乏主题区分度的泛化词；这些词只有与当前主题组合成明确短语时才能使用，例如“分包书面同意”“第三方履约批准”。
- 不得增加新的风险审查标准。
- 使用严格 JSON Schema，禁止额外字段。
- Pydantic 对关键词去除首尾空白、删除空字符串，并按规范化后的值稳定去重。
- 只有第二次 RAG 后仍没有任何累计 Evidence 时，才实际使用这些关键词。

## 5. 确定性全文扫描

新增 `app/contract_review/absence.py`，负责纯 Python 扫描，不依赖 embedding、Vector、Rerank 或 Evidence Selector。

### 5.1 数据来源

通过现有 `ContractService` 根据 `contract_id` 找到合同目录，读取：

```text
<contract.storage_dir>/merged_content_list.json
```

不重新解析 PDF，不调用 MinerU，不修改解析文件。

### 5.2 可扫描文本

扫描对象沿用当前入库逻辑支持的类型和文本生成函数：

- `text`：使用对象原始 `text`。
- `table`：复用 `table_to_searchable_text()`，覆盖 `table_body`、`table_caption`、`table_footnote`，以及现有图片识别补充。
- `image`：复用 `image_to_searchable_text()`，使用当前 `structured_data` 已生成的可检索文本。

其他当前没有进入 RAG 业务的对象类型不纳入扫描。空文本对象跳过。

### 5.3 匹配与规范化

匹配前对候选文本和关键词执行相同规范化：

1. Unicode `NFKC`，统一常见全角与半角形式。
2. `casefold()`，统一英文大小写。
3. 合并或移除无意义空白和换行，使跨简单空白的固定短语可以匹配。

规范化后仅执行确定性子串匹配，不使用编辑距离、分词相似度或其他模糊算法。

一个 source object 命中一个或多个关键词即形成一个候选，同一 `source_object_index` 只返回一次。候选按以下顺序稳定排序：

1. 命中关键词数量降序。
2. 最长命中关键词的规范化长度降序。
3. `source_object_index` 升序。

最多返回前 20 个候选。每个候选至少包含：

```text
source_object_index
page_idx
node_type
matched_keywords
evidence_text
text
```

`source_object_index` 直接使用对象在 `merged_content_list.json` 数组中的真实下标。候选转换为现有 `Evidence`，`matched_keywords` 作为允许的附加字段保留。

## 6. 风险判断

### 6.1 全文扫描找到候选

候选不自动等同于满足规则。程序将 `absence_candidates` 作为真实 Evidence 交给现有风险判断 LLM，输入仍然是：

```text
rule_basis + review_goal + Evidence
```

LLM 只能返回 `risk`、`no_obvious_risk` 或 `needs_review`。证据引用字段由程序附加，LLM 不得生成或修改。

### 6.2 全文扫描没有候选

新增专用 absence-result prompt，仍使用现有风险判断 LLM。Prompt 明确提供以下确定性事实：

- 已执行两次 RAG，均未获得有效 Evidence。
- 已使用列出的关键词扫描完整解析内容。
- 全文扫描没有候选。

LLM 必须严格依据 `rule_basis` 和 `review_goal` 判断“缺失是否构成风险”以及风险等级，不得增加规范外的标准。所有结论必须限定为“基于当前合同全文解析结果未发现……”，禁止使用“合同肯定没有……”等绝对措辞。

`EvidenceStatus` 增加：

```text
absence_verified
```

其含义仅为“在当前完整解析结果中，确定性关键词扫描没有发现候选”。`RiskDecision` 继续执行原有跨字段校验：`risk` 必须有风险等级，非 `risk` 的风险等级必须为空，`insufficient` 必须对应 `needs_review`。`ReviewResult` 为 `absence_verified` 时 Evidence 必须为空。

风险等级不在 Python 中固定，由 absence-result 风险判断 LLM 根据输入规范判断。

## 7. 缺失核验可解释性

新增严格 Schema `AbsenceCheckMetadata`，并在 `ReviewResult` 增加可选字段 `absence_check: AbsenceCheckMetadata | None`。它由程序生成，不由 LLM 输出，用于保留最小、稳定的全文核验痕迹：

```json
{
  "absence_check": {
    "keywords": ["分包", "转包", "转委托", "委托第三方"],
    "candidate_count": 0
  }
}
```

规则如下：

- 没有进入 `absence_check` 的 ReviewResult，`absence_check=null`。
- 进入全文扫描后，无论是否找到候选，都保存实际使用的去重关键词和扫描候选总数。
- `candidate_count` 是应用候选上限之前的唯一命中 source object 数量，避免因为最多返回 20 条而丢失核验规模信息。
- 不保存整份合同文本、未命中的对象或 Rerank Debug。
- `absence_check` 由 Python 根据实际扫描过程附加，LLM 不允许生成或修改。
- `evidence_status=absence_verified` 时，`evidence=[]` 且 `absence_check.candidate_count=0`。
- 全文扫描找到候选并交给风险 LLM 时，最终 Evidence 保存实际提交给 LLM 的候选，`absence_check` 同时记录关键词及扫描到的候选总数。

这样未来 Web 页面即使只展示“基于当前合同全文解析结果未发现分包转包条款”，仍能说明系统在两次语义检索失败后使用了哪些关键词扫描完整解析结果，以及最终发现了多少候选对象。

## 8. 节点职责与路由

- `prepare_review_item`：初始化当前 ReviewItem 的查询、次数、Evidence、关键词、候选和判断状态，并发送 `review_item_started`。
- `retrieve_evidence`：调用现有 `search_contract()`，保存并去重真实 Evidence，发送 RAG 与 Rerank Debug 事件。
- `rewrite_query`：调用扩展后的 `RetrievalQueryRewrite`，保存第二次查询和扫描关键词，将次数设为 2。
- `risk_decision`：对累积 RAG Evidence 或 absence candidates 执行风险判断。
- `absence_check`：读取并扫描 `merged_content_list.json`，不调用 RAG。
- `absence_result`：在没有全文候选时调用专用风险判断 Prompt，生成 `absence_verified` 判断。
- `finalize_review_item`：组装 `ReviewResult`，将其追加到 `review_results`，递增 `current_item_index`。
- `aggregate_results`：保持现有纯代码汇总。

路由条件只读取 State，不产生额外副作用。一个 ReviewItem 完成后进入下一项的 `prepare_review_item`，全部完成后进入 `aggregate_results`。

## 9. Progress Callback

保留现有事件，新增：

```text
absence_check_started
absence_keywords_generated
absence_candidates_found
absence_confirmed
```

事件含义：

- `absence_check_started`：第二次 RAG 后仍无累计 Evidence，开始全文核验。
- `absence_keywords_generated`：打印实际使用的去重关键词。
- `absence_candidates_found`：打印候选的 `source_object_index`、`page_idx`、`node_type`、`matched_keywords` 和 `text`。
- `absence_confirmed`：没有候选时打印受限措辞的最终缺失判断。

命令行脚本为这些事件增加清晰中文标题。Progress 数据仅用于观察执行过程，不进入最终 Evidence 或风险输入，除非候选明确转换为 `absence_candidates` 后进入风险判断节点。

## 10. 错误处理

- 合同存在性和 ready 校验仍只在 `ContractReviewService.run()` 进入 Graph 前执行，不额外触发 RAG。
- `merged_content_list.json` 不存在、不是合法 JSON 数组或对象结构无效时，`absence_check` 失败并保留阶段上下文，不得把文件错误误判为条款缺失。
- LLM Schema 校验失败、扫描读取失败或风险判断失败均包装当前节点名称和 ReviewItem ID，保留原始异常为 cause。
- 空关键词列表属于 Schema 校验错误，不执行无意义全文扫描。

## 11. 测试与真实验证

采用 TDD，至少覆盖：

1. `RetrievalQueryRewrite.keywords` 的严格 Schema、去空和稳定去重。
2. Prompt 明确要求业务区分度，并禁止单独输出“同意、批准、许可、责任、合同”等泛化词。
3. text、table、image 使用现有可检索文本进行扫描。
4. NFKC、大小写和简单空白规范化。
5. 确定性子串匹配，不发生模糊误命中。
6. 真实数组下标作为 `source_object_index`，候选去重、排序和最多 20 条限制。
7. 两次 RAG 上限及 State 路由。
8. 全文候选存在时交给风险判断 LLM，引用字段不由 LLM 生成。
9. 全文无候选时使用 `absence_verified`，措辞受限且风险等级来自 LLM。
10. `absence_check` 元数据保存实际关键词和候选总数，且由程序附加。
11. Rerank Debug 和 absence Debug 不混入最终 Evidence。
12. 合同预检不触发 RAG 或全文扫描。
13. 聊天、Evaluation、text/table/image RAG 及入库流程回归不变。

最终使用当前 ready 合同和“分包转包限制”审查项执行真实命令行验证，完整打印：

```text
第一次 RAG
→ 第二次改写 RAG
→ absence_check
→ 全文候选
→ 最终风险结论
```
