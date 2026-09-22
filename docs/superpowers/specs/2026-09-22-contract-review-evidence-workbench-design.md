# Contract Review Evidence Workbench Design

## Goal

在现有合同风险审查应用中增加一套独立的“人工取证模式”。系统负责解析风险规范、逐项检索合同证据、提取关键事实、指出缺失资料，并为需要外部或内部补充信息的事项生成查询任务包；系统不自动给出风险、无风险或风险等级结论，最终判断全部由人工填写。

第一期以《中国寰球工程有限公司投标或拟签约民营及海外工程项目合同风险评估负面清单》作为一个真实回归样本。该 PDF 共四页，包含“投标立项审查负面事项”和“投标报价审查及签约审批负面事项”两个阶段，按最细层级拆分后有 42 个检查项。产品能力面向不同公司、不同项目和不同结构的中文合同风险规范；样本文档的公司名称、章节名称、编号、层级深度、项目数量和分类结果都不能成为解析器的固定配置或 Prompt 前提。

## Confirmed Product Boundary

第一期只做辅助取证和人工判断：

1. 完整保留风险规范中的所有检查项，不过滤需要联网或补充资料的项目。
2. 对合同内可取证事项，返回带页码和来源对象索引的合同 Evidence。
3. 从 Evidence 中提取金额、比例、期限、主体、国家、适用法律、争议解决地等关键事实。
4. 对需要公开查询、内部资料、专业数据库或专家意见的事项生成查询任务包，但不执行联网查询。
5. 人工填写风险结论、风险等级和审核意见；汇总只统计人工结论。
6. “未检索到证据”不能解释为“合同没有约定”，也不能自动形成风险结论。
7. 保留现有自动判断审查流程，但不作为新模式的默认入口。

第一期不实现自动联网、内部系统集成、多人审批流、权限体系、自动法律意见、自动修改建议或正式报告导出。

## Generalization Requirements

规则导入是面向合同风险规范领域的通用解析能力，不是针对当前负面清单制作的模板解析器：

1. 不假定文档存在两个阶段，也不假定标题为“投标立项”或“投标报价及签约审批”。
2. 不假定固定检查项数量；页面、进度、数据库和汇总全部根据本次解析结果动态计算。
3. 支持中文数字、阿拉伯数字、小数级编号、括号编号和项目符号，例如 `一、`、`1.`、`1.2`、`（3）`、`①` 和无编号列表。
4. 不限制层级为两级；解析结果使用 `section_path` 和父子关系表达任意合理深度。
5. 是否把某一段作为可执行检查项由其内容和层级关系决定，不能通过 `5.x`、`二-1` 等样本编号特判。
6. `evidence_scope`、`decision_mode`、事实需求和查询需求由每项规则语义生成，不能按样本编号查表。
7. 原始文字、原始编号和来源页码必须保留；系统生成的名称、分类和检索问题与原文分开存储。
8. 解析预览必须允许用户发现拆分错误后放弃该草稿；未确认的解析结果不能用于合同取证。

第一期聚焦中文合同风险规范，不承诺把任意类型 PDF 当成风险清单，也不承诺多语言规则解析。未来增加特定模板优化时，只能作为通用解析失败后的可插拔适配器，不能改变通用数据模型。

## Source Checklist Findings

真实样本揭示了两个相互独立的分类维度，不能只用“是否联网”一个字段表示。

### Evidence Scope

- `contract`：主要证据来自当前合同及已入库附件。
- `internal_material`：需要公司内部国别报告、历史回款、项目现金流、资源计划或技术经验等资料。
- `external_query`：需要工商、司法、制裁、信用等外部查询。
- `hybrid`：需要结合合同事实与内部或外部信息。

### Decision Mode

- `automatic_structure_check`：可以核验是否存在某类约定、是否缺少某项机制或数值之间的确定性关系。
- `threshold_required`：合同中可以取得事实，但“过高、过低、过长、过多”等结论依赖公司阈值。
- `expert_review`：需要商务、法务或技术专家判断合理性、可实现性或公平性。
- `query_and_compare`：需要补充查询结果，再由人工与规则进行比对。

人工取证模式不会使用 `decision_mode` 自动生成风险结论。它只用该字段决定页面提示、需要提取的事实和是否生成查询任务包。

样本中的 42 个最细检查项初步分布为：

- 13 项可在合同内完成主要结构性取证；
- 17 项可以从合同取证，但仍需要阈值或专家判断；
- 11 项需要合同结合外部查询或内部项目资料；
- 1 项是签约阶段重新评估立项事项的流程控制项。

## Approaches Considered

### Selected: Independent Evidence Workbench

新增独立的数据模型、服务入口和页面。模型输出 Evidence、关键事实和查询任务，人工决定风险结论。现有自动审查保持兼容。

该方案把机器生成内容和人工结论明确隔离，后续接入联网查询时只需补充查询任务的结果，不需要改写人工决策模型。

### Rejected: Force Existing Results to `needs_review`

在现有自动审查结果中把所有项强制标记为 `needs_review`，并附加查询信息。虽然改动较小，但会把证据不足、天然需要外部资料和等待人工判断混为一个状态，且现有 `risk_status`、`risk_level`、`risk_description` 语义仍暗示模型拥有判断权。

### Deferred: Full Review Workflow Platform

一次性增加任务分配、多人复核、审批流、权限和外部数据连接。当前业务需求尚未明确，第一期引入这些能力会扩大实现和验证范围。

## Architecture

新模式的数据流为：

```text
风险规范 PDF/TXT/MD
  → 规范文件校验与文本提取
  → 规则层级解析和分类预览
  → 人工确认规则集
  → 对 ready 合同创建取证运行
  → 逐项检索 Evidence
  → 提取关键事实、缺失资料和查询任务包
  → 证据工作台
  → 人工填写逐项结论
  → 仅基于人工结论汇总
```

架构拆为四个独立边界：

1. `RuleSetImporter`：负责风险规范文件校验、文本提取、层级解析和规则集确认。
2. `EvidenceReviewService`：负责逐项合同检索、事实提取和查询任务生成，不生成风险判断。
3. `EvidenceReviewRepository`：分别持久化不可变的机器取证结果和可修改的人工决策。
4. Evidence Workbench 页面：展示证据、查询任务、缺失资料和人工表单。

`EvidenceReviewService` 复用现有 `ContractService.search_contract()`、BM25/Vector 混合检索、Rerank、Evidence Selector、二次检索和全文缺失扫描能力，但不能调用当前的风险判断节点。

`item_kind=process_control` 的项目不进入合同检索。服务为其生成待办，列出需要重新执行或复核的前序规则项，并保持人工审核状态为 `pending`。这样“签约阶段重新评估立项事项”不会被误当成可从某一段合同原文直接判断的风险。

## Risk Rule Import

### Supported Inputs

新模式支持 `.pdf`、`.txt` 和 `.md`：

- TXT/MD 使用 `utf-8-sig` 解码，保留现有 UTF-8 要求。
- PDF 校验扩展名和 `%PDF-` 文件头，保存原始文件副本。
- 默认最大上传大小为 20 MiB，并通过配置项覆盖。
- 文件名只用于展示；存储路径使用系统生成的规则集 ID。

PDF 优先复用现有 MinerU 服务提取带页面信息的结构化内容，不为规则文件建立向量索引。若 MinerU 返回的页面没有有效文本，则规则导入失败并提示用户提供清晰 PDF 或 UTF-8 文本版本；第一期不在 Web 进程内新增另一套 OCR 引擎。

### Hierarchy Parsing

规则解析必须保留：

- 文档标题；
- 一级阶段标题；
- 原始编号，如 `一-5.3`、`二-6.6`；
- 原始风险文字；
- 父子层级；
- 来源页码。

只有最细层级的可执行标准成为默认检查项。仅用于归纳子项的父标题不重复执行。重新评估前序事项等流程规则仍保留为检查项，并标记 `item_kind=process_control`；该类型通过规则语义识别，不能依赖样本中的“二-1”等编号。

规则解析 LLM 使用严格 JSON Schema，只允许整理文档明确存在的规则，不得补充法律知识、行业阈值或公司制度。每个规则项至少包含：

```text
item_id
source_number
section_path
name
rule_text
source_pages
item_kind
evidence_scope
decision_mode
retrieval_queries
fact_requirements
research_requirements
```

导入完成后规则集状态为 `draft`。页面展示完整解析预览和检查项数量；用户确认后状态变为 `active`。只有 `active` 规则集可以创建取证运行。规则集一经用于运行便不可原地修改；重新导入或调整产生新版本，保证历史运行可追溯。

## Evidence Package

每个运行项生成一个 `EvidencePackage`：

```text
rule_item_id
evidence_status
evidence
extracted_facts
missing_sources
research_package
item_error
```

### Evidence Status

- `found`：检索链路返回有效合同证据。
- `not_found`：检索和全文扫描正常完成，但未取得足以展示的证据。
- `source_missing`：规则明确需要附件、关联协议或其他资料，而当前合同资料集不包含该来源。
- `extraction_failed`：该检查项的检索或结构化提取发生技术错误。

`not_found` 只描述本次机器取证结果，页面固定提示“未找到不等于合同没有约定，请人工核对全文及附件”。

### Evidence

Evidence 沿用项目现有的可信引用边界，至少包含：

```text
source_object_index
page_idx
node_type
evidence_text
```

引用字段直接来自 RAG 结果，LLM 不得生成或修改。

### Extracted Facts

关键事实采用列表而不是任意对象：

```text
fact_key
label
value
unit
evidence_indices
```

`evidence_indices` 必须指向同一 EvidencePackage 内的 Evidence。没有合同证据支持的值不能进入 `extracted_facts`，只能进入 `missing_sources` 或查询任务的缺失标识。

系统可以提取但不限于：企业名称、统一社会信用代码、法定代表人、注册地、项目所在地、合同金额和币种、付款节点、付款周期、预付款比例、质保金比例、责任上限、罚款比例、适用法律、争议解决机构和地点。

## Research Package

当 `evidence_scope` 为 `internal_material`、`external_query` 或 `hybrid` 时，系统生成 `ResearchPackage`：

```text
required
research_status
source_types
reason
query_targets
query_topics
comparison_points
missing_identifiers
```

### Research Status

- `not_required`
- `pending`
- `completed`

第一期系统只创建 `pending` 任务，不自动执行查询。`completed` 由人工录入查询结果后设置。

### Source Types

- `public_query`：公开网络信息。
- `internal_material`：公司内部资料。
- `professional_database`：需要授权的工商、司法、信用或制裁数据库。
- `expert_opinion`：商务、法务或技术专家意见。

### Query Targets

查询对象按主体类型保存，避免把所有字段拼成自由文本：

- 企业：公司全称、统一社会信用代码、注册国家或地区、注册地址、法定代表人、母公司、实际控制人、担保方。
- 项目：项目名称、国家、城市、现场地址、业主、合同金额、签约日期、融资主体和付款主体。
- 法律救济：适用法律、法院或仲裁机构、争议解决地、合同语言和已知资产所在地。
- 技术事项：技术名称、性能指标、专利商、保证责任主体和保证期限。

`query_topics` 说明要查什么，如经营异常、失信执行、重大诉讼、注册资本、股权穿透、制裁名单、历史拖欠、融资落实、国别风险或裁决执行难度。`comparison_points` 只复述原规则需要人工比对的标准，不补充未提供的阈值。

缺少公司名称、国家等关键标识时仍保留查询任务，并将字段名写入 `missing_identifiers`，页面显示“关键信息缺失”。

## Human Decision

机器取证完成后，每个检查项有独立的 `HumanDecision`：

```text
human_review_status
decision
risk_level
opinion
research_notes
updated_at
```

字段约束为：

- `human_review_status` 为 `pending` 或 `completed`。
- `decision` 为 `risk`、`no_obvious_risk` 或 `cannot_determine`。
- 只有 `decision=risk` 时可以填写 `high`、`medium` 或 `low`；风险等级允许为空，因为原规范可能没有分级依据。
- `opinion` 在完成时不能为空。
- `research_status=pending` 的项目不能保存为 `risk` 或 `no_obvious_risk`，只能保持待审核或选择 `cannot_determine`。
- 每次保存更新 `updated_at`。当前应用没有用户身份系统，第一期不记录或伪造审核人身份。

机器生成的 `EvidencePackage` 在运行完成后保持不可变，人工决策单独存储。重新取证产生新的运行，不能静默覆盖历史证据或人工结论。

## Persistence

新增持久化边界，而不是继续把可编辑人工数据写回 `review_runs.result_json`：

### Rule Sets

`review_rule_sets` 保存规则集 ID、名称、版本、源文件名、源文件存储位置、SHA-256、状态、解析结果和创建时间。源文件存放在应用管理目录，数据库不保存 PDF 二进制。

### Evidence Runs

`evidence_review_runs` 保存运行 ID、合同 ID、规则集 ID、机器取证运行状态、人工审核状态、进度、错误和时间戳。

机器运行状态沿用：

- `queued`
- `processing`
- `ready`
- `failed`

`ready` 表示机器取证已经结束，不表示人工审核完成。人工审核状态独立为：

- `pending`
- `in_progress`
- `completed`

### Run Items and Decisions

`evidence_review_items` 以 `(run_id, rule_item_id)` 唯一，保存规则快照和 EvidencePackage。即使规则集后续产生新版本，历史运行仍能显示当时规则原文。

`evidence_review_decisions` 与运行项一对一，保存 HumanDecision。人工保存使用事务和更新时间条件，避免页面重复提交静默覆盖更新后的数据。

## Web Experience

### Rule Set Import

规则集页面支持上传 PDF/TXT/MD，显示解析进度、层级预览、总检查项数、证据来源分类和判定方式。用户确认后激活版本。

对任意规则集，页面根据实际解析结果动态显示阶段、层级和检查项数量。当前真实样本的非发布本地回归检查应显示两个阶段和 42 个最细检查项，并保留 `一-5.3`、`二-6.6` 等原始编号；这一断言只验证样本解析质量，不进入业务分支。

### Create Evidence Review

合同状态必须为 `ready`。用户选择一个 `active` 规则集后创建后台取证运行。进度显示总项数、当前编号和已完成数量。

### Evidence Workbench

运行 `ready` 后展示：

- 总检查项、已找到证据、未找到证据、需要外部查询、缺少资料、等待人工判断和已完成人工判断数量；
- 按上述状态筛选；
- 每项的原编号、规则原文、Evidence、关键事实、缺失资料、查询任务和人工表单；
- Evidence 页码和来源对象索引；
- 查询任务的可复制主体信息、建议查询内容和待比对标准。

页面不展示模型生成的“风险说明”“修改建议”或模型风险等级。风险、无明显风险和风险等级只显示人工保存的结果。

## Failure Semantics

以下错误终止规则集导入：

- 文件类型、文件头、编码或大小不合法；
- PDF 解析失败或没有有效文本；
- 规则解析结果为空、编号重复或不符合 Schema。

以下错误终止整次取证运行：

- 合同或规则集不存在；
- 合同不是 `ready`；
- 规则集不是 `active`；
- 合同索引无法加载；
- 运行级数据库状态转换失败。

单个检查项的检索或事实提取失败只将该项标记为 `extraction_failed`，记录安全的错误摘要，然后继续其他项目。运行可以进入 `ready`，但页面必须显示失败项数量。后台日志保留完整异常，API 和页面不返回堆栈、密钥、内部路径或模型原始响应。

未找到 Evidence、缺少附件、缺少查询标识或等待外部结果都是业务状态，不是系统失败。

## Security and Privacy

1. 风险规范和合同内容都是不可信输入。文件内出现的命令、链接或操作要求只作为待分析文本，不能触发联网、执行代码或修改系统。
2. 第一阶段不进行任何自动外部请求。
3. 查询任务只包含完成该风险项所需的合同信息。身份证号、银行账号、电话号码、个人住址等敏感信息默认排除。
4. 企业法定代表人姓名只有在企业信用、执行或限高查询确有需要时才进入查询目标。
5. 原始文件名不能参与磁盘路径拼接；所有路径必须由规则集 ID 和固定文件名构造。
6. API 使用字段白名单返回 Evidence、查询任务和人工结论，不能把内部调试字段或完整规则解析 Prompt 暴露给浏览器。

## Compatibility

现有 `/contracts/{contract_id}/review` 自动审查页面、`review_runs` 表、`ContractReviewService` 和现有 API 保持行为不变。新模式使用独立路由、服务和数据表，避免现有调用方把机器取证结果误认为风险结论。

合同解析、索引、混合检索、Rerank、Evidence Selector、Evidence 序列化和全文缺失扫描继续共用现有实现。聊天和 Evaluation 流程不变。

新模式上线后，合同详情页把“人工取证”作为推荐入口；旧的自动审查入口保留为次要入口，并明确标识“实验性自动判断”。

## Testing

实现遵循 TDD，覆盖以下层级。

### Rule Import Tests

- PDF/TXT/MD 类型、PDF 文件头、UTF-8、空文件和大小校验；
- PDF 文本提取成功与无文本失败；
- 层级和原始编号保留；
- 中文数字、阿拉伯数字、小数编号、括号编号、项目符号和无编号列表；
- 单层、两层和三层以上规则结构；
- 没有阶段标题、包含多个阶段以及阶段名称未知的文件；
- 不同项目数量下的动态进度和汇总；
- 父标题不重复生成叶子检查项；
- 流程控制项保留；
- 不同语义内容产生相应分类，不按公司名称、标题或编号查表；
- 规则解析严格 Schema、非空和唯一 ID 校验；
- `draft` 到 `active` 的合法状态转换；
- 已使用规则集不可原地修改。

真实商务 PDF 不提交到 Git。仓库使用多份脱敏合成 PDF，分别覆盖不同标题、编号体系、层级深度、项目数量和规则语义。真实文件只作为本地回归样本，验证该文件的四页、两个阶段和 42 个最细检查项，不能作为通用测试的唯一输入。

### Evidence Service Tests

- 服务只调用合同检索和事实提取，不调用风险判断 LLM；
- Evidence 引用只能来自检索结果；
- 事实必须引用当前项 Evidence；
- 合同、内部、外部和混合类型产生正确的 ResearchPackage；
- 缺少查询标识写入 `missing_identifiers`；
- `not_found`、`source_missing` 和 `extraction_failed` 语义分离；
- 单项失败后后续项目继续执行；
- 运行汇总不包含任何模型风险数量。

### Human Decision Tests

- 结论、等级、意见和研究状态的跨字段校验；
- 外部查询未完成时不能保存确定风险或无风险结论；
- 保存决策后人工审核进度正确更新；
- 汇总只统计人工结论；
- 并发更新时间冲突不会静默覆盖。

### Web and Privacy Tests

- 规则导入预览、合同取证进度、状态筛选和人工保存流程；
- 页面显示页码、关键事实、缺失资料和查询任务；
- 页面不显示模型风险判断字段；
- 查询任务排除身份证号、银行账号、电话和个人住址；
- 对外错误不泄露内部异常、路径、密钥或模型响应。

## Acceptance Criteria

第一期完成必须满足：

1. 可以上传并激活不同标题、编号体系、层级深度和项目数量的 PDF/TXT/MD 中文合同风险规则集。
2. 解析器不包含当前样本的公司名称、章节名称、编号、项目数量或分类查表；真实样本仅在本地回归检查中稳定解析为两个阶段、42 个最细检查项，并保留原编号和页码。
3. 可以对一个 `ready` 合同执行全部检查项的证据检索和事实提取。
4. 需要补充资料的项目生成包含合同关键信息、查询主题、来源类型和待比对标准的查询任务包。
5. 单项失败不阻断其余项目，所有非成功状态均可在页面筛选。
6. 系统不生成风险、无风险、风险等级、风险说明或修改建议。
7. 人工可以保存逐项结论和意见，汇总只使用人工结果。
8. 现有自动审查、聊天、Evaluation 和合同入库流程保持兼容。
