# Table RAG Design

## Goal

为现有合同 RAG 增加 MinerU table 的首版向量化、召回和 Evidence 支持，同时保持现有 text Node 的行为不变。

## Scope

- 一张 MinerU table 对应一个 Node。
- 使用纯 Python 代码把 `table_body` HTML 转成 `searchable_text`。
- 把 `table_caption`、`table_footnote` 纳入 `searchable_text` 和 table 的检索上下文输入。
- 保留 `source_object_index`、`page_idx`、`bbox`、原始 `table_body`，并设置 `node_type=table`。
- table 生成 `retrieval_context`，并参与现有向量召回、重排和 Evidence Selector 流程。
- 暂不处理跨页 table 合并、超大 table 拆分和视觉布局复原。

## Reference and dependency decision

参考同级项目 `pdf_trans` 的 `table_translation.py`：其 HTML 解析采用 Python 标准库 `html.parser.HTMLParser` 的事件式解析器，而不是 BeautifulSoup 或 lxml。合同 RAG 只借鉴其解析边界、标签栈校验和不依赖外部 HTML 包的方式，不引入翻译项目的公式保护、翻译批处理和 HTML 重建逻辑。

## Data flow

```text
merged_content_list.json
        │
        ├── text object ──> 现有 text Node 分支（行为保持不变）
        │
        └── table object
              ├── table_body HTML ──> table_searchable_text.py ──> searchable_text
              ├── searchable_text + section path ──> retrieval_context
              └── searchable_text + retrieval_context ──> table Node ──> embedding/index

table Node metadata ──> retrieval result ──> final Evidence
                         ├── source_object_index/page_idx/bbox
                         └── table_body/table_caption/table_footnote
```

## Searchable text

新增 `table_searchable_text.py`，使用 `HTMLParser` 按文档顺序读取 `table`、`tr`、`td`、`th` 和嵌套文本。输出是稳定的纯文本，不包含 HTML 标签，例如：

```text
表格标题：付款计划
第1行：支付次序 | 应付比例 | 支付条件 | 支付期限
第2行：1 | 30% | 合同生效 | 10个工作日内
表格注释：乙方提供正规合法的发票
```

`rowspan`、`colspan` 属性不参与首版布局展开；单元格仍按 HTML 文档顺序输出。原始属性和 HTML 通过 `table_body` 原样保留。`table_body` 缺失时仍为该 table 创建一个 Node，使用可用的 caption、footnote；全部为空时使用不含事实的固定占位文本 `表格`。

## Retrieval context

保留现有 text 的筛选、章节路径推导、Prompt 和生成逻辑。新增 table 分支：

- 章节路径从 table 前方最近的 text 标题推导；
- table 的 `searchable_text` 作为当前内容输入；
- caption 和 footnote 已在 searchable_text 中，因此也会进入 table context 生成；
- 输出仍按 `source_object_index` 保存到 `retrieval_context.json`。

## Node and embedding

现有 text Node 构建代码不改变。table 新建一个 `TextNode`：

- `node.text = searchable_text`；
- metadata 包含 `node_type=table`、`source_object_index`、`page_idx`、`bbox`、`table_body`、`table_caption`、`table_footnote` 和 `retrieval_context`；
- 仅 `retrieval_context` 作为 Embed metadata，原始表格 metadata 保留但排除出 embedding；
- 因此 table embedding 内容是 `searchable_text + retrieval_context`。

## Evidence

现有召回、重排和 Evidence Selector 继续读取 `node.text`，因此 table 使用 searchable_text 参与语义判断。最终 Evidence 在 table Node 上额外序列化 `node_type`、`bbox`、`table_body`、`table_caption` 和 `table_footnote`，同时保留现有的 source index 和 page 字段，使原始 HTML 表格可追溯。

## Non-goals

- 不改变普通 text Node 的文本、metadata、embedding 或 Evidence 行为。
- 不重新解析 PDF，不修改 MinerU 原始输出。
- 不做跨页 table 合并。
- 不做超大 table 的切分。
- 不引入 BeautifulSoup、lxml 或翻译项目专用依赖。

## Verification

- HTML 转 searchable_text 的单元测试。
- text Node 回归测试。
- table 一对一 Node、metadata 和 embedding 内容测试。
- table retrieval_context 生成与持久化测试。
- table Evidence 原始信息序列化测试。
- 定向测试、全量可运行测试、compileall 和 `git diff --check`。
