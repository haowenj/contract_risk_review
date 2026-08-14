# MinerU 原始解析测试脚本设计

## 目标

在合同审查项目中新增一个独立的最小测试脚本：接收一份 PDF 合同，调用本地或远端 MinerU，保存完整的原始 `content_list.json`，并打印用于分析数据结构的基础统计信息。

## 参考实现与边界

参考 `/Users/wenjuhao/code/python/pdf_trans/src/pdf_trans/client.py` 的 MinerU 3.4.4 异步 HTTP 协议，但不依赖 `pdf_trans` 包，也不复制其工作流层代码。

脚本明确不做以下工作：

- 不导入或使用 LlamaIndex
- 不创建 `Document`、Index、Retriever 或其他 LlamaIndex 对象
- 不做内容清洗、跨页合并、分类、翻译、公式处理或渲染
- 不修改 MinerU 返回的原始对象结构
- 不保存清洗后内容、Markdown、渲染文件或翻译结果

## 命令行接口

新增项目根目录脚本 `mineru_raw_parse.py`：

```text
python mineru_raw_parse.py INPUT_PDF [--output OUTPUT_JSON]
                         [--svr-url MINERU_URL]
                         [--backend BACKEND]
                         [--server-url SERVER_URL]
```

- `INPUT_PDF`：必填，必须存在且扩展名为 `.pdf`。
- `--output`：可选，默认写到输入 PDF 同目录下的 `<stem>_mineru_raw.json`。
- `--svr-url`：可选，默认 `http://127.0.0.1:7100`；也支持环境变量 `PDF_TRANS_MINERU_URL`。
- `--backend`：可选，默认 `hybrid-engine`；也支持环境变量 `PDF_TRANS_MINERU_BACKEND`。
- `--server-url`：可选，用于 `hybrid-http-client`；也支持环境变量 `PDF_TRANS_MINERU_SERVER_URL`。

命令行参数优先于环境变量。脚本可调用现有 `.env`，但不依赖任何 LLM 配置。

## MinerU 调用流程

使用 `httpx.Client` 完成以下最小流程：

1. 向 `<svr-url>/tasks` 以 multipart 方式上传 PDF。
2. 发送参考项目中的解析表单，至少保留：
   - `parse_method=auto`
   - `effort=medium`
   - `formula_enable=true`
   - `table_enable=true`
   - `image_analysis=false`
   - `return_content_list=true`
   - `return_images=true`
   - `response_format_zip=true`
   - `return_md=false`
   - `return_middle_json=false`
   - `return_model_output=false`
3. 读取 `task_id`、`status_url`、`result_url`。
4. 对 `status_url` 轮询 `pending`/`processing`，直到 `completed`；`failed` 或未知状态直接报错。
5. 下载结果 ZIP，定位唯一文件名以 `_content_list.json` 结尾的成员。
6. 将该成员的原始字节直接写入目标 JSON 文件，不经过 `json.dumps` 重序列化。

## 统计输出

保存原始字节后，再将同一 JSON 解析为列表，仅用于打印：

- 页数：解析对象中不同 `page_idx` 的数量；忽略缺少 `page_idx` 的对象。
- 解析对象总数：原始列表长度。
- 各 `type` 数量：按每个对象的 `type` 字段计数；缺少字段时计入 `<missing>`，非对象元素计入 `<non-object>`。

统计信息不写回 JSON 文件，输出使用中文标签并保持稳定、易读的顺序。

## 文件与依赖

- Create: `mineru_raw_parse.py`：独立 CLI 和 MinerU 最小调用流程。
- Create: `tests/test_mineru_raw_parse.py`：使用模拟 HTTP 响应验证提交、轮询、原始 JSON 保存和统计。
- Modify: `pyproject.toml`：添加直接依赖 `httpx`。
- Modify: `uv.lock`：由 `uv` 更新锁文件。

不修改 `pdf_trans` 项目，不新增 LlamaIndex 相关代码。

## 验证标准

- 测试能验证 `POST /tasks`、状态轮询和 ZIP 中原始 `content_list.json` 被保存。
- 测试能验证页数、对象总数和 `type` 计数。
- 测试能验证不对 JSON 内容做结构性改写。
- 脚本能通过 Python 编译检查。
- 项目现有测试继续通过。
