# 扫地机器人智能客服 Agent（LangChain + LangGraph + Chroma + RAG）

面向私有知识库场景的 **Agentic RAG 私有文档问答系统**：支持本地 PDF / TXT 文档批量导入构建向量知识库，
由 ReAct Agent 自主编排「知识库检索、天气查询、外部 CSV 业务数据读取、报告生成」等多工具完成任务，
并通过流式输出实现打字机式回答。整体围绕 **模块化 + 配置化 + 可观测** 三条工程主线设计。

---

## 一、功能特性

| 能力 | 说明 |
| --- | --- |
| 私有知识库问答 | 本地 PDF / TXT 批量导入，文本分块 → 向量化 → Chroma 入库，TopK 语义检索后由大模型总结回答 |
| 知识库增量更新 | 基于文件 MD5 判断是否已入库，重复文件自动跳过，避免重复向量化 |
| ReAct 多工具编排 | 注册 7 个工具，模型自主决定调用顺序（如「取用户ID → 取月份 → 查使用记录 → 检索保养知识」） |
| 自定义中间件 | 工具调用监控、工具参数规范化、模型调用埋点、动态提示词切换 |
| 动态 Prompt 切换 | 识别为报告类需求后自动切换为报告提示词，普通咨询使用系统提示词 |
| 流式输出 | 以生成器形式逐段返回模型输出，实现打字机效果 |
| 工程化容错 | 统一日志组件 + 单文件 / 单工具失败不中断整体任务 |

---

## 二、技术栈

- **语言**：Python 3.10
- **Agent 框架**：LangChain（`create_agent`）、LangGraph（状态图 / 运行时上下文 / 中间件）
- **RAG**：LangChain LCEL、RecursiveCharacterTextSplitter、Chroma 向量库
- **模型**：DashScope 通义千问 `qwen3-max`（对话）、`text-embedding-v4`（向量）
- **配置**：PyYAML（参数外部配置化）、python-dotenv（密钥管理）
- **文档解析**：txt / pdf 文档加载（pypdf）

---

## 三、目录结构

```
agent_项目开发/
├── agent/                     # 智能体层
│   ├── react_agent.py         # ReAct Agent 封装：工具注册 + 中间件挂载 + 流式输出
│   └── tools/
│       ├── agent_tools.py     # 7 个工具：RAG 检索 / 天气 / 用户信息 / 外部 CSV / 报告上下文
│       └── middleware.py      # 4 个中间件：工具监控 / 参数规范化 / 模型埋点 / 动态提示词切换
├── config/                    # 配置层（YAML 外部配置化）
│   ├── agent.yml              # 外部数据路径、演示用户池
│   ├── chroma.yml             # 向量库参数、文本分块参数、知识库源目录
│   ├── prompts.yml            # 各类提示词文件路径
│   └── rag.yml                # 模型名称、采样参数、重试次数
├── data/                      # 知识库源文件 + 外部业务数据
│   ├── *.txt                  # 扫地机器人知识库文档
│   └── external/records.csv   # 用户月度使用记录（外部业务数据）
├── model/
│   └── factory.py             # 模型工厂：抽象基类 + 对话模型 / 向量模型工厂（懒加载单例）
├── prompts/                   # 提示词（与代码解耦）
│   ├── main_prompt.txt        # 系统提示词（咨询 / 故障 / 选购 / 保养）
│   ├── rag_summarize.txt      # RAG 总结提示词
│   └── report_prompt.txt      # 报告生成提示词
├── rag/                       # 检索增强层
│   ├── vector_store.py        # 知识库构建、MD5 增量更新、向量检索
│   └── rag_service.py         # LCEL 链路：检索 → 提示词组装 → 模型 → 结果解析
├── scripts/
│   └── generate_demo_records.py  # 生成演示用外部业务数据
├── tests/
│   └── test_offline.py        # 离线自测（无需 API Key / 不产生网络调用）
├── utils/                     # 通用工具包
│   ├── config_handler.py      # 配置读取
│   ├── exception_handler.py   # 异常捕获 / 重试装饰器
│   ├── file_handler.py        # MD5 计算、目录扫描、txt/pdf 文档加载
│   ├── logger_handler.py      # 统一日志组件（控制台 + 按日期归档文件）
│   ├── path_tool.py           # 统一绝对路径
│   └── prompt_loader.py       # 提示词加载
├── main.py                    # 工程入口（构建知识库 / 单次提问 / 交互式对话）
└── requirements.txt
```

---

## 四、快速开始

### 1. 环境要求

- Python 3.10+
- 可访问阿里云百炼（DashScope）的网络环境

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

编辑 `.env`，填入自己的 DashScope API Key：

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 4. 构建向量知识库

```bash
python main.py --build-kb          # 增量构建（已入库文件自动跳过）
python main.py --rebuild-kb        # 清空向量库后全量重建
```

### 5. 运行

```bash
python main.py                                     # 交互式对话
python main.py --ask "扫地机器人迷路了怎么办"        # 单次提问
python main.py --report                            # 生成当前用户当月使用报告
```

---

## 五、配置说明

| 配置文件 | 关键配置项 | 作用 |
| --- | --- | --- |
| `config/rag.yml` | `chat_model_name` / `embedding_model_name` / `temperature` / `max_retries` | 模型名称与采样参数，切换模型只改这里 |
| `config/chroma.yml` | `persist_directory` / `k` / `chunk_size` / `chunk_overlap` / `separators` / `data_path` | 向量库与分块参数；`k` 为检索 TopK |
| `config/prompts.yml` | `main_prompt_path` / `rag_summarize_prompt_path` / `report_prompts_path` | 提示词文件路径 |
| `config/agent.yml` | `external_data_path` / `user_id_pool` / `user_city_pool` | 外部业务数据路径与演示用户池 |

> `persist_directory` 等相对路径在代码内部会通过 `utils.path_tool.get_abs_path` 转换为绝对路径，
> 因此无论从哪个目录启动脚本，读写到的都是同一份向量库。

---

## 六、核心实现说明

### 1. 模块化工程分层

`utils` 工具包抽离通用能力（路径处理、日志管理、文档加载、配置读取、提示词加载、异常容错），
业务层（`agent` / `rag` / `model`）只关注自身职责，实现业务逻辑与底层工具解耦。

### 2. 模型工厂模式

`model/factory.py` 基于抽象基类 `BaseModelFactory` 统一 `generator()` 接口，
`ChatModelFactory` / `EmbeddingsFactory` 分别生产对话大模型与 Embedding 向量模型，
模型名称、采样参数、重试次数全部来自 `config/rag.yml`，便于快速切换模型。
实例采用 `lru_cache` 懒加载单例，**导入模块时不会校验 API Key**，
因此未配置密钥也能正常执行 `python main.py --help` 等命令。

### 3. 基于 MD5 的知识库增量更新

`rag/vector_store.py` 对每个候选文件计算 MD5，与 `md5.txt` 中记录比对：
已入库的文件直接跳过，避免重复向量化；新增 / 修改过的文件才会重新分块入库。
当前配置为 `chunk_size=200`、`chunk_overlap=20`、检索 `TopK=3`。

### 4. LCEL 构建 RAG 推理链路

```
PromptTemplate(rag_summarize.txt) | ChatTongyi | StrOutputParser
```

检索命中的文档片段会携带来源元数据拼接为「参考资料」，与用户问题一起送入大模型总结，
未命中时返回明确提示而不是让模型自由发挥，从工程层面抑制幻觉。

### 5. ReAct Agent 多工具编排与中间件

- 工具：`rag_summarize`、`get_weather`、`get_user_id`、`get_user_location`、`get_current_month`、
  `fetch_external_data`、`fill_context_report`
- 中间件（共 4 个）：
  - `monitor_tool`（`@wrap_tool_call`）：记录工具名 / 入参 / 耗时，**工具异常时返回错误 ToolMessage 而不是抛出**，
    保证单工具失败不中断整体任务；
  - `normalize_tool_calls`（`@wrap_model_call`）：修正历史消息中不合法的 `function.arguments`。
    部分模型（如 qwen3-max）流式返回工具调用时会拼接出非法 JSON，LangChain 仍能解析出可用 args 并执行工具，
    但把原始字符串回传给 DashScope 会被拒绝并报 `400 ...must be in JSON format`，导致整轮对话中断；
    这里统一以解析后的 args 为准重新序列化，保证回传的历史消息始终是合法 JSON；
  - `log_before_model`（`@before_model`）：记录进入模型前的消息规模与最新消息摘要；
  - `report_prompt_switch`（`@dynamic_prompt`）：按运行时上下文动态选择系统提示词。
- 工具入参容错：`fetch_external_data` 的 `month` 允许省略（默认取当前月份），
  避免模型漏传参数导致整轮报告任务失败。

### 6. 动态 Prompt 切换闭环

```
模型调用 fill_context_report
      ↓
monitor_tool 捕获工具调用 → runtime.context["report"] = True
      ↓
report_prompt_switch 检测到 report=True → 返回报告提示词
```

每轮会话使用独立的上下文对象（`ReactAgent._new_context`），避免报告场景串到普通问答。

### 7. 流式输出

`ReactAgent.execute_stream` 使用 `stream_mode="messages"` 以「消息增量」为粒度消费 LangGraph 流，
只透出模型节点的正文增量，避免 `values` 模式下整段内容重复输出，前端可获得打字机效果。

### 8. 全局日志与异常容错

`utils/logger_handler.py` 统一封装日志组件，同时输出控制台与按日期归档的本地日志文件；
知识库加载（逐文件 try/except）、工具调用（中间件兜底）、报告数据读取（缺失返回空串）等关键环节均做了异常捕获，
并配合 `utils/exception_handler.py` 的 `catch_exceptions` / `retry` 装饰器使用。

---

## 七、离线自测

不依赖 API Key、不产生网络调用，用于验证工程结构是否正确：

```bash
python tests/test_offline.py
```

覆盖内容：配置读取、路径工具、MD5 计算、外部 CSV 数据解析、Agent 图构建、中间件上下文切换、
工具调用参数规范化、工具异常容错（共 30 项断言）。

---

## 八、常见问题

| 问题 | 处理方式 |
| --- | --- |
| `Did not find dashscope_api_key` | 检查 `.env` 是否配置，或直接设置环境变量 `DASHSCOPE_API_KEY` |
| 检索不到内容 / 回答「知识库中未检索到」 | 先执行 `python main.py --build-kb` 构建知识库 |
| PDF 文件解析失败提示缺少依赖 | 执行 `pip install pypdf` |
| 想更换模型 | 修改 `config/rag.yml` 中的 `chat_model_name` / `embedding_model_name` 后重新构建知识库 |
| Ctrl+C 中断后无响应 | 已做 KeyboardInterrupt 捕获，终端会提示「已中断，再见！」 |