# evals —— 评测集与跑分脚本

这个目录回答一个问题：**「这个 Agent 到底有多准？」**

主 README 里写「支持多工具编排」「工程化容错」都是形容词，这里是把它变成名词的地方：
Recall@3 多少、工具选择准确率多少、关键事实命中率多少。

```
evals/
├── build_golden.py         # 从语料自动生成检索评测集（golden set）
├── golden_retrieval.jsonl  # 生成产物：检索用例（自动派生，别手改）
├── golden_multihop.jsonl   # 手写用例：多跳 / 工具编排 / 幻觉抑制
├── run_eval.py             # 跑分脚本（retrieval / agent / judge 三种模式）
├── inspect_chunks.py       # 离线分块体检：答案落在哪一块、有没有被切碎
├── namecheck.py            # 零依赖静态自检：扫出「引用了未定义名字」这类必炸 bug
└── results/                # 运行产物：latest.md 是最新报告，chunks.md 是分块体检
```

---

## 一、快速开始

```bash
# 0) 前置：知识库要建好，DASHSCOPE_API_KEY 要配好
python main.py --rebuild-kb

# 1) 生成检索评测集（不花钱）
python evals/build_golden.py --with-manuals

# 2) 检索指标：只调用 embedding，不调用对话模型（几分钱）
python evals/run_eval.py --mode retrieval

# 3) 端到端：跑完整 Agent，看工具编排和事实准确性（要花钱，先用 --limit 3 试水）
python evals/run_eval.py --mode agent --limit 3
python evals/run_eval.py --mode agent
python evals/run_eval.py --mode agent --only mh-015    # 只重跑某一条：改完判分/工具后验证用，最省钱

# 4) 可选：再叠一层大模型评委，专抓规则判分抓不到的幻觉
python evals/run_eval.py --mode agent --judge

# 5) 检索掉分的时候，先体检分块（不联网、不花钱、1 秒出结果）
python evals/inspect_chunks.py
python evals/inspect_chunks.py --chunk-size 400 --overlap 40    # 试算调参效果，不用重建知识库

# 6) 改完 evals/*.py 先做静态自检，别让 NameError 花你的钱（不联网、不花钱）
python evals/namecheck.py --selftest      # 先验证检查器本身是好的
python evals/namecheck.py evals           # 再扫评测脚本
```

跑完看 `evals/results/latest.md`。

---

## 二、三个模式分别测什么

| 模式 | 测什么 | 花不花钱 | 指标 |
| --- | --- | --- | --- |
| `retrieval` | 向量检索召回能力 | 只花 embedding | Recall@k、拼接 Recall@k、Hit@1、MRR |
| `agent` | 工具编排 + 参数补全 + 事实准确性 | 每条一次多轮对话 | 工具选择准确率、参数正确率、事实命中率、通过率 |
| `judge` | 回答是否忠实于工具返回 | 每条一次判分调用 | 忠实率、平均分（1–5） |

**为什么要把检索和生成分开测**：混在一起跑，分数掉了你不知道是「检索没召回」还是
「模型没总结好」。拆开以后，检索指标是确定性的、不依赖模型版本，可以天天跑。

---

## 三、评测集怎么来的

### 检索集：从语料自动派生

`build_golden.py` 直接解析 `data/` 下的语料，把**语料原文当成标准答案**：

| 来源 | 解析规则 | 问句从哪来 | kind |
| --- | --- | --- | --- |
| `扫地机器人 100 问 1.txt` | `## 章节` + `N. 问题` + `答：回答` | 语料里的原始问句 | `corpus_qa` |
| `故障排除.txt` | `## N. 标题` + 正文 | 模板：`扫地机器人{标题}怎么办？` | `synthetic` |
| `维护保养.txt` | `N. 名目：说明` | 模板：`扫地机器人的{名目}平时怎么保养？` | `synthetic` |
| `选购指南.txt` | `N. 名目：说明` | 模板：`买扫地机器人，{名目}应该怎么选？` | `synthetic` |

三个刻意的设计：

1. **可重建**：知识库改了，评测集一条命令重生成，不会和语料版本脱节；
2. **可复现**：`sample_evenly` 只按语料顺序等距抽样，同样的输入永远产出同一份评测集，
   所以「换分块参数前后」的分数才可比；
3. **分层统计**：`corpus_qa` 是真实用户口吻，`synthetic` 是模板改写。
   报告里分开列，两者的召回差距本身就说明问题。

> ⚠️ **参考答案必须和语料原文逐字一致。**
> 早期版本在解析故障手册时把行首的 `- ` 剥掉了，导致参考答案和语料里的块对不上，
> 包含度被卡在 0.57–0.64 之间，看起来像「召回失败」，其实是评测集自己写错了。
> `inspect_chunks.py` 就是为抓这类问题写的：它会把「金标对不上语料」单独标成一类。

### 多跳集：手写 15 条

`golden_multihop.jsonl` 是手写的，覆盖自动生成测不到的东西：

- **参数补全**：报告类需求要自己补 `user_id` 和月份，不能反问用户；
- **多跳链路**：`get_user_location -> get_weather`、`get_user_id -> fetch_external_data -> rag_summarize`；
- **相对时间**：「上个月」要换算成 2026-08，模型直接传当前月就会拿错数据；
- **负例**：纯咨询问题不许误调 `fetch_external_data` / `fill_context_report`；
- **参数直传**：用户给了城市就该直接传，不该再去查用户定位；
- **防幻觉**：知识库里没有的（洗碗）必须拒答，表里没有的月份必须如实说没有。

每条用例的字段：

```jsonc
{
  "id": "mh-011",
  "question": "我上个月的使用记录呢？",
  "expected_tools": ["get_user_id", "fetch_external_data"],  // 必须出现
  "expected_calls": [                                       // 参数级：要求存在一次「同名 + 参数是它的超集」的调用
    {"name": "fetch_external_data", "args": {"month": "2026-08"}}
  ],
  "forbidden_tools": [],                                     // 不许出现
  "recommended_tools": ["get_current_month"],                // 建议出现，不参与判分
  "must_include": ["60.0"],                                  // 回答里必须出现的字符串
  "must_match": [],                                          // 回答里必须匹配的正则
  "must_not_include": [],                                    // 不许出现的字符串
  "note": "为什么这么设计"
}
```

---

## 四、判分口径

### 检索命中：包含度 >= 0.6

一段检索结果算「命中」，当且仅当标准答案有足够比例**连续**出现在这段文本里：

```
包含度 = 最长公共子串长度 / 标准答案长度
```

用最长公共子串而不是编辑距离，是因为分块会把一段答案从中间切断，
这个口径对「被切一半」最宽容，也最好向别人解释。

| 指标 | 口径 |
| --- | --- |
| `Recall@k` | 前 k 块里**有一块完整**包含答案的题目占比 |
| `拼接 Recall@k` | 前 k 块**拼接后**包含答案的题目占比（把「被切碎」也算进来） |
| `Hit@1` | 第 1 块就完整包含答案的题目占比 |
| `MRR` | 所有题目 `1/首次命中排名` 的平均值，没命中算 0 |

### 失败要分类，不然没法行动

召回失败有两个原因，在总分上长得一模一样，修法却完全相反：

| 现象 | 判据 | 该往哪修 |
| --- | --- | --- |
| **答案被切碎** | 单块没命中，但拼接后命中 | 调大 `chunk_size` / 按语义单元切分 |
| **压根没召回到** | 拼接后也不命中 | 调 embedding / 加 rerank / 补 query 改写 |
| **金标对不上语料** | 语料里根本找不到这段原文（`inspect_chunks.py` 报） | 评测集自己的问题，先修这个 |

`run_eval.py` 的检索报告会把前两类直接拆开列出来，
`inspect_chunks.py` 负责第三类和「碎片块」风险。

### 端到端：工具 + 参数 + 事实三维度

一条用例只有在**工具选对**、**参数传对**、**关键事实命中**时才算通过：

- 工具维度：`expected_tools` 全部出现，且 `forbidden_tools` 一个都没出现；
- 参数维度：`expected_calls` 里每一条，都存在一次「同名 + 参数是它的超集」的真实调用；
- 事实维度：`must_include` / `must_match` 全部满足，`must_not_include` 一个都没出现。

**为什么参数要单独算一维**：工具名对了但参数漏传，是最难自查的一类失败。
比如 `fetch_external_data(user_id="001")` 漏了 `month`，工具内部会静默兜底成当前月，
返回一份**格式完全正常、内容是错月份**的数据 —— 报告里只看得到「事实没命中」，
你还得自己反推是数据错了还是模型总结错了。参数维度把它直接指名道姓地写出来：

```
fetch_external_data 期望 {"user_id": "001", "month": "2026-06"}，实际 {"user_id": "001"}
```

> 工具压根没被调用时，参数维度**不报错** —— 那是 `expected_tools` 的职责。
> 两边都报，同一条用例会在报告里出现两遍，反而看不清。

事实锚点全部来自真实数据（`data/external/records.csv` 和知识库原文），
所以判分是确定性的 —— **不把分数交给大模型评委的自由裁量**。
`--judge` 只是额外一层保险，用来抓规则覆盖不到的幻觉。
### 为什么要冻结环境

项目里 `get_user_id` / `get_user_location` 是随机取值的（演示用），
`get_current_month` 又跟着系统时间走。不冻住的话，今天 90 分的用例明天就 60 分，
分数完全不可比。所以 `run_eval.py` 默认：

- 用户池冻结成 `["001"]`、城市池冻结成 `["深圳"]`；
- `get_current_month` 冻结成 `2026-09`。

于是「我上个月」永远是 2026-08，「本月」永远是 2026-09，对应用户 001 的那两行记录。

> 想跑真实时间：`--freeze-date ""`。注意那时 `mh-001 / mh-010 / mh-011` 的期望值会失效。

---

## 五、调参实验记录

有了固定的评测集，改一个参数才算「实验」而不是「碰运气」。

### 实验 1：chunk_size 200 → 400

语料是条目化手册（每条 20–40 字），而 `chunk_size=200` 会把 200–400 字的手册文件切成
「主体块 + 尾巴块」：尾巴块只有 51–76 字符，语义弱、被挤出 TopK；
同时 180/200 的边界还会把一条问答从中间切断。

| 阶段 | chunk_size / overlap | 向量数 | Recall@3 | Hit@1 | MRR | 未命中 |
| --- | --- | --- | --- | --- | --- | --- |
| 起点 | 200 / 20 | 19 | 90.0% | 86.0% | 0.880 | 5 |
| 修完评测集 | 200 / 20 | 19 | 94.0% | 90.0% | 0.920 | 3 |
| 调分块 | 400 / 40 | 9 | **100.0%** | **94.0%** | **0.970** | 0 |

分层（修完评测集 + 400/40）：

| 类型 | 条数 | Recall@3 | MRR |
| --- | --- | --- | --- |
| corpus_qa（语料原句） | 30 | 100.0% | 0.967 |
| synthetic（模板改写） | 20 | 100.0% | 0.975 |

失败归因（`inspect_chunks.py` 离线复算出来的，不花一分钱）：

- 起点那 5 条里，**2 条是评测集自身的 bug**：参考答案被剥掉了行首的 `- `，
  和语料原文对不上，包含度被卡在 0.57–0.64，看起来像「召回失败」。
  修好之后这两条直接满分 —— **先修评测集，再调参数，否则是在调空气。**
- 剩下 3 条是真问题：答案没被切断，但落在 51 / 76 字符的碎片块里，被挤出了 Top-3。
  `chunk_size=400` 后三份手册各只剩 1 块，碎片块归零，3 条全部转绿。

> 注意这不是「越大越好」：块越大，一个块里塞的主题越多，embedding 越糊。
> 这里能到 400 是因为语料本身就只有 200–400 字、且是条目化的。

复现命令：

```bash
# 改 config/chroma.yml 的 chunk_size / chunk_overlap 之后：
python main.py --rebuild-kb          # 必须用 --rebuild-kb！
python evals/run_eval.py --mode retrieval
```

> ⚠️ **改了 `chunk_size` 一定要 `--rebuild-kb`。**
> 知识库是 MD5 增量更新的，文件内容没变 → `--build-kb` 会全部跳过，
> 你会以为「参数没用」，其实新参数根本没生效。

---

### 实验 2：端到端跑出来的第一个真 bug

`python evals/run_eval.py --mode agent --limit 3` 首跑：

```
工具选择=100.0%  事实命中=33.3%  通过率=33.3%
```

工具选择满分，事实全挂 —— 这个组合本身就是一条线索：**模型知道该调什么，但喂给工具的东西不对。**
加上参数维度后，两条失败用例的原因立刻摊开：

```
mh-002  tool_ok=False  call_errors=['fetch_external_data 期望 {"user_id": "001", "month": "2026-06"}，实际 {"user_id": "001"}']
mh-003  tool_ok=False  call_errors=['fetch_external_data 期望 {"user_id": "003", "month": "2026-05"}，实际 {"user_id": "003"}']
```

模型只传了 `user_id`，`month` 全部漏掉。顺着查工具实现：

```python
if not month:
    month = get_current_month.func()   # 静默默认成当前月
```

这就是个真 bug：**兜底掩盖了调用方的错误。** 用户问「上个月」，拿到的是本月的记录，
格式完全正常、数字看着也合理，只有跟基准数据比才发现不对。

修法（同时改夹具和文档，让 `month` 变成必填）：

| 改哪 | 改成什么 | 为什么 |
| --- | --- | --- |
| `agent/tools/agent_tools.py` | `month` 缺参时返回明确提示，不再静默兜底 | 让错误在调用点暴露，而不是伪装成一次成功调用 |
| `evals/golden_multihop.jsonl` | 给 12 条用例补 `expected_calls` | 让「参数漏传」可被判分，而不是只在事实维度上表现为掉分 |

> 这一段是「有评测集」和「只有 demo」的分界线：没有 `expected_calls` 时，
> 这两条失败在报告里长得和「模型总结错了」一模一样，你会去调 prompt；
> 有了它，你直接去修工具。

---


### 实验 3：跑分脚本里的悬空引用

改判分逻辑时删掉了一行 `tool_names = ...`，但前面那句
`print(f"  [{flag}] {row['id']} tools={tool_names} {elapsed}s")` 忘了改，于是：

```
NameError: name 'tool_names' is not defined
  File "evals/run_eval.py", line 404, in run_agent
```

这个错什么时候才会现形？`--mode agent` 已经连上大模型、跑完几跳、**真金白银花掉之后**。
静态检查 1 秒就能抓到它 —— 但 pyflakes / ruff 在这个环境里装不上，
所以顺手写了个零依赖的 AST 检查器 `evals/namecheck.py`（只做 pyflakes 的 F821：引用了未定义的名字）：

```bash
python evals/namecheck.py --selftest   # 先自检：该抓的抓、该放过的放过（闭包/全局/推导式/海象/except-as）
python evals/namecheck.py evals        # 扫评测脚本
python evals/namecheck.py .            # 扫整个工程（自动跳过 .venv / __pycache__ / chroma_db）
```

判据很直白：函数里每个名字引用，能不能在「本函数绑定 / 外层闭包 / 模块全局 / 内置」里找到。
它抓不到的东西同样明确：跨模块属性、`try/except ImportError` 的条件导入、装饰器和默认参数里的名字。

> 面试可讲：**改完脚本先静态自检，再跑要花钱的端到端** —— 这是「评测工程」和
> 「随手写个脚本」之间的习惯差。

---

### 实验 4：判分误伤，比漏判更值得修

全量端到端首跑：

```
工具选择=86.7%  参数正确=100.0%  事实命中=80.0%  通过率=73.3%
```

参数正确率 100% 说明 `month` 那个修复生效了（模型第一次漏传，工具把「Field required」回传，模型自己补上）。
但 80% 的事实命中里，有两条失败长这样：

```
- 缺失事实：滤网接近寿命上限
- 回答节选：...耗材状态：滤网已接近寿命上限，建议本月及时更换...
```

金标要的是 `data/external/records.csv` 里的原文「滤网接近寿命上限」，模型写的是「滤网**已**接近寿命上限」——
多插了一个字，`must_include` 的子串判定就判它没命中。**这是判分器误伤，不是模型答错。**

修法：把「允许改写」的事实从严格子串换成正则，并且**给判据本身写负控**。

| 判据 | 放什么 | 例子 |
| --- | --- | --- |
| `must_include` | 数字、专名这类**改一个字就是错**的 | `47`、`62.7`、`复式` |
| `must_match` | 允许插语气词的整句事实（正则） | `滤网[^。；\n]{0,6}接近[^。；\n]{0,6}上限` |
| `must_not_include` | 不许出现的**错误结论**（负控） | `滤网状态良好`、`滤网未接近` |

这三条判据本身就是实验对象，一共改了两版：

| 版本 | 判据 | 事实命中 | 通过率 |
| --- | --- | --- | --- |
| 首跑 | `must_include` 严格子串 | 80.0% (12/15) | 73.3% (11/15) |
| v1 | 正则只在「滤网 → 接近」之间放宽：`滤网.{0,3}接近寿命上限` | 93.3% (14/15) | 93.3% (14/15) |
| v2 | 容错放到两侧 + 补负控（定稿版） | **100%** (15/15) | **100%** (15/15) |

v1 之后只剩 mh-003 还挂着，模型这次写的是：

```
- **滤网**：已接近使用寿命上限，**建议本月内更换**
```

v1 把容错留在「滤网 → 接近」之间，而这次**插字插在「接近 → 寿命」之间**（多了「使用」）。
放宽的位置放错了地方，等于没放宽。v2 在两侧各给 6 个字容错，5 种真实改写写法全部命中。

第二个坑是 `.` 配 `re.S`：`.` 会跨行匹配，
一段正常报告里「滤网…（换行）…接近寿命上限」可能被**跨行拼出一次误命中**。
v2 用 `[^。；\n]` 代替 `.`，把「同一句、同一行」写进判据 —— 容错范围本身就是一条规格。

第三件事：正则分不清否定。「滤网未接近寿命上限，无需更换」用 v2 的正则照样能匹配上，
所以补 `must_not_include` 负控（判分器本来就支持，只是评测集里一直没人用）。
负控同样做了双向验证：6 条「该挂」的写法（状态良好 / 已更换 / 未接近 / 接近了更换周期）全部判失败，
5 条「该过」的改写全部判通过。

> 放宽判据必须配负控，否则你只是在把分数调好看。判分器放宽的是**改写容忍度**，不是结论的正确性。

> 定稿效果：拿同一批真实回答离线复算（`score_case` 是纯函数，复算等价于重跑判分），
> 15/15 全过；同一批回答用旧判据是 14/15。

---

### 实验 5：评测脚本自己把用例打断了

mh-015（报告 + 知识库联合场景）只跑了 2.05s 就结束，明细里只有一行：

```
| mh-015 | 报错 | (未调用工具) | 50.0 | - | 2.05s |
- 报错：KeyError: 'name'
```

问题不在模型，在**链路**：这条用例只调了一次模型就抛异常，工具一次都没跑到。
而报告只说 `KeyError: 'name'`，不说是哪一行 —— 这种报错等于没报。两处修：

| 改哪 | 改成什么 | 为什么 |
| --- | --- | --- |
| `evals/run_eval.py` | 异常时把 `traceback.format_exc()` 存进 JSONL，控制台和报告只打最后 3 个调用帧 | 定位靠栈，不靠猜 |
| `agent/tools/middleware.py` | `monitor_tool` 不再直接取 `request.tool_call["name"]`，缺 name 时记 error 日志并回一条错误 ToolMessage | 模型偶尔会吐出没有 name 的工具调用；一个畸形调用不该打断整轮对话 |

再加一个 `--only`，重放一条用例就不用再烧全量的钱：

```bash
python evals/run_eval.py --mode agent --only mh-015
```

重放之后再确认一次：如果控制台还在报同一个 `KeyError`，明细 JSONL 里的 `error_traceback`
会直接指出是哪一层 —— 定位到框架内部就照着那一帧继续修，不用再猜。

两个修的实测效果：

| 用例 | 修之前 | 修之后 |
| --- | --- | --- |
| mh-015（`--only mh-015` 单条重放） | 2.05s 就结束、一次工具没调、报 `KeyError: 'name'` | **通过**：18.1s，6 次工具调用（fill_context_report → get_user_id → get_current_month → fetch_external_data ×2 → rag_summarize）全部成功 |
| 全量 15 条 | 执行报错 1 条 | 执行报错 **0** 条 |

当时的推断：`logs/agent_260927.log` 里 mh-015 最后一轮停在「即将调用模型」之后、
「[工具监控]开始调用工具」之前，所以怀疑问题出在 `monitor_tool` 里那处 `tool_call["name"]` 取值。

> **这条推断后来被 traceback 证伪了 —— 见实验 6。** 真正崩的不是 `monitor_tool`，而是更下面一层：
> 上游适配器解析流式响应时硬取 `prev_function["name"]`。`monitor_tool` 那处防御**保留**，
> 它管的是「工具调用序列被截断」这种情况，是第二道防线；模型响应解析阶段的异常它抓不到。

---

### 实验 6：同一份代码，跑分从 100% 掉到 50%

上面的 100% 是 17:13 那次全量（`results/agent-20260927-171344.md`）。**代码一个字没改**，
17:23 又跑一次全量：工具选择 50.0%、参数正确 100.0%、事实命中 50.0%、通过率 50.0%
（`results/agent-20260927-172320.md`）。

同一份代码、同一批用例、同一个模型，两次差这么多 —— 这不是「模型变差了」，是**概率性故障**。
看挂掉的那几条，明细里不是判分失败，而是**整条用例直接报错**：

```
| mh-003 | 报错 | (未调用工具) | 1.74s |
- 报错：KeyError: 'name'
```

1.74s、工具一次没调 —— 和实验 5 的 mh-015 是同一个签名。区别在于这次 JSONL 里存了
**完整 traceback**（就是实验 5 加的那条改动），一眼就能读出崩在哪一层：

```
langchain_core/language_models/chat_models.py:1981  _generate_with_cache
    for chunk in self._stream(messages, stop=stop, **kwargs):
langchain_community/chat_models/tongyi.py:733  _stream
    for stream_resp, is_last_chunk in generate_with_last_element_mark(
langchain_community/chat_models/tongyi.py:578  _stream_completion_with_retry
    delta_resp = self.subtract_client_response(resp, prev_resp)
langchain_community/chat_models/tongyi.py:611  subtract_client_response
    prev_function["name"], ""
KeyError: 'name'
```

> **实验 5 的推断在这里被证伪。** 崩点根本不在 `monitor_tool`，而在 `subtract_client_response`：
> 上游适配器把流式增量块拼成完整响应时，假定每个块都带 `name` 字段；qwen3-max 偶发吐出不带
> `name` 的工具调用增量，硬取字典键就崩了。这发生在**工具中间件之前**，所以 `monitor_tool` 里
> 再多的防御也拦不住。

三条结论直接决定了修法：

| 结论 | 影响 |
| --- | --- |
| `_stream` 被 `_generate` 和 `_generate_with_cache` 共用 | 改成 `streaming=False` **逃不掉**这条路径，别指望关流式绕开 |
| 崩在「模型响应解析」阶段，早于工具中间件 | `monitor_tool` 抓不到，兜底必须做在**模型调用级** |
| `max_retries=3` 只在 HTTP 层生效 | 救不了这类解析异常，得自己在中间件层重试 |

修法：新增第 5 个中间件 `retry_model_call`（`@wrap_model_call`），挂在中间件列表**最后一位**
—— 列表越靠后越贴近模型调用，异常就在这一层被接住并重试一次：

```python
@wrap_model_call
def retry_model_call(request: ModelRequest, handler: Callable) -> Any:
    try:
        return handler(request)
    except KeyError as error:
        logger.warning(f"[模型调用]响应解析失败（{type(error).__name__}: {error}），重试一次")
        return handler(request)
```

同时 `run_eval.py` 加了**可见重试**：`call_agent(agent, question, attempts=2)` 返回
`(result, retries)`，`retries` 写进明细 JSONL，报告里新增「一次通过 / 发生重试」一栏 ——
重试不藏起来，藏起来就把不稳定性掩盖了。

> 说清楚边界：这是**兜底**，不是**根治**。根治要么给 `langchain_community` 打本地补丁改
> `subtract_client_response`（跟上游升级打架），要么换一个不这样解析的适配器。对业务项目来说，
> 在离模型最近的一层做一次重试，性价比最高，也最诚实。

效果：

| 项 | 加 `retry_model_call` 之前 | 之后 |
| --- | --- | --- |
| mh-003（`--only mh-003` 单条重放） | 1.74s、0 次工具、报 `KeyError: 'name'` | 多跳链路跑完、**通过** |
| 端到端全量 15 条 | 概率性掉到 50.0% | 稳定满分（偶发重试会在报告里显示次数） |

这条经验面试可以直接讲：**「我的评测集抓出了一个概率性、只在模型响应里偶发的适配器缺陷 ——
它平时躲在 100% 后面，你只有一个 demo 的时候永远看不见它。」**

---

## 六、把结果写进主 README

跑完 `--mode retrieval` 和 `--mode agent` 之后，把 `results/latest.md` 里的数字填进主 README。
下面这份是本次实测值（主仓库 `README.md` 第九节与之一致）：

```markdown
## 九、评测结果

评测集：检索 50 条（自动派生自语料）+ 多跳 15 条（手写，覆盖工具编排 / 参数补全 / 幻觉抑制）。
环境冻结：用户 001、城市深圳、当前月份 2026-09（保证可复现）。

| 指标 | 数值 |
| --- | --- |
| 检索 Recall@3 | 100.0% |
| 检索 Hit@1 | 94.0% |
| 检索 MRR | 0.970 |
| 工具选择准确率 | 100.0% |
| 工具参数正确率 | 100.0% |
| 关键事实命中率 | 100.0% |
| 端到端通过率 | 100.0% |

调参记录：chunk_size 200 → 400 后 Recall@3 从 90.0% 提升到 100.0%。

复现：`python evals/build_golden.py --with-manuals && python evals/run_eval.py --mode retrieval`
```

面试时这句话的分量：**「我没有只做一个 demo，我给它配了评测集，知道它现在几分、哪几条挂了、为什么挂、改哪个参数能好。」**

---

## 七、常见坑

| 现象 | 原因 / 处理 |
| --- | --- |
| `向量库是空的` | 先跑 `python main.py --rebuild-kb` |
| `缺少 DASHSCOPE_API_KEY` | 在工程根目录建 `.env`（参考 `.env.example`），或设同名系统环境变量 |
| `评测集不存在` | 先跑 `python evals/build_golden.py` |
| 改了 chunk_size 但分数没变 | 用了 `--build-kb`，MD5 没变所以整批跳过；要 `--rebuild-kb` |
| 检索指标突然掉了 | 大概率是换了 embedding 模型但没 `--rebuild-kb`，旧向量和新向量不在同一空间 |
| 报告说「金标对不上语料」 | 重新生成评测集前先确认解析逻辑没改动语料原文（尤其是行首的 `- ` 和编号） |
| `agent` 模式很慢/很贵 | 每条用例是一整轮多跳对话。先 `--limit 3` 验证链路，再全量 |
| 报告里中文变问号 | 只影响控制台显示，`results/*.md` 里始终是正确的 UTF-8 |
| 跑分中途 `NameError: name 'xxx' is not defined` | 改脚本时删了变量定义、忘了删引用。跑 `python evals/namecheck.py evals`，1 秒给出文件和行号 |
| 报告只写「报错：KeyError: 'xxx'」看不出在哪 | 明细 JSONL 里现在带完整 traceback；再用 `--only <id>` 单条重放，不用跑全量 |
| `agent` 模式整条用例报 `KeyError: 'name'`（0 次工具、1~2s 就结束） | 上游适配器拼接流式工具调用时缺 `name`。已加 `retry_model_call` 中间件兜底重试，报告里会显示重试次数 |
