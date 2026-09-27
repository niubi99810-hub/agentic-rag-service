"""Agentic RAG 项目评测脚本：检索指标 + 端到端工具/事实判分（+ 可选 LLM 评委）。

三种模式::

    python evals/run_eval.py --mode retrieval            # 只跑向量检索：Recall@k / MRR
    python evals/run_eval.py --mode agent --limit 5      # 端到端 Agent：工具编排 + 事实命中
    python evals/run_eval.py --mode judge --run evals/results/agent-xxx.jsonl

三条设计原则（面试可以展开讲）：

1. **检索与生成解耦**：召回率只用 embedding 就能算，不需要大模型，指标确定、可反复跑；
   生成环节才用大模型，判分走「必含事实 + 禁止出现」的规则，不把分数交给评委的自由裁量；
2. **可复现**：--freeze-date 固定「当前月份」、--freeze-user 固定随机用户池，
   否则 get_current_month / get_user_id 每次调用都换值，今天的分数明天就不成立；
3. **失败可见**：每条用例的完整回答、工具链路、失败原因都落盘成 JSONL，
   报告里打印失败明细，而不是只给一个总分。
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import difflib
import json
import os
import pathlib
import re
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

RETRIEVAL_SET = "evals/golden_retrieval.jsonl"
MULTIHOP_SET = "evals/golden_multihop.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

DEFAULT_FREEZE_DATE = "2026-09"
DEFAULT_FREEZE_USER = "001"
DEFAULT_FREEZE_CITY = "深圳"
CONTAINMENT_THRESHOLD = 0.6

WHITESPACE_RE = re.compile(r"[\s\u3000]+")


# --------------------------------------------------------------------------- #
# 环境准备
# --------------------------------------------------------------------------- #
def use_project_modules() -> None:
    """把工程根目录放进 sys.path，让脚本能 import utils / agent / rag。"""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def bootstrap_env() -> None:
    """加载工程根目录的 .env（不存在就跳过，直接用系统环境变量）。"""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(str(env_path))


def require_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "缺少 DASHSCOPE_API_KEY，无法调用模型。\n"
            "修复：在工程根目录新建 .env（参考 .env.example），写入：\n"
            "    DASHSCOPE_API_KEY=sk-你的百炼密钥\n"
            "或者直接设置同名系统环境变量后重新打开终端。"
        )
    return key


def force_safe_stdout() -> None:
    """中文 Windows 控制台编码可能吃不消特殊字符，退化成占位符而不是直接崩。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


# --------------------------------------------------------------------------- #
# 评测集读取
# --------------------------------------------------------------------------- #
def load_jsonl(relative_path: str) -> list[dict]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise SystemExit(f"评测集不存在：{path}\n先生成：python evals/build_golden.py")
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    if not cases:
        raise SystemExit(f"评测集是空的：{path}")
    return cases


def newest_run() -> str:
    if not RESULTS_DIR.exists():
        raise SystemExit("还没有任何运行记录，先跑 --mode agent。")
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit("还没有任何运行记录，先跑 --mode agent。")
    return str(files[-1].relative_to(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# 文本比对与判分
# --------------------------------------------------------------------------- #
def squeeze(text: str) -> str:
    """去掉所有空白（含全角空格），避免因为排版差异误判。"""
    return WHITESPACE_RE.sub("", text or "")


def containment(reference: str, text: str) -> float:
    """参考答案有多大比例连续出现在 text 里（0~1）。

    用「最长公共子串长度 / 参考答案长度」而不是编辑距离：
    分块可能把一段答案从中间切断，这个口径对「被切一半」最宽容也最好解释。
    """
    reference, text = squeeze(reference), squeeze(text)
    if not reference or not text:
        return 0.0
    if reference in text:
        return 1.0
    matcher = difflib.SequenceMatcher(None, reference, text, autojunk=False)
    block = matcher.find_longest_match(0, len(reference), 0, len(text))
    return block.size / len(reference)


def check_expected_calls(tool_calls: list[dict], expected_calls: list[dict]) -> list[str]:
    """参数级校验：工具「调对了」但参数传错，是完全不同的一类失败。

    expected_calls 里每一条都要求「有一次实际调用：同名，且参数是它的超集」。
    比如 month 被漏传、被静默默认成当前月，都会在这里被指名道姓地抓出来；
    否则报告里只会看到一个「事实没命中」，还得自己反推原因。

    工具压根没被调用时这里不报错：那是 expected_tools / missing_tools 的职责，
    两边都报只会让同一条用例在报告里出现两遍，反而看不清。
    """
    errors: list[str] = []
    for wanted in expected_calls or []:
        name = wanted.get("name", "")
        args = wanted.get("args") or {}
        candidates = [call.get("args") or {} for call in tool_calls if call.get("name") == name]
        if not candidates:
            continue
        matched = any(
            all(str(candidate.get(key, "")) == str(value) for key, value in args.items())
            for candidate in candidates
        )
        if matched:
            continue
        actual = "；".join(
            json.dumps({key: candidate.get(key) for key in args}, ensure_ascii=False)
            for candidate in candidates
        )
        errors.append(f"{name} 期望 {json.dumps(args, ensure_ascii=False)}，实际 {actual}")
    return errors


def score_case(
    case: dict,
    answer: str,
    tool_calls: list[dict],
    threshold: float = CONTAINMENT_THRESHOLD,
) -> dict:
    """三维度判分：工具选对没有 + 参数传对没有 + 关键事实命中没有。"""
    answer_text = answer or ""
    squeezed = squeeze(answer_text)
    tool_names = [call.get("name", "") for call in tool_calls]

    missing_tools = [name for name in case.get("expected_tools", []) if name not in tool_names]
    forbidden_hit = [name for name in case.get("forbidden_tools", []) if name in tool_names]
    call_errors = check_expected_calls(tool_calls, case.get("expected_calls", []))

    missing_facts = [item for item in case.get("must_include", []) if squeeze(item) not in squeezed]
    missing_patterns = [pat for pat in case.get("must_match", []) if not re.search(pat, answer_text, re.S)]
    leaked_facts = [item for item in case.get("must_not_include", []) if squeeze(item) in squeezed]

    tool_ok = not missing_tools and not forbidden_hit and not call_errors
    fact_ok = not missing_facts and not missing_patterns and not leaked_facts

    return {
        "tool_ok": tool_ok,
        "fact_ok": fact_ok,
        "passed": bool(answer_text.strip()) and tool_ok and fact_ok,
        "missing_tools": missing_tools,
        "forbidden_tools_hit": forbidden_hit,
        "call_errors": call_errors,
        "missing_facts": missing_facts,
        "missing_patterns": missing_patterns,
        "leaked_facts": leaked_facts,
        "_threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# 模式一：检索指标
# --------------------------------------------------------------------------- #
def run_retrieval(cases: list[dict], k: int, limit: int, threshold: float) -> list[dict]:
    use_project_modules()
    from rag.vector_store import VectorStoreService

    service = VectorStoreService()
    vector_count = service.count()
    if vector_count <= 0:
        raise SystemExit("向量库是空的。\n修复：先构建知识库 -> python main.py --build-kb")

    selected = cases[:limit] if limit else cases
    print(f"[检索评测] 向量库 {vector_count} 个片段 | TopK={k} | 用例 {len(selected)} 条")

    retriever = service.get_retriever(k=k)
    rows: list[dict] = []
    for index, case in enumerate(selected, start=1):
        docs = retriever.invoke(case["question"])
        scores = [containment(case["reference_answer"], doc.page_content) for doc in docs]
        hits = [rank for rank, value in enumerate(scores, start=1) if value >= threshold]
        first_hit = min(hits) if hits else None

        # 失败必须分类：答案是被切碎在相邻几块里，还是压根没召回到？
        # 两种原因长得像，修法却完全相反，混在一起就没法行动。
        joined = containment(
            case["reference_answer"], "\n".join(doc.page_content for doc in docs)
        )
        if first_hit:
            fail_reason = ""
        elif joined >= threshold:
            fail_reason = "split"
        else:
            fail_reason = "missing"

        previews = [
            {"score": round(score, 3), "text": doc.page_content[:120].replace("\n", " ").strip()}
            for score, doc in zip(scores, docs)
        ]
        rows.append(
            {
                "id": case.get("id", f"ret-{index:03d}"),
                "kind": case.get("kind", "unknown"),
                "section": case.get("section", ""),
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "first_hit": first_hit,
                "hit_at_k": first_hit is not None,
                "joined_hit": joined >= threshold,
                "best_score": round(max(scores), 3) if scores else 0.0,
                "joined_score": round(joined, 3),
                "top_scores": [round(value, 3) for value in scores],
                "top_previews": [] if first_hit else previews,
                "fail_reason": fail_reason,
            }
        )
        flag = "OK  " if first_hit else ("切碎" if fail_reason == "split" else "漏召")
        print(f"  [{flag}] {rows[-1]['id']} rank={first_hit}  {case['question'][:34]}")
        if not first_hit:
            # 失败时必须能看到「到底检索回了什么」，否则只能靠猜
            print(f"         参考答案：{case['reference_answer'][:60]}")
            for rank, item in enumerate(previews, start=1):
                print(f"         #{rank} ({item['score']:.2f}) {item['text'][:60]}")
    return rows



def aggregate_retrieval(rows: list[dict], k: int) -> dict:
    total = len(rows)
    if not total:
        return {}
    summary = {
        "total": total,
        "k": k,
        "top1": sum(1 for row in rows if row["first_hit"] == 1) / total,
        "recall": sum(1 for row in rows if row["hit_at_k"]) / total,
        "recall_joined": sum(1 for row in rows if row.get("joined_hit")) / total,
        "mrr": sum(1.0 / row["first_hit"] for row in rows if row["first_hit"]) / total,
        "split_cases": sum(1 for row in rows if row.get("fail_reason") == "split"),
        "missing_cases": sum(1 for row in rows if row.get("fail_reason") == "missing"),
        "by_kind": {},
    }
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        summary["by_kind"][kind] = {
            "total": len(subset),
            "recall": sum(1 for row in subset if row["hit_at_k"]) / len(subset),
            "mrr": sum(1.0 / row["first_hit"] for row in subset if row["first_hit"]) / len(subset),
        }
    return summary




# --------------------------------------------------------------------------- #
# 模式二：端到端 Agent
# --------------------------------------------------------------------------- #
def freeze_environment(user_id: str, city: str, freeze_date: str) -> None:
    """冻结评测环境里的随机与时间来源，保证同一条命令永远得到同一份分数。

    项目里 get_user_id / get_user_location 是随机取值的（演示用），
    get_current_month 又跟着系统时间走，不冻住就没法比较两次运行。
    """
    import agent.tools.agent_tools as tools

    if user_id:
        tools.USER_ID_POOL = [user_id]
    if city:
        tools.CITY_POOL = [city]

    if freeze_date:
        year, month = (int(part) for part in freeze_date.split("-"))
        original = tools.datetime

        class FrozenDatetime(original):  # type: ignore[misc, valid-type]
            @classmethod
            def now(cls, tz=None):  # noqa: ANN001, ANN206
                return cls(year, month, 15, 10, 30, 0)

        tools.datetime = FrozenDatetime


def extract_tool_calls(messages) -> list[dict]:
    """从消息历史里还原工具调用链路（比挂 callback 更稳，不依赖框架内部钩子）。"""
    calls: list[dict] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict):
                calls.append({"name": call.get("name", ""), "args": call.get("args")})
            else:
                calls.append({"name": getattr(call, "name", ""), "args": getattr(call, "args", None)})
    return calls


def extract_tool_outputs(messages) -> list[dict]:
    """收集工具真实返回值，供 LLM 评委做「是否有幻觉」的对照。"""
    outputs: list[dict] = []
    for message in messages:
        if message.__class__.__name__ == "ToolMessage":
            outputs.append(
                {
                    "name": getattr(message, "name", "") or "",
                    "content": str(getattr(message, "content", ""))[:600],
                }
            )
    return outputs


def run_agent(
    cases: list[dict],
    limit: int,
    freeze_user: str,
    freeze_city: str,
    freeze_date: str,
) -> list[dict]:
    use_project_modules()
    require_api_key()
    freeze_environment(freeze_user, freeze_city, freeze_date)

    from agent.react_agent import ReactAgent

    agent = ReactAgent()
    selected = cases[:limit] if limit else cases
    print(
        f"[端到端评测] 用例 {len(selected)} 条 | 冻结 用户={freeze_user} "
        f"城市={freeze_city} 月份={freeze_date or '真实时间'}"
    )

    rows: list[dict] = []
    for index, case in enumerate(selected, start=1):
        started = time.perf_counter()
        answer, error, tool_calls, tool_outputs = "", "", [], []
        try:
            # 走底层图调用而不是 execute()，是为了拿到完整的消息历史（工具链路）
            result = agent.agent.invoke(
                {"messages": [{"role": "user", "content": case["question"]}]},
                context=agent._new_context(),
            )
            answer = ReactAgent._final_content(result["messages"])
            tool_calls = extract_tool_calls(result["messages"])
            tool_outputs = extract_tool_outputs(result["messages"])
        except Exception as exc:  # noqa: BLE001 - 评测脚本要把失败也记下来而不是中断
            error = f"{type(exc).__name__}: {exc}"

        elapsed = round(time.perf_counter() - started, 2)
        verdict = score_case(case, answer, tool_calls)

        row = {
            "id": case.get("id", f"case-{index:03d}"),
            "question": case["question"],
            "answer": answer,
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs,
            "elapsed_s": elapsed,
            "error": error,
            **verdict,
        }
        rows.append(row)

        flag = "PASS" if row["passed"] else ("ERR " if error else "FAIL")
        tool_names = [call.get("name", "") for call in tool_calls]
        print(f"  [{flag}] {row['id']} tools={tool_names} {elapsed}s")
        if not row["passed"]:
            print(
                f"         缺工具={verdict['missing_tools']} "
                f"多调用={verdict['forbidden_tools_hit']} "
                f"缺事实={verdict['missing_facts']} "
                f"缺句式={verdict['missing_patterns']}"
            )
            for item in verdict["call_errors"]:
                print(f"         参数错误：{item}")
            if error:
                print(f"         错误：{error[:160]}")
    return rows


# --------------------------------------------------------------------------- #
# 模式三：LLM 评委（可选，用来抓规则判分抓不到的幻觉）
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = (
    "你是严格的答案质检员。判断下面的「Agent 回答」是否忠实于「工具返回的事实」，"
    "不要评价语气、排版和长度。\n\n"
    "用户问题：{question}\n\n"
    "工具返回的事实：\n{facts}\n\n"
    "Agent 回答：\n{answer}\n\n"
    '只输出一行 JSON，不要任何多余内容：'
    '{{"faithful": 0或1, "score": 1到5的整数, "reason": "20字以内的中文理由"}}'
)


def build_facts(row: dict) -> str:
    outputs = row.get("tool_outputs") or []
    if not outputs:
        calls = row.get("tool_calls") or []
        if not calls:
            return "（本轮没有调用任何工具）"
        return "\n".join(
            f"- {call['name']}({json.dumps(call.get('args'), ensure_ascii=False)})：无返回内容"
            for call in calls
        )
    return "\n".join(f"- {item['name']} 返回：{item['content']}" for item in outputs)


def parse_judge_json(text: str) -> dict:
    matched = re.search(r"\{.*\}", text or "", re.S)
    if not matched:
        return {"faithful": None, "score": None, "reason": "评委输出无法解析"}
    try:
        data = json.loads(matched.group(0))
    except json.JSONDecodeError:
        return {"faithful": None, "score": None, "reason": "评委输出不是合法 JSON"}
    return {
        "faithful": data.get("faithful"),
        "score": data.get("score"),
        "reason": str(data.get("reason", ""))[:60],
    }


def run_judge(rows: list[dict]) -> list[dict]:
    use_project_modules()
    require_api_key()
    from model.factory import get_chat_model

    model = get_chat_model()
    print(f"[LLM 评委] 逐条核对 {len(rows)} 条回答是否忠实于工具返回的事实")

    for row in rows:
        prompt = JUDGE_PROMPT.format(
            question=row["question"],
            facts=build_facts(row),
            answer=(row.get("answer") or "")[:2000],
        )
        try:
            raw = model.invoke(prompt)
            text = getattr(raw, "content", str(raw))
            verdict = parse_judge_json(text if isinstance(text, str) else str(text))
        except Exception as exc:  # noqa: BLE001
            verdict = {"faithful": None, "score": None, "reason": f"评委调用失败：{exc}"[:60]}
        row["judge"] = verdict
        print(f"  [{verdict['faithful']}] {row['id']} {verdict['reason']}")
    return rows


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def now_text() -> str:
    return _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def render_retrieval_report(rows: list[dict], summary: dict, args) -> str:
    misses = [row for row in rows if not row["hit_at_k"]]
    split_cases = [row for row in misses if row.get("fail_reason") == "split"]
    missing_cases = [row for row in misses if row.get("fail_reason") == "missing"]

    lines = [
        "# 检索评测报告",
        "",
        f"- 运行时间：{now_text()}",
        f"- 用例数：{summary.get('total', 0)}",
        f"- TopK：{summary.get('k')}",
        f"- 命中阈值：单块包含度 >= {args.threshold}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| Recall@{summary.get('k')} | {summary.get('recall', 0):.1%} | 前 K 块里有一块**完整**包含答案 |",
        f"| 拼接 Recall@{summary.get('k')} | {summary.get('recall_joined', 0):.1%} | 前 K 块拼接后包含答案（含被切碎） |",
        f"| Hit@1 | {summary.get('top1', 0):.1%} | 第 1 块就完整包含答案 |",
        f"| MRR | {summary.get('mrr', 0):.3f} | 首次命中的平均倒数排名 |",
        "",
        "## 失败原因拆分",
        "",
        "| 原因 | 条数 | 该往哪修 |",
        "| --- | --- | --- |",
        f"| 答案被切碎（拼接后才完整） | {len(split_cases)} | 调大 chunk_size / 按语义单元切分 |",
        f"| 压根没召回到 | {len(missing_cases)} | 调 embedding / 加 rerank / 补 query 改写 |",
        "",
        "## 分层指标（按问句来源）",
        "",
        "| 类型 | 条数 | Recall | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for kind, item in sorted(summary.get("by_kind", {}).items()):
        lines.append(f"| {kind} | {item['total']} | {item['recall']:.1%} | {item['mrr']:.3f} |")

    lines += ["", f"## 未命中用例（{len(misses)} 条）", ""]
    if not misses:
        lines.append("无。")
    else:
        lines += [
            "| id | 章节 | 问题 | 最佳单块 | 拼接 | 原因 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in misses:
            reason = "被切碎" if row.get("fail_reason") == "split" else "没召回"
            lines.append(
                f"| {row['id']} | {row['section']} | {row['question'][:30]} "
                f"| {row.get('best_score', 0):.2f} | {row.get('joined_score', 0):.2f} | {reason} |"
            )
        lines += ["", "### 这些用例实际检索到了什么", ""]
        for row in misses:
            lines.append(f"**{row['id']}** {row['question']}")
            lines.append("")
            lines.append(f"- 参考答案：{row['reference_answer'][:70]}")
            for rank, item in enumerate(row.get("top_previews", []), start=1):
                lines.append(f"- Top{rank}（{item['score']:.2f}）：{item['text'][:70]}")
            lines.append("")
    return "\n".join(lines) + "\n"



def render_agent_report(rows: list[dict], args, judged: bool) -> str:
    total = len(rows)
    name_ok = sum(
        1 for row in rows if not row.get("missing_tools") and not row.get("forbidden_tools_hit")
    )
    arg_ok = sum(1 for row in rows if not (row.get("call_errors") or []))
    fact_ok = sum(1 for row in rows if row["fact_ok"])
    passed = sum(1 for row in rows if row["passed"])
    errors = sum(1 for row in rows if row.get("error"))
    avg_elapsed = sum(row.get("elapsed_s", 0) for row in rows) / total if total else 0.0

    lines = [
        "# 端到端评测报告（Agent）",
        "",
        f"- 运行时间：{now_text()}",
        f"- 用例数：{total}",
        f"- 冻结环境：用户={args.freeze_user} 城市={args.freeze_city} "
        f"月份={args.freeze_date or '真实时间'}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 工具选择准确率 | {ratio(name_ok, total)} |",
        f"| 工具参数正确率 | {ratio(arg_ok, total)} |",
        f"| 关键事实命中率 | {ratio(fact_ok, total)} |",
        f"| 整体通过率 | {ratio(passed, total)} |",
        f"| 执行报错条数 | {errors} |",
        f"| 平均耗时 | {avg_elapsed:.1f}s |",
    ]

    if judged:
        scored = [row["judge"] for row in rows if row.get("judge")]
        faithful = [item for item in scored if item.get("faithful") == 1]
        scores = [item["score"] for item in scored if isinstance(item.get("score"), (int, float))]
        lines += [
            "",
            "## LLM 评委",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 忠实率（faithful=1） | {ratio(len(faithful), len(scored))} |",
        ]
        if scores:
            lines.append(f"| 平均分（1-5） | {sum(scores) / len(scores):.2f} |")

    lines += [
        "",
        "## 用例明细",
        "",
        "| id | 结果 | 工具链路 | 缺失事实 | 参数错误 | 耗时 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.get("error"):
            flag = "报错"
        elif row["passed"]:
            flag = "通过"
        else:
            flag = "失败"
        chain = " -> ".join(call["name"] for call in row.get("tool_calls", [])) or "(未调用工具)"
        missing = ", ".join(row["missing_facts"] + row["missing_patterns"]) or "-"
        arg_bad = "; ".join(row.get("call_errors") or []) or "-"
        lines.append(
            f"| {row['id']} | {flag} | {chain} | {missing} | {arg_bad} | {row.get('elapsed_s', 0)}s |"
        )

    failures = [row for row in rows if not row["passed"]]
    lines += ["", f"## 失败明细（{len(failures)} 条）", ""]
    if not failures:
        lines.append("无。")
    else:
        for row in failures:
            lines.append(f"### {row['id']} {row['question']}")
            lines.append("")
            lines.append(f"- 工具链路：{', '.join(call['name'] for call in row.get('tool_calls', [])) or '无'}")
            if row.get("missing_tools"):
                lines.append(f"- 漏调用：{', '.join(row['missing_tools'])}")
            if row.get("forbidden_tools_hit"):
                lines.append(f"- 误调用：{', '.join(row['forbidden_tools_hit'])}")
            for item in row.get("call_errors") or []:
                lines.append(f"- 参数错误：{item}")
            if row.get("missing_facts"):
                lines.append(f"- 缺失事实：{', '.join(row['missing_facts'])}")
            if row.get("missing_patterns"):
                lines.append(f"- 未满足句式：{', '.join(row['missing_patterns'])}")
            if row.get("error"):
                lines.append(f"- 报错：{row['error'][:200]}")
            excerpt = (row.get("answer") or "").replace("\n", " ")[:160]
            lines.append(f"- 回答节选：{excerpt or '(空)'}")
            lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def save(rows: list[dict], markdown: str, mode: str) -> tuple[pathlib.Path, pathlib.Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = RESULTS_DIR / f"{mode}-{stamp}.jsonl"
    md_path = RESULTS_DIR / f"{mode}-{stamp}.md"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown, encoding="utf-8")
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")
    return jsonl_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic RAG 评测：检索指标 / 端到端 / LLM 评委")
    parser.add_argument("--mode", choices=("retrieval", "agent", "judge"), default="retrieval")
    parser.add_argument("--set", dest="case_set", default=None, help="评测集路径，默认按模式选择")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 = 全部")
    parser.add_argument("--k", type=int, default=3, help="检索 TopK")
    parser.add_argument("--threshold", type=float, default=CONTAINMENT_THRESHOLD, help="命中判定阈值")
    parser.add_argument("--freeze-date", default=DEFAULT_FREEZE_DATE, help="冻结的当前月份 YYYY-MM，留空用真实时间")
    parser.add_argument("--freeze-user", default=DEFAULT_FREEZE_USER, help="冻结的用户池")
    parser.add_argument("--freeze-city", default=DEFAULT_FREEZE_CITY, help="冻结的城市池")
    parser.add_argument("--run", default=None, help="judge 模式要复评的历史运行文件")
    parser.add_argument("--judge", action="store_true", help="agent 模式跑完追加 LLM 评委")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    force_safe_stdout()
    bootstrap_env()
    args = parse_args(argv)  # 放在校验密钥之前，--help 才能在没有密钥时也能看
    require_api_key()  # 三种模式都要用 DashScope：embedding 也算调用

    if args.mode == "retrieval":
        rows = run_retrieval(load_jsonl(args.case_set or RETRIEVAL_SET), args.k, args.limit, args.threshold)
        summary = aggregate_retrieval(rows, args.k)
        markdown = render_retrieval_report(rows, summary, args)
        headline = (
            f"Recall@{args.k}={summary.get('recall', 0):.1%} "
            f"Hit@1={summary.get('top1', 0):.1%} "
            f"MRR={summary.get('mrr', 0):.3f}"
        )
    elif args.mode == "agent":
        rows = run_agent(
            load_jsonl(args.case_set or MULTIHOP_SET),
            args.limit,
            args.freeze_user,
            args.freeze_city,
            args.freeze_date,
        )
        if args.judge:
            rows = run_judge(rows)
        total = len(rows)
        markdown = render_agent_report(rows, args, judged=args.judge)
        headline = (
            f"工具选择={sum(1 for r in rows if not r['missing_tools'] and not r['forbidden_tools_hit']) / total:.1%} "
            f"参数正确={sum(1 for r in rows if not (r.get('call_errors') or [])) / total:.1%} "
            f"事实命中={sum(1 for r in rows if r['fact_ok']) / total:.1%} "
            f"通过率={sum(1 for r in rows if r['passed']) / total:.1%}"
        )
    else:
        rows = run_judge(load_jsonl(args.run or newest_run()))
        markdown = render_agent_report(rows, args, judged=True)
        headline = "评委复评完成"

    jsonl_path, md_path = save(rows, markdown, args.mode)
    print("")
    print(markdown)
    print(f"结果：{headline}")
    print(f"明细：{jsonl_path}")
    print(f"报告：{md_path}")
    print(f"最新：{RESULTS_DIR / 'latest.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
