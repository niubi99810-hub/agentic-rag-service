"""离线自测脚本：不依赖 API Key、不产生网络调用。

运行：python tests/test_offline.py

覆盖内容：
1. 配置读取 / 路径工具 / 提示词加载；
2. 文件 MD5 计算与目录扫描；
3. 外部 CSV 业务数据解析（含 fetch_external_data 返回纯字符串契约）；
4. Agent 图构建；
5. 中间件：工具失败不中断整体任务；
6. 中间件：工具触发后动态提示词切换生效；
7. 中间件：修正模型返回的非法 function.arguments（避免回传接口 400）；
8. 中间件：模型响应解析抛 KeyError 时自动重试一次，不打断整轮对话。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from agent.react_agent import ReactAgent  # noqa: E402
from agent.tools.agent_tools import fetch_external_data  # noqa: E402
from agent.tools.middleware import (  # noqa: E402
    _normalize_tool_call_arguments,
    monitor_tool,
    retry_model_call,
)
from utils.config_handler import agent_conf, chroma_conf, rag_conf  # noqa: E402
from utils.file_handler import get_file_md5, listdir_with_allowed_type  # noqa: E402
from utils.path_tool import get_abs_path, get_project_root  # noqa: E402
from utils.prompt_loader import load_report_prompt, load_system_prompt  # noqa: E402

SEEN_SYSTEM_PROMPTS: list[str] = []


class OfflineFakeChatModel(GenericFakeChatModel):
    """离线假模型：支持 bind_tools，并记录每次调用时收到的系统提示词。"""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        for message in messages:
            if getattr(message, "type", "") == "system":
                SEEN_SYSTEM_PROMPTS.append(str(message.content))
                break
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# 记录「故意抽风的假模型」被调用了几次（用模块级变量，避免 Agent 内部复制模型实例后读不到）
FLAKY_MODEL_CALLS: list[int] = []


class FlakyFirstCallChatModel(OfflineFakeChatModel):
    """首次调用抛 KeyError('name')，第二次正常返回。

    复现的是上游 tongyi 适配器 ``subtract_client_response`` 里
    ``prev_function["name"]`` 硬取字典键、模型偶发不返回 name 时的崩溃现场。
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        FLAKY_MODEL_CALLS.append(1)
        if len(FLAKY_MODEL_CALLS) == 1:
            raise KeyError("name")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@tool(description="必然失败的工具，用于验证中间件容错")
def always_fail() -> str:
    raise ValueError("模拟工具异常")


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed.append(name)
            print(f"  [FAIL] {name} {detail}")

    def report(self) -> int:
        total = self.passed + len(self.failed)
        print(f"\n{'=' * 60}")
        print(f"自测结果：{self.passed}/{total} 通过")
        if self.failed:
            for name in self.failed:
                print(f"  - 失败项：{name}")
            return 1
        print("全部通过")
        return 0


def test_config_and_utils(checker: Checker) -> None:
    print("\n[1] 配置 / 路径 / 提示词")
    checker.check("工程根目录存在", os.path.isdir(get_project_root()))
    checker.check("config/rag.yml 读取成功", rag_conf["chat_model_name"] == "qwen3-max")
    checker.check("config/chroma.yml 读取成功", chroma_conf["collection_name"] == "agent")
    checker.check("文本分块参数生效", chroma_conf["chunk_size"] == 400 and chroma_conf["chunk_overlap"] == 40)
    checker.check("中文分隔符未被破坏", "。" in chroma_conf["separators"])
    checker.check("外部数据路径配置存在", bool(agent_conf["external_data_path"]))
    checker.check("系统提示词非空", len(load_system_prompt()) > 100)
    checker.check("报告提示词非空", len(load_report_prompt()) > 100)


def test_file_handler(checker: Checker) -> None:
    print("\n[2] 文档加载 / MD5")
    data_dir = get_abs_path("data")
    files = listdir_with_allowed_type(data_dir, (".txt", ".pdf"))
    checker.check("扫描到知识库源文件", len(files) >= 4, f"实际 {len(files)} 个")

    md5_value = get_file_md5(files[0]) if files else None
    checker.check("MD5 计算返回 32 位摘要", isinstance(md5_value, str) and len(md5_value) == 32)
    checker.check("不存在的文件返回 None", get_file_md5(get_abs_path("data/not_exist.txt")) is None)

    from utils.file_handler import load_document

    documents = load_document(files[0]) if files else []
    checker.check("txt 文档解析出内容", bool(documents) and len(documents[0].page_content) > 0)
    # .csv 不在 allow_knowledge_file_type 中，应被优雅拒绝（返回空列表而不是抛异常）
    checker.check("不支持的文件类型返回空列表", load_document(get_abs_path("data/external/records.csv")) == [])


def test_external_data(checker: Checker) -> None:
    print("\n[3] 外部 CSV 业务数据")
    record = fetch_external_data.invoke({"user_id": "001", "month": "2025-06"})
    checker.check("命中记录且返回纯字符串", isinstance(record, str) and len(record) > 0, repr(record)[:80])
    checker.check("记录包含清洁效率字段", "清洁效率" in record, record[:60])

    missing = fetch_external_data.invoke({"user_id": "999", "month": "1999-01"})
    checker.check("未命中时返回空字符串", missing == "", repr(missing))

    # 配置中的全部演示用户都应能查到数据
    users = agent_conf["user_id_pool"]
    all_hit = all(
        fetch_external_data.invoke({"user_id": uid, "month": "2025-06"}) for uid in users
    )
    checker.check("演示用户池数据完整", all_hit)


def test_agent_graph(checker: Checker) -> None:
    print("\n[4] Agent 图构建 / 中间件")
    SEEN_SYSTEM_PROMPTS.clear()
    model = OfflineFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "fill_context_report", "args": {}, "id": "call_1"}]),
                AIMessage(content="已生成使用报告"),
            ]
        )
    )
    agent = ReactAgent(model=model)
    checker.check("Agent 构建成功", agent.agent is not None)
    checker.check("注册工具数量为 7", len(agent.tools) == 7, f"实际 {len(agent.tools)}")
    checker.check("挂载中间件数量为 5", len(agent.middleware) == 5, f"实际 {len(agent.middleware)}")

    result = agent.execute("生成我的本月使用报告")
    checker.check("非流式执行返回回答", result == "已生成使用报告", repr(result))

    checker.check("首次调用使用系统提示词", bool(SEEN_SYSTEM_PROMPTS) and SEEN_SYSTEM_PROMPTS[0] == load_system_prompt())
    checker.check(
        "工具触发后切换为报告提示词",
        len(SEEN_SYSTEM_PROMPTS) >= 2 and SEEN_SYSTEM_PROMPTS[-1] == load_report_prompt(),
        f"记录到 {len(SEEN_SYSTEM_PROMPTS)} 次模型调用",
    )


def test_tool_failure_tolerance(checker: Checker) -> None:
    print("\n[5] 工具失败容错（下方 ERROR 堆栈属于预期日志，用于验证异常已被捕获并记录）")
    model = OfflineFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "always_fail", "args": {}, "id": "call_err"}]),
                AIMessage(content="工具失败了，但我仍然可以回答用户"),
            ]
        )
    )
    agent = create_agent(
        model=model,
        tools=[always_fail],
        middleware=[monitor_tool],
        system_prompt="测试用系统提示词",
    )

    try:
        result = agent.invoke({"messages": [HumanMessage("随便问点东西")]})
        texts = [str(m.content) for m in result["messages"]]
        checker.check("工具异常未中断整体任务", True)
        checker.check("最终仍返回模型回答", "工具失败了，但我仍然可以回答用户" in texts)
        checker.check(
            "异常信息以 ToolMessage 回传模型",
            any("执行失败" in text for text in texts),
        )
    except Exception as e:  # pragma: no cover
        checker.check("工具异常未中断整体任务", False, f"抛出异常 {type(e).__name__}: {e}")


def test_tool_call_argument_normalization(checker: Checker) -> None:
    print("\n[6] 工具调用参数规范化")
    broken = AIMessage(
        content="",
        tool_calls=[{"name": "fetch_external_data", "args": {"user_id": "001"}, "id": "call_bad"}],
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "fetch_external_data",
                        # 模拟流式拼接出的非法 JSON
                        "arguments": '{"user_id": "001", month: 2026-09"}',
                    },
                }
            ]
        },
    )
    fixed = _normalize_tool_call_arguments(broken)
    checker.check("识别出非法 function.arguments 并修正", fixed is not None)

    if fixed is not None:
        raw_arguments = fixed.additional_kwargs["tool_calls"][0]["function"]["arguments"]
        try:
            parsed = json.loads(raw_arguments)
            is_valid = True
        except ValueError:
            parsed = {}
            is_valid = False
        checker.check("修正后为合法 JSON", is_valid, raw_arguments)
        checker.check("修正后与解析出的 args 一致", parsed == {"user_id": "001"}, raw_arguments)

    legal = AIMessage(
        content="",
        tool_calls=[{"name": "get_user_id", "args": {}, "id": "call_ok"}],
        additional_kwargs={
            "tool_calls": [
                {"index": 0, "id": "call_ok", "type": "function", "function": {"name": "get_user_id", "arguments": "{}"}}
            ]
        },
    )
    checker.check("合法参数不会被改动", _normalize_tool_call_arguments(legal) is None)


def test_model_call_retry(checker: Checker) -> None:
    print("\n[7] 模型调用兜底重试（上游适配器解析畸形响应）")
    FLAKY_MODEL_CALLS.clear()
    model = FlakyFirstCallChatModel(messages=iter([AIMessage(content="重试之后拿到了回答")]))
    agent = create_agent(
        model=model,
        tools=[always_fail],
        middleware=[retry_model_call],
        system_prompt="测试用系统提示词",
    )

    try:
        result = agent.invoke({"messages": [HumanMessage("随便问点东西")]})
        texts = [str(message.content) for message in result["messages"]]
        checker.check("模型解析抛 KeyError 未打断整轮对话", True)
        checker.check("重试后拿到模型回答", "重试之后拿到了回答" in texts)
        checker.check("确实重试了模型调用", len(FLAKY_MODEL_CALLS) == 2, f"实际调用 {len(FLAKY_MODEL_CALLS)} 次")
    except Exception as e:  # pragma: no cover
        checker.check("模型解析抛 KeyError 未打断整轮对话", False, f"抛出异常 {type(e).__name__}: {e}")
        checker.check("重试后拿到模型回答", False)
        checker.check("确实重试了模型调用", False)


def main() -> int:
    checker = Checker()
    test_config_and_utils(checker)
    test_file_handler(checker)
    test_external_data(checker)
    test_agent_graph(checker)
    test_tool_failure_tolerance(checker)
    test_tool_call_argument_normalization(checker)
    test_model_call_retry(checker)
    return checker.report()


if __name__ == "__main__":
    sys.exit(main())