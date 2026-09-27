"""Agent 中间件模块。

五个自定义中间件分别解决五件事：
1. ``monitor_tool``            —— 工具调用监控：记录入参 / 耗时 / 结果，失败不中断整体任务；
2. ``normalize_tool_calls``    —— 模型调用前修正历史消息里不合法的 ``function.arguments``；
3. ``log_before_model``        —— 模型调用埋点：记录每次进入模型前的上下文规模；
4. ``report_prompt_switch``    —— 动态提示词切换：报告场景自动切换到报告提示词；
5. ``retry_model_call``        —— 模型调用兜底重试：上游适配器解析畸形响应抛 KeyError 时重试一次。

其中 ``monitor_tool`` 在检测到 ``fill_context_report`` 被调用后，
会向运行时上下文写入 ``report=True``，``report_prompt_switch`` 据此切换提示词，
这就是「工具运行状态驱动动态 Prompt 切换」的完整闭环。

``normalize_tool_calls`` 解决的是一个很典型的工程问题：部分模型（如 qwen3-max）在
流式返回工具调用时，``function.arguments`` 可能拼接出非法 JSON；LangChain 仍能解析出
可用的 args 并执行工具，但把这段原始字符串回传给 DashScope 时会被拒绝并报
``400 The "function.arguments" parameter ... must be in JSON format``，导致整轮对话中断。
这里统一以解析后的 args 为准重新序列化，保证回传的历史消息始终是合法 JSON。
"""
import json
import time
from typing import Any, Callable, Optional, Union

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompt, load_system_prompt

# 运行时上下文中「报告场景」的标记位
REPORT_CONTEXT_FLAG = "report"
# 触发提示词切换的工具名（需与 agent_tools.py 中注册的工具名保持一致）
REPORT_SWITCH_TOOL = "fill_context_report"


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]],
) -> Union[ToolMessage, Command]:
    """工具调用监控中间件。"""
    tool_call = request.tool_call or {}
    tool_name = tool_call.get("name") or ""
    tool_args = tool_call.get("args")
    tool_call_id = tool_call.get("id") or ""

    if not tool_name:
        # 模型偶发会返回没有 name 的工具调用（流式拼接截断等）。
        # 直接取 ["name"] 会抛 KeyError 把整轮对话打断，这里降级成一条错误回执，
        # 让模型自己重发一次合法调用，而不是让整条用例挂掉。
        logger.error(f"[工具监控]收到缺少 name 的工具调用，已拒绝执行：{tool_call}")
        return ToolMessage(
            content="工具调用缺少 name 字段，无法执行。请重新发起一次完整的工具调用。",
            tool_call_id=tool_call_id,
            status="error",
        )

    start_time = time.perf_counter()

    logger.info(f"[工具监控]开始调用工具：{tool_name}，入参：{tool_args}")

    try:
        result = handler(request)
    except Exception as e:
        # 单个工具失败不中断整体任务：把错误信息回传给模型，由其决定下一步动作
        logger.error(f"[工具监控]工具 {tool_name} 调用失败：{e}", exc_info=True)
        return ToolMessage(
            content=f"工具 {tool_name} 执行失败：{e}",
            tool_call_id=tool_call_id,
            status="error",
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"[工具监控]工具 {tool_name} 调用完成，耗时 {elapsed_ms:.1f} ms")

    if tool_name == REPORT_SWITCH_TOOL:
        _mark_report_context(request)

    return result


def _mark_report_context(request: ToolCallRequest) -> None:
    """把当前会话标记为报告生成场景（供动态提示词切换使用）。"""
    context = getattr(request.runtime, "context", None)
    if isinstance(context, dict):
        context[REPORT_CONTEXT_FLAG] = True
        logger.info("[工具监控]已切换为报告生成场景，后续对话将使用报告提示词")
    else:
        logger.warning(
            "[工具监控]当前运行上下文不支持写入标记位，报告提示词切换可能不生效"
        )


def _normalize_tool_call_arguments(message: BaseMessage) -> Optional[BaseMessage]:
    """把单条消息里不规范的 ``function.arguments`` 修正为合法 JSON 字符串。

    返回修正后的消息；无需修正时返回 None。
    """
    raw_calls = getattr(message, "additional_kwargs", None) or {}
    if not isinstance(raw_calls, dict):
        return None
    raw_calls = raw_calls.get("tool_calls")
    if not raw_calls:
        return None

    parsed_calls: list[Any] = list(getattr(message, "tool_calls", None) or [])
    normalized_calls = []
    changed = False

    for index, raw_call in enumerate(raw_calls):
        raw_call = dict(raw_call)
        function = dict(raw_call.get("function") or {})

        parsed = parsed_calls[index] if index < len(parsed_calls) else None
        if parsed is not None:
            args = parsed.get("args")
            expected = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        else:
            # 完全无法解析时，至少要保证回传给接口的是合法 JSON
            current = function.get("arguments")
            if _is_valid_json(current):
                continue
            expected = "{}"

        if function.get("arguments") != expected:
            function["arguments"] = expected
            raw_call["function"] = function
            changed = True

        normalized_calls.append(raw_call)

    if not changed:
        return None

    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs["tool_calls"] = normalized_calls
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def _is_valid_json(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        json.loads(value)
        return True
    except (TypeError, ValueError):
        return False


@wrap_model_call
def normalize_tool_calls(request: ModelRequest, handler: Callable) -> Any:
    """模型调用前规范化历史消息中的工具调用参数。

    模型流式返回工具调用时，``function.arguments`` 有概率拼接出非法 JSON；
    直接回传给 DashScope 会返回 400 并中断整轮对话，这里统一修正为合法 JSON。
    """
    messages = list(request.messages)
    fixed_count = 0

    for index, message in enumerate(messages):
        fixed = _normalize_tool_call_arguments(message)
        if fixed is not None:
            messages[index] = fixed
            fixed_count += 1

    if fixed_count:
        logger.warning(
            f"[模型调用]修正了 {fixed_count} 条历史消息中不规范的 function.arguments"
        )
        request = request.override(messages=messages)

    return handler(request)


@wrap_model_call
def retry_model_call(request: ModelRequest, handler: Callable) -> Any:
    """模型调用兜底重试：上游适配器解析畸形响应时重试一次，而不是打断整轮对话。

    实测现场（``evals/results/`` 里 mh-003 的完整 traceback）::

        langchain_community/chat_models/tongyi.py:611  subtract_client_response
            prev_function["name"], ""            # KeyError: 'name'

    ``ChatTongyi`` 在流式增量拼接工具调用时，假定每个增量块都带 ``name`` 字段；
    qwen3-max 偶发返回不带 name 的工具调用块，拼接阶段就抛 ``KeyError('name')``。
    这个异常发生在「模型响应解析」阶段，比工具中间件更早，所以 ``monitor_tool``
    的防御抓不到它；``max_retries`` 只在 HTTP 层重试，同样救不了它。

    只重试一次、且只针对 ``KeyError``：模型重新生成一次大概率就是正常响应；
    网络类错误已经有 ``max_retries`` 兜底，不在这里重复打。
    """
    try:
        return handler(request)
    except KeyError as error:
        logger.warning(
            f"[模型调用]本次响应解析失败（{type(error).__name__}: {error}），"
            f"疑似上游适配器拼接工具调用时缺少 name 字段，重试一次"
        )
        return handler(request)


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    """模型调用埋点：记录进入模型前的上下文规模与最新消息摘要。"""
    messages = state["messages"]
    last_message = messages[-1]
    logger.info(
        f"[模型调用]即将调用模型，当前上下文共 {len(messages)} 条消息，"
        f"最新消息类型：{type(last_message).__name__}"
    )
    logger.debug(f"[模型调用]最新消息内容：{str(last_message.content)[:200]}")


@dynamic_prompt
def report_prompt_switch(request: ModelRequest) -> str:
    """动态提示词切换：报告场景使用报告提示词，其余场景使用系统提示词。"""
    context = getattr(request.runtime, "context", None) or {}
    if context.get(REPORT_CONTEXT_FLAG):
        logger.info("[提示词切换]当前为报告生成场景，使用报告提示词")
        return load_report_prompt()

    return load_system_prompt()