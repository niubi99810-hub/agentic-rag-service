"""ReAct Agent 封装模块。

基于 ``langchain.agents.create_agent`` 构建扫地机器人智能体：
- 注册 7 个工具（知识库检索 / 天气 / 用户信息 / 外部业务数据 / 报告上下文）；
- 挂载 4 个自定义中间件（工具监控、工具参数规范化、模型埋点、动态提示词切换）；
- 对外提供 ``execute``（一次性返回）与 ``execute_stream``（逐段流式返回）两种调用方式。
"""
from collections.abc import Iterator
from typing import Any, Optional, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import (
    log_before_model,
    monitor_tool,
    normalize_tool_calls,
    report_prompt_switch,
)
from model.factory import get_chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt


class AgentContext(TypedDict, total=False):
    """Agent 运行时上下文。

    ``report`` 为 True 时中间件会切换到报告生成提示词；
    每次会话传入独立的上下文对象，避免上一轮场景串到下一轮。
    """

    report: bool


class ReactAgent:
    """扫地机器人智能客服 ReAct Agent。"""

    def __init__(self, model: Optional[BaseChatModel] = None) -> None:
        self.tools = [
            rag_summarize,
            get_weather,
            get_user_id,
            get_user_location,
            get_current_month,
            fetch_external_data,
            fill_context_report,
        ]
        self.middleware = [
            monitor_tool,
            normalize_tool_calls,
            log_before_model,
            report_prompt_switch,
        ]
        self.agent = create_agent(
            model=model or get_chat_model(),
            tools=self.tools,
            system_prompt=load_system_prompt(),
            middleware=self.middleware,
            context_schema=AgentContext,
            name="sweeper_agent",
        )

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    @staticmethod
    def _new_context() -> AgentContext:
        """每次会话使用独立的上下文，默认非报告场景。"""
        return {"report": False}

    def execute(self, query: str) -> str:
        """一次性执行并返回最终回答文本。"""
        logger.info(f"[Agent]收到提问（非流式）：{query}")
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            context=self._new_context(),
        )
        return self._final_content(result["messages"])

    def execute_stream(self, query: str) -> Iterator[str]:
        """流式执行：以生成器形式逐段返回模型输出，实现打字机效果。"""
        logger.info(f"[Agent]收到提问（流式）：{query}")
        context = self._new_context()
        try:
            for chunk, metadata in self.agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="messages",  # 以「消息增量」为粒度，避免整段内容重复输出
                context=context,
            ):
                # 只输出模型节点产生的正文增量，过滤工具节点与工具调用参数
                if metadata.get("langgraph_node") != "model":
                    continue
                text = self._chunk_text(chunk)
                if text:
                    yield text
        except Exception as e:
            logger.error(f"[Agent]流式执行失败：{e}", exc_info=True)
            yield f"\n[系统提示]本次回答生成失败：{e}"

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        """从消息增量中提取文本内容。"""
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return ""

    @classmethod
    def _final_content(cls, messages: list[Any]) -> str:
        """取最后一条 AI 消息的文本作为最终回答。"""
        for message in reversed(messages):
            if message.__class__.__name__.startswith("AI"):
                text = cls._chunk_text(message)
                if text:
                    return text
        return ""


if __name__ == "__main__":
    agent = ReactAgent()
    for piece in agent.execute_stream("扫地机器人在我所在地区的气温下如何保养"):
        print(piece, end="", flush=True)
    print()