"""Agent 工具集。

包含 7 个工具：
1. ``rag_summarize``       —— 私有知识库检索（RAG）
2. ``get_weather``         —— 城市天气查询（演示数据）
3. ``get_user_id``         —— 获取当前用户 ID
4. ``get_user_location``   —— 获取用户所在城市
5. ``get_current_month``   —— 获取当前月份（YYYY-MM）
6. ``fetch_external_data`` —— 读取外部 CSV 业务数据（用户月度使用记录）
7. ``fill_context_report`` —— 触发中间件切换「报告生成」场景

说明：除知识库与外部 CSV 以外的工具当前为演示实现，
接入真实业务时只需替换函数内部实现，工具签名（模型可见的契约）保持不变。
"""
import csv
import hashlib
import json
import os
import random
from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.exception_handler import catch_exceptions
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

MONTH_FORMAT = "%Y-%m"
DEFAULT_USER_ID_POOL = ["001", "002", "003", "004", "005", "006", "007"]
DEFAULT_CITY_POOL = ["深圳", "合肥", "杭州"]

USER_ID_POOL: list[str] = list(agent_conf.get("user_id_pool", DEFAULT_USER_ID_POOL))
CITY_POOL: list[str] = list(agent_conf.get("user_city_pool", DEFAULT_CITY_POOL))

# 外部业务数据缓存：{user_id: {month: {字段: 值}}}
_EXTERNAL_DATA: Optional[dict[str, dict[str, dict[str, str]]]] = None

# RAG 服务延迟初始化：避免导入工具模块时就加载向量库与模型
_RAG_SERVICE: Optional[RagSummarizeService] = None


def _get_rag_service() -> RagSummarizeService:
    global _RAG_SERVICE
    if _RAG_SERVICE is None:
        logger.info("[工具]初始化 RAG 检索服务")
        _RAG_SERVICE = RagSummarizeService()
    return _RAG_SERVICE


# --------------------------------------------------------------------------- #
# 1. 知识库检索
# --------------------------------------------------------------------------- #
@tool(description="从扫地机器人私有知识库中检索参考资料，入参为检索词，返回最相关的知识库片段文本")
def rag_summarize(query: str) -> str:
    """检索私有知识库并总结回答，用于故障排查、选购、保养、参数等咨询场景。"""
    return _get_rag_service().rag_summarize(query)


# --------------------------------------------------------------------------- #
# 2. 天气查询
# --------------------------------------------------------------------------- #
@tool(description="获取指定城市的天气，结果以字符串的形式返回")
def get_weather(city: str) -> str:
    """查询指定城市天气（演示数据：真实环境可替换为和风天气 / 高德等开放接口）。"""
    # 用城市名做种子，保证同一城市多次查询结果稳定，便于演示与回归
    seed = int(hashlib.md5(str(city).encode("utf-8")).hexdigest()[:8], 16)
    rnd = random.Random(seed)

    weather = rnd.choice(["晴", "多云", "阴", "小雨"])
    temperature = rnd.randint(8, 34)
    humidity = rnd.randint(30, 85)
    aqi = rnd.randint(20, 120)
    rain_probability = rnd.choice(["极低", "较低", "中等", "较高"])

    logger.info(f"[get_weather]查询城市 {city} 的天气（演示数据）")
    return (
        f"城市{city}为{weather}，气温{temperature}摄氏度，空气湿度{humidity}%，"
        f"AQI为{aqi}，最近六小时降雨概率{rain_probability}"
    )


# --------------------------------------------------------------------------- #
# 3 / 4 / 5. 用户与时间信息
# --------------------------------------------------------------------------- #
@tool(description="获取用户所在城市的名称，以字符串的形式返回")
def get_user_location() -> str:
    """获取用户所在城市（演示数据）。"""
    return random.choice(CITY_POOL)


@tool(description="获取用户id，以纯字符串返回")
def get_user_id() -> str:
    """获取当前对话用户 ID（演示数据）。"""
    return random.choice(USER_ID_POOL)


@tool(description="获取当前月份，以纯字符串形式返回，格式固定为 YYYY-MM")
def get_current_month() -> str:
    """获取系统当前月份，格式 YYYY-MM。"""
    month = datetime.now().strftime(MONTH_FORMAT)
    logger.info(f"[get_current_month]当前月份：{month}")
    return month


# --------------------------------------------------------------------------- #
# 6. 外部 CSV 业务数据
# --------------------------------------------------------------------------- #
@catch_exceptions(default=None)
def _read_external_data(external_data_path: str) -> dict[str, dict[str, dict[str, str]]]:
    """从 CSV 文件读取外部业务数据，异常时由装饰器兜底返回 None。"""
    data: dict[str, dict[str, dict[str, str]]] = {}
    # utf-8-sig 兼容 Excel 导出的带 BOM 文件
    with open(external_data_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            user_id = (row.get("user_id") or "").strip()
            month = (row.get("month") or "").strip()
            if not user_id or not month:
                continue
            record = {
                key: (value or "").strip()
                for key, value in row.items()
                if key not in ("user_id", "month")
            }
            data.setdefault(user_id, {})[month] = record
    return data


def _load_external_data() -> dict[str, dict[str, dict[str, str]]]:
    """加载外部 CSV 业务数据（进程内缓存，只读一次）。

    文件缺失或解析失败时返回空字典，保证报告场景不会因为数据源异常而崩溃。
    """
    global _EXTERNAL_DATA
    if _EXTERNAL_DATA is not None:
        return _EXTERNAL_DATA

    external_data_path = get_abs_path(agent_conf["external_data_path"])
    if not os.path.exists(external_data_path):
        logger.error(f"[外部数据]数据文件不存在：{external_data_path}")
        _EXTERNAL_DATA = {}
        return _EXTERNAL_DATA

    data = _read_external_data(external_data_path) or {}
    logger.info(f"[外部数据]加载完成，共 {len(data)} 位用户，数据文件：{external_data_path}")
    _EXTERNAL_DATA = data
    return _EXTERNAL_DATA


def _format_record(record: dict[str, Any]) -> str:
    """把 CSV 行格式化为模型易读的文本。"""
    labels = {
        "feature": "使用特征",
        "efficiency": "清洁效率",
        "consumables": "耗材状态",
        "comparison": "环比对比",
        "time": "统计周期",
    }
    parts = [
        f"{labels.get(key, key)}：{value}"
        for key, value in record.items()
        if str(value).strip()
    ]
    return "；".join(parts)


@tool(
    description=(
        "从外部系统中获取指定用户在指定月份的扫地机器人使用记录，"
        "入参为 user_id（用户ID）与 month（月份，格式 YYYY-MM，可省略，省略时默认当前月份），"
        "结果以纯字符串返回；未检索到时返回空字符串"
    )
)
def fetch_external_data(user_id: str, month: str = "") -> str:
    """读取外部业务系统的用户月度使用记录。

    ``month`` 可省略：省略时默认取当前月份，避免模型漏传参数导致整轮任务失败。
    """
    user_id = str(user_id).strip()
    month = str(month).strip()
    if not month:
        month = get_current_month.func()
        logger.warning(f"[fetch_external_data]未传入月份，默认使用当前月份：{month}")

    data = _load_external_data()
    record = data.get(user_id, {}).get(month)
    if not record:
        logger.warning(f"[fetch_external_data]未能检索到用户 {user_id} 在 {month} 的使用记录")
        return ""

    logger.info(f"[fetch_external_data]命中用户 {user_id} / {month} 的使用记录")
    return _format_record(record)


# --------------------------------------------------------------------------- #
# 7. 报告场景上下文注入
# --------------------------------------------------------------------------- #
@tool(
    description=(
        "无入参，无返回值，调用后由中间件把当前会话切换为「报告生成」场景，"
        "为后续提示词提供上下文信息"
    )
)
def fill_context_report() -> str:
    """触发中间件为报告生成场景注入上下文（提示词动态切换的开关）。"""
    logger.info("[fill_context_report]已触发报告生成场景上下文注入")
    return "fill_context_report已调用"


if __name__ == "__main__":
    # 工具自检：不依赖大模型，直接调用函数体
    print(rag_summarize.invoke({"query": "扫地机器人迷路了怎么办"}))
    print(get_weather.invoke({"city": "杭州"}))
    print(get_user_id.invoke({}))
    print(get_user_location.invoke({}))
    print(get_current_month.invoke({}))
    print(fetch_external_data.invoke({"user_id": "001", "month": "2025-06"}))
    print(fill_context_report.invoke({}))
    print(json.dumps(_EXTERNAL_DATA, ensure_ascii=False)[:200] if _EXTERNAL_DATA else "无外部数据")