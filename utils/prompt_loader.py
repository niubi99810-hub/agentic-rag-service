"""提示词加载模块：把各场景提示词从 txt 文件读取出来，与业务代码解耦。"""
import functools

from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

SYSTEM_PROMPT_KEY = "main_prompt_path"
RAG_PROMPT_KEY = "rag_summarize_prompt_path"
REPORT_PROMPT_KEY = "report_prompts_path"


def _read_prompt(config_key: str, scene: str) -> str:
    """按配置项读取提示词文件内容。"""
    if config_key not in prompts_conf:
        raise KeyError(f"[提示词加载]config/prompts.yml 中缺少配置项：{config_key}")

    prompt_path = get_abs_path(prompts_conf[config_key])
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError as e:
        logger.error(f"[提示词加载]{scene}提示词读取失败：{prompt_path}，原因：{e}")
        raise

    if not content:
        logger.warning(f"[提示词加载]{scene}提示词内容为空：{prompt_path}")
    return content


@functools.lru_cache(maxsize=None)
def load_system_prompt() -> str:
    """系统提示词：Agent 主场景（咨询 / 故障排查 / 选购 / 保养）。"""
    return _read_prompt(SYSTEM_PROMPT_KEY, "系统")


@functools.lru_cache(maxsize=None)
def load_rag_prompt() -> str:
    """RAG 总结提示词：基于检索到的参考资料做概括回答。"""
    return _read_prompt(RAG_PROMPT_KEY, "RAG 总结")


@functools.lru_cache(maxsize=None)
def load_report_prompt() -> str:
    """报告生成提示词：动态切换后的报告场景提示词。"""
    return _read_prompt(REPORT_PROMPT_KEY, "报告生成")


if __name__ == "__main__":
    print("========== 系统提示词 ==========")
    print(load_system_prompt())
    print("========== RAG 提示词 ==========")
    print(load_rag_prompt())
    print("========== 报告提示词 ==========")
    print(load_report_prompt())