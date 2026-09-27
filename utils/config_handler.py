"""配置读取模块：统一从 config/*.yml 读取工程配置，实现参数与业务代码解耦。"""
import functools
import os
from typing import Any, Optional

import yaml

from utils.logger_handler import logger
from utils.path_tool import get_abs_path

DEFAULT_ENCODING = "utf-8"

RAG_CONFIG_PATH = "config/rag.yml"
CHROMA_CONFIG_PATH = "config/chroma.yml"
PROMPTS_CONFIG_PATH = "config/prompts.yml"
AGENT_CONFIG_PATH = "config/agent.yml"


def load_yaml(config_path: str, encoding: str = DEFAULT_ENCODING) -> dict[str, Any]:
    """读取单个 yaml 配置文件，失败时抛出带明确原因的异常。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config is None:
        logger.warning(f"[配置读取]{config_path}内容为空，已按空配置处理")
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误，期望 YAML 字典：{config_path}")

    return config


@functools.lru_cache(maxsize=None)
def _load_cached(config_path: str, encoding: str) -> dict[str, Any]:
    """带缓存的配置读取：同一个配置文件在一次进程内只解析一次。"""
    logger.debug(f"[配置读取]解析配置文件：{config_path}")
    return load_yaml(config_path, encoding)


def load_rag_config(config_path: Optional[str] = None, encoding: str = DEFAULT_ENCODING) -> dict[str, Any]:
    """大模型 / 向量模型相关配置。"""
    return _load_cached(config_path or get_abs_path(RAG_CONFIG_PATH), encoding)


def load_chroma_config(config_path: Optional[str] = None, encoding: str = DEFAULT_ENCODING) -> dict[str, Any]:
    """向量库与文本分块相关配置。"""
    return _load_cached(config_path or get_abs_path(CHROMA_CONFIG_PATH), encoding)


def load_prompts_config(config_path: Optional[str] = None, encoding: str = DEFAULT_ENCODING) -> dict[str, Any]:
    """各类提示词文件路径配置。"""
    return _load_cached(config_path or get_abs_path(PROMPTS_CONFIG_PATH), encoding)


def load_agent_config(config_path: Optional[str] = None, encoding: str = DEFAULT_ENCODING) -> dict[str, Any]:
    """Agent 运行时配置（外部业务数据路径、演示用户池等）。"""
    return _load_cached(config_path or get_abs_path(AGENT_CONFIG_PATH), encoding)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == "__main__":
    print("rag_conf       ->", rag_conf)
    print("chroma_conf    ->", chroma_conf)
    print("prompts_conf   ->", prompts_conf)
    print("agent_conf     ->", agent_conf)