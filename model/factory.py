"""模型工厂模块。

设计要点：
1. ``BaseModelFactory`` 抽象基类统一 ``generator()`` 接口，对话模型与向量模型各自实现；
2. 模型名称、温度、重试次数全部来自 ``config/rag.yml``，切换模型无需改动业务代码；
3. 通过 ``functools.lru_cache`` 做进程内单例 + 懒加载，避免导入即校验 API Key。

用法::

    from model.factory import get_chat_model, get_embed_model
"""
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from utils.config_handler import rag_conf
from utils.logger_handler import logger


class BaseModelFactory(ABC):
    """模型工厂抽象基类。"""

    @abstractmethod
    def generator(self) -> BaseChatModel | Embeddings:
        """生产模型实例。"""
        raise NotImplementedError


class ChatModelFactory(BaseModelFactory):
    """对话大模型工厂（DashScope 通义千问）。"""

    def generator(self) -> BaseChatModel:
        model_name = rag_conf["chat_model_name"]

        # ChatTongyi 通过 model_kwargs 透传采样参数（temperature 等）
        model_kwargs: Dict[str, Any] = {}
        if rag_conf.get("temperature") is not None:
            model_kwargs["temperature"] = rag_conf["temperature"]
        if rag_conf.get("top_p") is not None:
            model_kwargs["top_p"] = rag_conf["top_p"]

        logger.info(
            f"[模型工厂]创建对话模型：{model_name}，"
            f"采样参数={model_kwargs or '默认'}，"
            f"max_retries={rag_conf.get('max_retries', 3)}"
        )
        return ChatTongyi(
            model=model_name,
            model_kwargs=model_kwargs,
            streaming=True,  # 开启流式，配合 Agent 打字机输出
            max_retries=rag_conf.get("max_retries", 3),
        )


class EmbeddingsFactory(BaseModelFactory):
    """Embedding 向量模型工厂（DashScope text-embedding 系列）。"""

    def generator(self) -> Embeddings:
        model_name = rag_conf["embedding_model_name"]
        logger.info(f"[模型工厂]创建向量模型：{model_name}")
        return DashScopeEmbeddings(model=model_name)


@functools.lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """获取全局唯一的对话模型实例（懒加载）。"""
    return ChatModelFactory().generator()


@functools.lru_cache(maxsize=1)
def get_embed_model() -> Embeddings:
    """获取全局唯一的 Embedding 模型实例（懒加载）。"""
    return EmbeddingsFactory().generator()


def __getattr__(name: str):
    """兼容 ``from model.factory import chat_model / embed_model`` 的旧写法。"""
    if name == "chat_model":
        return get_chat_model()
    if name == "embed_model":
        return get_embed_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    print(get_chat_model())
    print(get_embed_model())