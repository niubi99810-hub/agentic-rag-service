"""RAG 总结服务：基于 LangChain LCEL 编排「检索 -> 提示词组装 -> 模型调用 -> 结果解析」。"""
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import get_chat_model
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.exception_handler import retry
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompt

NO_RESULT_TIP = "知识库中未检索到与该问题相关的资料，无法基于知识库回答。"


class RagSummarizeService:
    """检索增强总结服务：供 Agent 的 rag_summarize 工具调用。"""

    def __init__(self) -> None:
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_template = PromptTemplate.from_template(load_rag_prompt())
        self.model = get_chat_model()
        self.chain = self._init_chain()

    def _init_chain(self):
        """LCEL 链路：提示词组装 -> 大模型 -> 字符串解析。"""
        return self.prompt_template | self.model | StrOutputParser()

    def retrieve_docs(self, query: str) -> list[Document]:
        """向量检索，返回与 query 最相关的 TopK 文档片段。"""
        return self.retriever.invoke(query)

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        """把检索结果拼接为「参考资料 + 来源元数据」，降低模型幻觉。"""
        lines = []
        for index, doc in enumerate(docs, start=1):
            source = os.path.basename(str(doc.metadata.get("source", "未知来源")))
            lines.append(f"【参考资料{index}】（来源：{source}）：{doc.page_content}")
        return "\n".join(lines)

    @retry(times=2, delay=1.0, exceptions=(Exception,))
    def _invoke_chain(self, query: str, context: str) -> str:
        """执行 LCEL 链路；网络抖动等瞬时失败自动重试一次。"""
        return self.chain.invoke({"input": query, "context": context})

    def rag_summarize(self, query: str) -> str:
        """检索知识库并总结回答，任一步骤失败都返回可读提示而不是抛异常。"""
        logger.info(f"[RAG]开始检索，query={query}")
        try:
            docs = self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"[RAG]检索失败：{e}", exc_info=True)
            return "知识库检索失败，请稍后重试。"

        if not docs:
            logger.warning(f"[RAG]未检索到相关内容，query={query}")
            return NO_RESULT_TIP

        context = self.format_context(docs)
        logger.debug(f"[RAG]命中 {len(docs)} 个片段，TopK={chroma_conf['k']}")

        try:
            return self._invoke_chain(query, context)
        except Exception as e:
            logger.error(f"[RAG]总结失败：{e}", exc_info=True)
            return "知识库内容总结失败，请稍后重试。"


if __name__ == "__main__":
    service = RagSummarizeService()
    print(service.rag_summarize("小户型适合哪些扫地机器人"))