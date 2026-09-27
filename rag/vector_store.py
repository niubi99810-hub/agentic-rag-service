"""向量库服务模块。

职责：
1. 本地 PDF / TXT 文档批量导入，解析 -> 分块 -> 向量化 -> 入库；
2. 基于文件 MD5 的增量更新，已入库文件自动跳过，避免重复向量化；
3. 对外提供 Retriever / 相似度检索能力。

单文件加载失败只会记录日志并跳过，不会中断整批知识库构建。
"""
import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import get_embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import get_file_md5, listdir_with_allowed_type, load_document
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    """Chroma 向量库服务：知识库构建 + 增量更新 + 语义检索。"""

    def __init__(self) -> None:
        # 统一转绝对路径，避免不同启动目录导致「知识库明明建过却检索不到」
        self.persist_directory = get_abs_path(chroma_conf["persist_directory"])
        self.collection_name = chroma_conf["collection_name"]
        self.md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = self._create_store()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    # ------------------------------------------------------------------ #
    # 基础能力
    # ------------------------------------------------------------------ #
    def _create_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=get_embed_model(),
            persist_directory=self.persist_directory,
        )

    def get_retriever(self, k: Optional[int] = None):
        """获取向量检索器，默认 TopK 取自 yaml 配置。"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": k or chroma_conf["k"]}
        )

    def similarity_search(self, query: str, k: Optional[int] = None) -> list[Document]:
        """直接做一次相似度检索，便于调试与上层复用。"""
        return self.vector_store.similarity_search(query, k=k or chroma_conf["k"])

    def count(self) -> int:
        """当前集合内的向量数量。"""
        try:
            return len(self.vector_store.get(include=[])["ids"])
        except Exception as e:
            logger.error(f"[向量库]统计向量数量失败：{e}")
            return 0

    # ------------------------------------------------------------------ #
    # MD5 增量更新
    # ------------------------------------------------------------------ #
    def _read_md5_store(self) -> set[str]:
        if not os.path.exists(self.md5_store_path):
            return set()
        try:
            with open(self.md5_store_path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except OSError as e:
            logger.error(f"[知识库]MD5 记录读取失败：{e}")
            return set()

    def is_indexed(self, md5_hex: str) -> bool:
        """判断文件是否已经入库过。"""
        return md5_hex in self._read_md5_store()

    def mark_indexed(self, md5_hex: str) -> None:
        """记录已入库文件的 MD5。"""
        try:
            with open(self.md5_store_path, "a", encoding="utf-8") as f:
                f.write(md5_hex + "\n")
        except OSError as e:
            logger.error(f"[知识库]MD5 记录写入失败：{e}")

    # ------------------------------------------------------------------ #
    # 知识库构建
    # ------------------------------------------------------------------ #
    def add_file(self, filepath: str, force: bool = False) -> bool:
        """把单个文件写入向量库，返回是否成功入库。"""
        md5_hex = get_file_md5(filepath)
        if not md5_hex:
            logger.error(f"[知识库]{filepath} 无法计算 MD5，跳过")
            return False

        if not force and self.is_indexed(md5_hex):
            logger.info(f"[知识库]{filepath} 内容已存在知识库内，跳过")
            return False

        try:
            documents: list[Document] = load_document(filepath)
            if not documents:
                logger.warning(f"[知识库]{filepath} 内没有有效内容，跳过")
                return False

            split_documents: list[Document] = self.splitter.split_documents(documents)
            if not split_documents:
                logger.warning(f"[知识库]{filepath} 分块后内容为空，跳过")
                return False

            self.vector_store.add_documents(split_documents)
            self.mark_indexed(md5_hex)
            logger.info(
                f"[知识库]{filepath} 加载成功，切分为 {len(split_documents)} 个文本块"
            )
            return True
        except Exception as e:
            # 单个文件失败不影响其它文件入库
            logger.error(f"[知识库]{filepath} 加载失败：{e}", exc_info=True)
            return False

    def build_index(self, force: bool = False) -> dict[str, int]:
        """扫描数据目录并批量构建知识库，返回本次构建的统计信息。"""
        data_path = get_abs_path(chroma_conf["data_path"])
        allowed_types = tuple(chroma_conf["allow_knowledge_file_type"])
        files = listdir_with_allowed_type(data_path, allowed_types)

        stats = {"total": len(files), "loaded": 0, "skipped": 0, "failed": 0}
        if not files:
            logger.warning(f"[知识库]{data_path} 下没有可入库的 {allowed_types} 文件")
            return stats

        logger.info(f"[知识库]开始构建，共发现 {len(files)} 个待处理文件（force={force}）")
        for path in files:
            md5_hex = get_file_md5(path)
            if not force and md5_hex and self.is_indexed(md5_hex):
                logger.info(f"[知识库]{path} 内容已存在知识库内，跳过")
                stats["skipped"] += 1
                continue
            if self.add_file(path, force=force):
                stats["loaded"] += 1
            else:
                stats["failed"] += 1

        logger.info(f"[知识库]构建完成：{stats}，当前集合向量数={self.count()}")
        return stats

    def reset(self) -> None:
        """清空向量集合与 MD5 记录，用于全量重建知识库。"""
        try:
            self.vector_store.delete_collection()
            logger.warning("[知识库]已清空向量集合，准备全量重建")
        except Exception as e:
            logger.error(f"[知识库]清空向量集合失败：{e}")
        self.vector_store = self._create_store()

        if os.path.exists(self.md5_store_path):
            os.remove(self.md5_store_path)
            logger.warning("[知识库]已清空 MD5 增量更新记录")


if __name__ == "__main__":
    service = VectorStoreService()
    print("构建统计：", service.build_index())

    retriever = service.get_retriever()
    for index, doc in enumerate(retriever.invoke("扫地机器人迷路了怎么办"), start=1):
        print(f"---- 命中片段 {index}（来源：{doc.metadata.get('source')}）----")
        print(doc.page_content)