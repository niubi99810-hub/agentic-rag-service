"""文档加载模块：负责文件 MD5 计算、目录扫描以及 txt / pdf 文档解析。

知识库构建过程中「单个文件解析失败不应中断整体任务」，因此所有对外函数
都做了异常捕获并返回空结果，具体原因写入日志。
"""
import hashlib
import os
from typing import Optional, Sequence

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import logger

# 依次尝试的文本编码：优先 UTF-8，再兼容 Windows 上常见的 GBK / GB18030
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030")

MD5_CHUNK_SIZE = 4096


def get_file_md5(filepath: str) -> Optional[str]:
    """分块计算文件 MD5，用于知识库增量更新判断。"""
    if not os.path.exists(filepath):
        logger.error(f"[MD5计算]文件不存在：{filepath}")
        return None
    if not os.path.isfile(filepath):
        logger.error(f"[MD5计算]路径不是文件：{filepath}")
        return None

    md5_obj = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(MD5_CHUNK_SIZE):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except OSError as e:
        logger.error(f"[MD5计算]文件 {filepath} 计算失败：{e}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: Sequence[str]) -> tuple[str, ...]:
    """列出目录下指定后缀的文件（不递归），路径不存在或类型为空时返回空元组。"""
    if not os.path.isdir(path):
        logger.error(f"[目录扫描]{path} 不是有效文件夹")
        return ()

    allowed = tuple(allowed_types or ())
    files = [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.isfile(os.path.join(path, name)) and name.endswith(allowed)
    ]
    return tuple(files)


def txt_loder(filepath: str) -> list[Document]:
    """加载 txt 文档，自动兼容 UTF-8 / GBK 等常见编码。"""
    last_error: Optional[Exception] = None
    for encoding in TEXT_ENCODINGS:
        try:
            return TextLoader(filepath, encoding=encoding).load()
        except Exception as e:  # TextLoader 会把解码异常包装成 RuntimeError
            last_error = e

    logger.error(f"[文档加载]{filepath} 解析失败（已尝试 {TEXT_ENCODINGS}）：{last_error}")
    return []


def pdf_loder(filepath: str, password: Optional[str] = None) -> list[Document]:
    """加载 pdf 文档，未安装 pypdf 时给出可读的提示。"""
    try:
        return PyPDFLoader(filepath, password).load()
    except ImportError:
        logger.error("[文档加载]缺少 PDF 解析依赖，请先执行：pip install pypdf")
    except Exception as e:
        logger.error(f"[文档加载]{filepath} 解析失败：{e}")
    return []


def load_document(filepath: str) -> list[Document]:
    """按文件后缀分发到对应的文档解析器。"""
    suffix = os.path.splitext(filepath)[1].lower()
    if suffix == ".txt":
        return txt_loder(filepath)
    if suffix == ".pdf":
        return pdf_loder(filepath)
    logger.warning(f"[文档加载]暂不支持的文件类型：{filepath}")
    return []