"""路径处理模块：为整个工程提供统一的绝对路径，屏蔽不同启动目录带来的差异。"""
import os


def get_project_root() -> str:
    """返回工程根目录（即 utils 包的上一级目录）。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def get_abs_path(relative_path: str) -> str:
    """把工程内的相对路径转换为绝对路径。

    统一入口可以保证无论从哪个目录启动脚本，读写到的都是同一份文件。
    """
    return os.path.join(get_project_root(), relative_path)


if __name__ == "__main__":
    print(get_abs_path("config/rag.yml"))