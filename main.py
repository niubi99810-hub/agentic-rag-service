"""工程入口：知识库构建 + 交互式 / 单次问答 CLI。

用法示例::

    python main.py --build-kb                 # 只构建（增量）知识库
    python main.py --rebuild-kb               # 清空后全量重建知识库
    python main.py --ask "扫地机器人迷路了怎么办"
    python main.py --report                   # 生成当前用户当月使用报告
    python main.py                            # 进入交互式对话
"""
import argparse
import os
import sys

from utils.logger_handler import logger
from utils.path_tool import get_abs_path

BANNER = """
==============================================================
 扫地机器人智能客服 Agent（LangChain + LangGraph + Chroma + RAG）
 输入问题直接回车提问；输入 kb 构建知识库；输入 exit / quit 退出
==============================================================
"""


def bootstrap_env() -> None:
    """加载工程根目录下的 .env（存放 DASHSCOPE_API_KEY 等敏感配置）。"""
    env_path = get_abs_path(".env")
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        logger.debug(f"[启动]已加载环境变量文件：{env_path}")
    except ImportError:
        logger.warning("[启动]未安装 python-dotenv，跳过 .env 加载")


def build_knowledge_base(rebuild: bool = False) -> None:
    """构建向量知识库。"""
    from rag.vector_store import VectorStoreService

    service = VectorStoreService()
    if rebuild:
        service.reset()

    stats = service.build_index()
    logger.info(
        f"[启动]知识库构建完成：扫描 {stats['total']} 个文件，"
        f"新增 {stats['loaded']}，跳过 {stats['skipped']}，失败 {stats['failed']}"
    )


def build_agent():
    """延迟导入并构建 Agent，保证 --help / --build-kb 等场景不需要 API Key。"""
    from agent.react_agent import ReactAgent

    return ReactAgent()


def ask_once(agent, query: str) -> None:
    """单次提问并以流式方式打印回答。"""
    print(f"\n用户：{query}\nAgent：", end="", flush=True)
    for piece in agent.execute_stream(query):
        print(piece, end="", flush=True)
    print("\n")


def interactive(agent) -> None:
    """交互式多轮对话。"""
    print(BANNER)
    while True:
        try:
            query = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("再见！")
            break
        if query.lower() in {"kb", "build"}:
            build_knowledge_base()
            continue

        print("Agent：", end="", flush=True)
        for piece in agent.execute_stream(query):
            print(piece, end="", flush=True)
        print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫地机器人智能客服 Agent")
    parser.add_argument("--build-kb", action="store_true", help="增量构建向量知识库后退出")
    parser.add_argument("--rebuild-kb", action="store_true", help="清空向量库后全量重建后退出")
    parser.add_argument("--ask", type=str, default=None, help="单次提问，直接返回回答")
    parser.add_argument("--report", action="store_true", help="生成当前用户当月使用报告")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    bootstrap_env()
    args = parse_args(argv)

    try:
        if args.build_kb or args.rebuild_kb:
            build_knowledge_base(rebuild=args.rebuild_kb)
            return 0

        if args.report:
            agent = build_agent()
            ask_once(agent, "请生成我的扫地机器人本月使用报告")
            return 0

        if args.ask:
            agent = build_agent()
            ask_once(agent, args.ask)
            return 0

        agent = build_agent()
        interactive(agent)
        return 0
    except KeyboardInterrupt:
        print("\n已中断，再见！")
        return 130
    except Exception as e:
        logger.error(f"[启动]运行失败：{e}", exc_info=True)
        print(f"\n运行失败：{e}")
        print("请检查：1) 是否配置 DASHSCOPE_API_KEY；2) 是否已构建知识库（python main.py --build-kb）")
        return 1


if __name__ == "__main__":
    sys.exit(main())