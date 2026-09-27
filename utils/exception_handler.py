"""异常容错模块：为关键链路提供统一的异常捕获与重试装饰器。

配合 utils/logger_handler.py 使用，保证「单文件加载失败 / 单工具调用失败」
不会中断整体任务，同时把失败原因完整落到日志里。
"""
import functools
import time
from typing import Any, Callable, Tuple, Type

from utils.logger_handler import logger

ExceptionTypes = Tuple[Type[BaseException], ...]


def catch_exceptions(
    default: Any = None,
    exceptions: ExceptionTypes = (Exception,),
    log=None,
) -> Callable:
    """捕获异常并返回兜底值，异常详情写入日志。"""

    def decorator(func: Callable) -> Callable:
        error_logger = log or logger

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                error_logger.error(
                    f"[异常捕获]{func.__qualname__} 执行失败：{e}", exc_info=True
                )
                return default

        return wrapper

    return decorator


def retry(
    times: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: ExceptionTypes = (Exception,),
    log=None,
) -> Callable:
    """失败自动重试装饰器，常用于外部接口 / 大模型调用等不稳定链路。"""

    def decorator(func: Callable) -> Callable:
        error_logger = log or logger

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= times:
                        error_logger.error(
                            f"[重试]{func.__qualname__} 已重试 {times} 次仍失败：{e}",
                            exc_info=True,
                        )
                        raise
                    error_logger.warning(
                        f"[重试]{func.__qualname__} 第 {attempt} 次失败：{e}，"
                        f"{current_delay:.1f}s 后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None

        return wrapper

    return decorator