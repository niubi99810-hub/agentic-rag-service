"""极简未定义名检查器：等价于 pyflakes 的 F821 那一小部分。

为什么要自己写一个：跑分脚本（`evals/*.py`）里到处是 print 和格式化字符串引用局部变量，
删代码时很容易留一个悬空引用 —— 这种错只有真跑起来才炸，而 `--mode agent` 要连大模型、
花真钱、跑好几分钟才会走到那一行。这个检查器不执行代码，纯 AST 静态扫描，1 秒出结果。

检查范围：
  - 每个函数体内 Load 上下文的名字，是否能在「本函数绑定 / 外层闭包 / 模块全局 / 内置」里找到；
刻意不做：
  - 跨模块属性分析；`try/except ImportError` 之类的条件定义；
  - 类体作用域不算闭包（Python 语义如此），所以类里的名字在方法中必须写 `self.x` / `Cls.x`；
  - 装饰器和默认参数在外层作用域求值，这里按函数自身处理（漏报，不会误报）。

自带自测：`python namecheck.py --selftest`
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import sys

SKIP_DIRS = {"__pycache__", ".venv", "venv", ".git", "chroma_db", "node_modules"}
BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__",
}


def _bindings_of(node: ast.AST) -> set[str]:
    """单个节点「自身」绑定的名字，不递归。"""
    out: set[str] = set()
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        out.add(node.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            out.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        out.add(node.name)
        args = node.args
        for item in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            out.add(item.arg)
        if args.vararg:
            out.add(args.vararg.arg)
        if args.kwarg:
            out.add(args.kwarg.arg)
    elif isinstance(node, ast.ClassDef):
        out.add(node.name)
    elif isinstance(node, ast.Lambda):
        args = node.args
        for item in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            out.add(item.arg)
        if args.vararg:
            out.add(args.vararg.arg)
        if args.kwarg:
            out.add(args.kwarg.arg)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        out.add(node.name)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        out.update(node.names)
    elif isinstance(node, ast.arg):
        out.add(node.arg)
    elif isinstance(node, ast.MatchAs) and node.name:
        out.add(node.name)
    elif isinstance(node, ast.MatchStar) and node.name:
        out.add(node.name)
    elif isinstance(node, ast.MatchMapping) and node.rest:
        out.add(node.rest)
    return out


def _iter_in_scope(node: ast.AST):
    """遍历 node 的直接作用域，不进入嵌套函数/类的函数体（但把定义本身产出）。"""
    for child in ast.iter_child_nodes(node):
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from _iter_in_scope(child)


def _bound_names(node: ast.AST, scope_only: bool = False) -> set[str]:
    """node 作用域里被绑定的全部名字。

    `scope_only=True` 用于模块作用域：只认模块这一层绑定的名字，
    不能把某个函数内部的局部变量算成全局名（否则会漏报真实的未定义名）。
    """
    iterator = _iter_in_scope(node) if scope_only else ast.walk(node)
    bound: set[str] = set()
    for child in iterator:
        if scope_only and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
            continue
        bound |= _bindings_of(child)
    return bound


def _used_names(node: ast.AST) -> set[str]:
    """node 子树里 Load 上下文的名字（含嵌套函数，属于过近似，只会更宽松）。"""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    mapping: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            mapping[child] = node
    return mapping


def check_source(source: str, label: str) -> list[str]:
    tree = ast.parse(source, filename=label)
    parents = _parents(tree)
    module_scope = _bound_names(tree, scope_only=True)
    problems: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        enclosing: set[str] = set()
        cursor = parents.get(node)
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                enclosing |= _bound_names(cursor)
            cursor = parents.get(cursor)

        local = _bound_names(node)
        for name in sorted(_used_names(node)):
            if name in local or name in enclosing or name in module_scope:
                continue
            if name in BUILTINS or name in {"self", "cls"}:
                continue
            problems.append(f"{label}:{node.lineno} 函数 {node.name}() 引用了未定义的名字：{name}")
    return problems


def check_file(path: pathlib.Path) -> list[str]:
    try:
        return check_source(path.read_text(encoding="utf-8"), path.name)
    except SyntaxError as exc:
        return [f"{path.name}:{exc.lineno} 语法错误：{exc.msg}"]


SELFTEST_SHOULD_FLAG = """\
def outer(value):
    return value + never_defined


def sibling():
    return also_missing()


def comprehension_leak():
    return [item + oops for item in range(3)]
"""

SELFTEST_SHOULD_PASS = """\
import os
from typing import Any

CONST = 3


def uses_global(value: int) -> int:
    return value + CONST


def closure_ok():
    total = 1

    def inner():
        return total + CONST + len(os.sep)

    return inner


def bindings_ok(data: Any):
    for index, item in enumerate(data or []):
        try:
            pass
        except ValueError as exc:
            _ = exc
        if (found := len(str(item))) > index:
            _ = found
    total = [key for key in range(3)]
    kw = {"a": 1}
    return total, {**kw}, f"{index}"


def lambda_ok():
    return sorted([1, 2, 3], key=lambda value: value + CONST)
"""


def selftest() -> int:
    failures = 0

    flagged = check_source(SELFTEST_SHOULD_FLAG, "<should-flag>")
    expected = {"never_defined", "also_missing", "oops"}
    got = {item.rsplit("：", 1)[-1] for item in flagged}
    if got != expected:
        failures += 1
        print(f"  FAIL 应报出的未定义名不对：期望 {sorted(expected)}，实际 {sorted(got)}")
    else:
        print(f"  PASS 抓出 3 个未定义名 {sorted(got)}")

    clean = check_source(SELFTEST_SHOULD_PASS, "<should-pass>")
    if clean:
        failures += 1
        print("  FAIL 合法代码被误报：")
        for item in clean:
            print("        " + item)
    else:
        print("  PASS 闭包/全局/推导式/海象/except-as 都不误报")

    print(f"自测失败 {failures} 项")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()

    targets: list[pathlib.Path] = []
    for raw in argv:
        path = pathlib.Path(raw)
        found = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        targets.extend(item for item in found if not SKIP_DIRS & set(item.parts))

    problems: list[str] = []
    for target in targets:
        problems.extend(check_file(target))

    print(f"扫描 {len(targets)} 个文件")
    for item in problems:
        print("  " + item)
    print(f"问题 {len(problems)} 处")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
