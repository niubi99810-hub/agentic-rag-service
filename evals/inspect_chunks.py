"""离线分块体检：不联网、不调模型，直接看「语料被切成了什么样、金标答案落在哪一块」。

用法::

    python evals/inspect_chunks.py                                 # 用 chroma.yml 里的现有参数
    python evals/inspect_chunks.py --chunk-size 400 --overlap 40    # 试算调参之后的效果

为什么需要这个脚本：

召回失败有两种原因，报告上看起来一模一样，修法却完全相反 ——

- **答案被切碎**：一条问答横跨两个块，两块各占一半，谁都不完整 → 该动 chunk_size / 切分策略；
- **压根没召回到**：答案块是完整的，但排不进 TopK → 该动 embedding / rerank / query 改写。

还有一种最隐蔽的：**金标答案和语料对不上**（生成评测集时动了原文），
这种情况会在你调了半天参数之后才发现问题根本不在检索上。这个脚本会把它单独标出来。

它用的是项目里**同一个** RecursiveCharacterTextSplitter 和同一份 chroma.yml，
所以看到的块就是向量库里的块。
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import re
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SET = "evals/golden_retrieval.jsonl"
WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def squeeze(text: str) -> str:
    """去掉所有空白（含全角空格）。"""
    return WHITESPACE_RE.sub("", text or "")


def containment(reference: str, text: str) -> float:
    """参考答案有多大比例连续出现在 text 里（和 run_eval.py 同一口径）。"""
    reference, text = squeeze(reference), squeeze(text)
    if not reference or not text:
        return 0.0
    if reference in text:
        return 1.0
    matcher = difflib.SequenceMatcher(None, reference, text, autojunk=False)
    block = matcher.find_longest_match(0, len(reference), 0, len(text))
    return block.size / len(reference)


# --------------------------------------------------------------------------- #
# 1. 用项目里的真实分块器复算（需要 langchain / 工程配置）
# --------------------------------------------------------------------------- #
def load_chunks(chunk_size: int | None, overlap: int | None) -> tuple[dict[str, list[str]], dict[str, str]]:
    """返回 ({文件名: [块...]}, {文件名: 原文})。

    只有直接读语料这一步是纯 stdlib，其余都走工程里现成的组件，保证复算结果一致。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from utils.config_handler import chroma_conf
    from utils.file_handler import listdir_with_allowed_type, load_document
    from utils.path_tool import get_abs_path

    size = chunk_size or chroma_conf["chunk_size"]
    chunk_overlap = overlap if overlap is not None else chroma_conf["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=chunk_overlap,
        separators=chroma_conf["separators"],
        length_function=len,
    )

    data_path = get_abs_path(chroma_conf["data_path"])
    allowed = tuple(chroma_conf["allow_knowledge_file_type"])
    files = [p for p in listdir_with_allowed_type(data_path, allowed) if p.lower().endswith(".txt")]

    per_file: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    for path in files:
        documents = load_document(path)
        text = "\n".join(doc.page_content for doc in documents)
        texts[os.path.basename(path)] = text
        per_file[os.path.basename(path)] = splitter.split_text(text)
    return per_file, texts


# --------------------------------------------------------------------------- #
# 2. 定位金标答案（纯计算，可以离线单测）
# --------------------------------------------------------------------------- #
def locate(cases: list[dict], per_file: dict[str, list[str]], texts: dict[str, str], threshold: float) -> list[dict]:
    """对每条金标用例，找出它在分块体系里的落位情况。"""
    flat = [(name, index, chunk) for name, chunks in per_file.items() for index, chunk in enumerate(chunks, start=1)]

    rows: list[dict] = []
    for case in cases:
        reference = case["reference_answer"]

        best_score, best_name, best_index, best_len = 0.0, "", 0, 0
        for name, index, chunk in flat:
            score = containment(reference, chunk)
            if score > best_score:
                best_score, best_name, best_index, best_len = score, name, index, len(chunk)

        # 答案在语料原文里是否完整？不完整说明金标和语料对不上，而不是检索的问题
        intact_score = max((containment(reference, text) for text in texts.values()), default=0.0)

        if best_score >= threshold:
            verdict = "ok"
        elif intact_score >= threshold:
            verdict = "cut"
        else:
            verdict = "mismatch"

        rows.append(
            {
                "id": case.get("id", ""),
                "question": case["question"],
                "reference_answer": reference,
                "best_score": round(best_score, 3),
                "best_file": best_name,
                "best_chunk_index": best_index,
                "best_chunk_len": best_len,
                "intact_score": round(intact_score, 3),
                "fragment": best_len < 100,
                "verdict": verdict,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 3. 报告
# --------------------------------------------------------------------------- #
def render(per_file: dict[str, list[str]], rows: list[dict], chunk_size: int, overlap: int, threshold: float) -> str:
    lines = [
        "# 分块体检报告",
        "",
        f"- 参数：chunk_size={chunk_size}，chunk_overlap={overlap}",
        f"- 命中阈值：{threshold}",
        "",
        "## 一、分块概览",
        "",
        "| 文件 | 字符数 | 块数 | 各块长度 | 碎片块（<100 字符） |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, chunks in per_file.items():
        lengths = [len(chunk) for chunk in chunks]
        fragments = sum(1 for value in lengths if value < 100)
        preview = ", ".join(str(value) for value in lengths[:12]) + ("..." if len(lengths) > 12 else "")
        lines.append(f"| {name} | {sum(lengths)} | {len(chunks)} | {preview} | {fragments} |")

    buckets = {"ok": [], "cut": [], "mismatch": []}
    for row in rows:
        buckets[row["verdict"]].append(row)

    lines += [
        "",
        "## 二、金标落位",
        "",
        f"共 {len(rows)} 条用例：",
        "",
        "| 结论 | 条数 | 含义 | 该往哪修 |",
        "| --- | --- | --- | --- |",
        f"| 完整落在单块内 | {len(buckets['ok'])} | 答案在某一块里是完整的 | 不用动分块 |",
        f"| 被切碎 | {len(buckets['cut'])} | 语料里有完整答案，但没有任何单块完整包含它 | 调大 chunk_size / 按语义单元切分 |",
        f"| 金标对不上语料 | {len(buckets['mismatch'])} | 语料里根本找不到这段完整原文 | 是评测集的问题，不是检索的问题 |",
    ]

    if buckets["mismatch"]:
        lines += ["", "### 金标对不上语料的用例（先修这个，否则调参是在调空气）", ""]
        lines += ["| id | 问题 | 单块最佳包含度 | 语料最佳包含度 |", "| --- | --- | --- | --- |"]
        for row in buckets["mismatch"]:
            lines.append(
                f"| {row['id']} | {row['question'][:34]} | {row['best_score']:.2f} | {row['intact_score']:.2f} |"
            )

    if buckets["cut"]:
        lines += ["", "### 被切碎的用例", ""]
        lines += ["| id | 问题 | 单块最佳 | 语料拼接 | 最佳块来自 | 块长 |", "| --- | --- | --- | --- | --- | --- |"]
        for row in buckets["cut"]:
            lines.append(
                f"| {row['id']} | {row['question'][:34]} | {row['best_score']:.2f} | {row['intact_score']:.2f} "
                f"| {row['best_file']}#{row['best_chunk_index']} | {row['best_chunk_len']} |"
            )

    fragments = [row for row in rows if row["fragment"] and row["verdict"] == "ok"]
    if fragments:
        lines += [
            "",
            "### 答案落在碎片块里的用例（检索时容易被挤出 TopK）",
            "",
            "| id | 问题 | 所在块长 |",
            "| --- | --- | --- |",
        ]
        for row in fragments:
            lines.append(f"| {row['id']} | {row['question'][:34]} | {row['best_chunk_len']} |")

    return "\n".join(lines) + "\n"


def load_cases(relative_path: str) -> list[dict]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise SystemExit(f"评测集不存在：{path}\n先生成：python evals/build_golden.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="离线分块体检：语料被切成什么样、金标落在哪一块")
    parser.add_argument("--set", dest="case_set", default=DEFAULT_SET, help="评测集路径")
    parser.add_argument("--chunk-size", type=int, default=None, help="试算用的 chunk_size，默认取 chroma.yml")
    parser.add_argument("--overlap", type=int, default=None, help="试算用的 chunk_overlap，默认取 chroma.yml")
    parser.add_argument("--threshold", type=float, default=0.6, help="判定包含度阈值")
    args = parser.parse_args(argv)

    per_file, texts = load_chunks(args.chunk_size, args.overlap)
    if not per_file:
        raise SystemExit("data/ 下没有可解析的 txt 语料。")

    from utils.config_handler import chroma_conf

    size = args.chunk_size or chroma_conf["chunk_size"]
    overlap = args.overlap if args.overlap is not None else chroma_conf["chunk_overlap"]

    rows = locate(load_cases(args.case_set), per_file, texts, args.threshold)
    report = render(per_file, rows, size, overlap, args.threshold)

    results_dir = PROJECT_ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "chunks.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"报告已写入：{results_dir / 'chunks.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
