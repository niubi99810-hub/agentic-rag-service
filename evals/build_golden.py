"""从知识库语料自动生成检索评测集（golden set）。

用法::

    python evals/build_golden.py                  # 问答手册全量：30 条
    python evals/build_golden.py --with-manuals   # 再加三份手册合成的问句：50 条
    python evals/build_golden.py --limit 10       # 只等距抽 10 条
    python evals/build_golden.py --show           # 只看解析结果，不写文件

为什么评测集要从语料自动派生（面试常问「你的评测集哪来的」）：

1. **可重建**：知识库改了，评测集一条命令重建，永远不会和语料版本脱节；
2. **可复现**：等距采样只依赖语料里的先后顺序，同样的输入永远产出同一份评测集，
   换分块参数 / 换 embedding 模型前后跑出来的分数才可比；
3. **无生成噪声**：reference_answer 就是语料原文，所以「答案有没有被检索到」
   是纯召回指标，不掺生成环节的干扰；
4. **分层可解释**：kind 把「语料原句问法」(corpus_qa) 和「模板改写问法」(synthetic)
   分开统计，两者召回率的差距本身就是很值钱的结论。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT = PROJECT_ROOT / "evals" / "golden_retrieval.jsonl"

QA_SOURCE = "扫地机器人 100 问 1.txt"
TROUBLE_SOURCE = "故障排除.txt"
MAINTAIN_SOURCE = "维护保养.txt"
BUYING_SOURCE = "选购指南.txt"

SECTION_RE = re.compile(r"^##\s*(.+?)\s*$")
NUMBERED_RE = re.compile(r"^(\d+)\s*[.、]\s*(.+?)\s*$")
QA_ANSWER_RE = re.compile(r"^答\s*[：:]\s*(.*)$")
TROUBLE_HEADING_RE = re.compile(r"^##\s*(\d+)\s*[.、]\s*(.+?)\s*$")
KEY_VALUE_RE = re.compile(r"^(.+?)\s*[：:]\s*(.+)$")

MAINTAIN_TEMPLATE = "扫地机器人的{head}平时怎么保养？"
BUYING_TEMPLATE = "买扫地机器人，{head}应该怎么选？"


def squeeze(text: str) -> str:
    """把连续空白压成单个空格并去首尾，避免全角空格/换行影响比对。"""
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- #
# 三类语料的解析器
# --------------------------------------------------------------------------- #
def parse_qa_file(path: pathlib.Path) -> list[dict]:
    """解析「## 章节 / N. 问题 / 答：回答」格式的问答语料。"""
    items: list[dict] = []
    section = ""
    current: dict | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        matched = SECTION_RE.match(line)
        if matched:
            section, current = matched.group(1), None
            continue

        matched = NUMBERED_RE.match(line)
        if matched:
            current = {"question": matched.group(2), "lines": [], "section": section}
            items.append(current)
            continue

        matched = QA_ANSWER_RE.match(line)
        if matched and current is not None:
            current["lines"].append(matched.group(1))
            continue

        # 答案可能跨行（多行答案直到下一个编号问题为止）
        if current is not None and current["lines"]:
            current["lines"].append(line)

    cases: list[dict] = []
    for item in items:
        question = squeeze(item["question"])
        answer = squeeze(" ".join(item["lines"]))
        if len(question) < 4 or len(answer) < 8:
            continue
        cases.append(
            {
                "question": question,
                "reference_answer": answer,
                "section": item["section"],
                "kind": "corpus_qa",
            }
        )
    return cases


def parse_troubleshoot_file(path: pathlib.Path) -> list[dict]:
    """解析「## N. 标题 + 正文」的故障手册：标题造问句，正文当参考答案。"""
    cases: list[dict] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        if not title or not body:
            return
        reference = squeeze(" ".join(body))
        if len(reference) >= 8:
            cases.append(
                {
                    "question": f"扫地机器人{title}怎么办？",
                    "reference_answer": reference,
                    "section": "故障排除",
                    "kind": "synthetic",
                }
            )

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        matched = TROUBLE_HEADING_RE.match(line)
        if matched:
            flush()
            title, body = squeeze(matched.group(2)), []
            continue
        if not line or title is None or line.startswith("#"):
            continue
        # 注意：这里不能剥掉行首的 "-"，否则参考答案和语料原文对不上，
        # 包含度会被判成 0.6 左右而误报「召回失败」。
        body.append(line)

    flush()
    return cases


def parse_numbered_list_file(path: pathlib.Path, section: str, template: str) -> list[dict]:
    """解析「N. 名目：说明」的手册：名目造问句，说明当参考答案。"""
    cases: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        matched = NUMBERED_RE.match(line)
        if not matched:
            continue
        pair = KEY_VALUE_RE.match(matched.group(2))
        if not pair:
            continue
        head, body = squeeze(pair.group(1)), squeeze(pair.group(2))
        if len(body) < 6:
            continue
        cases.append(
            {
                "question": template.format(head=head),
                "reference_answer": body,
                "section": section,
                "kind": "synthetic",
            }
        )
    return cases


# --------------------------------------------------------------------------- #
# 抽样与组装
# --------------------------------------------------------------------------- #
def sample_evenly(items: list[dict], limit: int) -> list[dict]:
    """等距抽样：覆盖各章节，且同样的输入总产出同样的评测集。"""
    if limit <= 0 or limit >= len(items):
        return list(items)
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def build(limit: int = 30, with_manuals: bool = False) -> list[dict]:
    """组装评测集：问答手册按 limit 抽样，手册合成问句全量追加。"""
    cases = sample_evenly(parse_qa_file(DATA_DIR / QA_SOURCE), limit)

    if with_manuals:
        cases += parse_troubleshoot_file(DATA_DIR / TROUBLE_SOURCE)
        cases += parse_numbered_list_file(DATA_DIR / MAINTAIN_SOURCE, "维护保养", MAINTAIN_TEMPLATE)
        cases += parse_numbered_list_file(DATA_DIR / BUYING_SOURCE, "选购指南", BUYING_TEMPLATE)

    for index, case in enumerate(cases, start=1):
        case["id"] = f"ret-{index:03d}"
        case["type"] = "retrieval"
        case["expected_tools"] = ["rag_summarize"]
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从语料生成检索评测集")
    parser.add_argument("--limit", type=int, default=30, help="问答手册抽样条数，0 = 全量")
    parser.add_argument("--with-manuals", action="store_true", help="追加三份手册生成的合成问句")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSONL 路径")
    parser.add_argument("--show", action="store_true", help="只打印，不写文件")
    args = parser.parse_args(argv)

    source_path = DATA_DIR / QA_SOURCE
    if not source_path.exists():
        print(f"找不到语料文件：{source_path}")
        print("请确认脚本放在工程根目录下的 evals/ 里。")
        return 1

    cases = build(limit=args.limit, with_manuals=args.with_manuals)

    if not args.show:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    stats: dict[str, int] = {}
    for case in cases:
        key = f"{case['kind']} / {case['section'] or '未标注章节'}"
        stats[key] = stats.get(key, 0) + 1

    print(f"生成 {len(cases)} 条检索评测用例" + ("（未写文件）" if args.show else f" -> {args.out}"))
    for key in sorted(stats):
        print(f"  {key}: {stats[key]} 条")

    if args.show:
        print("\n前 3 条预览：")
        for case in cases[:3]:
            print(f"  [{case['id']}] {case['question']}")
            print(f"        参考答案：{case['reference_answer'][:60]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
