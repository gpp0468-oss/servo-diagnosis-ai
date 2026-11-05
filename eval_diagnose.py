# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：诊断评测失败用例，输出「补什么、改什么」的建议
#   依赖：csvio, kb_category, rag_query
#   输入输出：读 eval.csv → 写 eval_diagnosis.md + 控制台汇总
#   调试提示：先看「缺案例」与「表述差异」两类，前者补 cases，后者加同义词
# ---------------------------------------------------------------------------
import re

from csvio import read_rows
from kb_category import guess_category
from rag_query import RAG_MISSING_HINT, Rag, RagIndexMismatch, describe

PUNCT = re.compile(r"[\s，。；、：（）()【】\[\]「」“”\"'.,;:!?\-]+")


def norm(s):
    return PUNCT.sub("", (s or "").lower())


def main():
    rows = [r for r in read_rows("data/eval.csv") if r.get("symptom", "").strip()]
    rules = read_rows("data/rules.csv")
    cases = read_rows("data/cases.csv")
    codes = {r["code"] for r in rules}
    cat_map = {r["code"]: r["category"] for r in rules}

    kb_text = norm(" ".join(r["cause"] + r["name"] for r in rules) + " " +
                   " ".join(c["cause"] + c["phenomenon"] for c in cases))

    try:
        rag = Rag()
    except RagIndexMismatch as exc:   # 索引错位：宁可失败，也不静默出错数
        print(str(exc))
        return 2
    except (FileNotFoundError, OSError):
        print(RAG_MISSING_HINT)
        return 2

    l0_miss, ret_miss, cat_miss = [], [], []

    for r in rows:
        sym = r["symptom"]
        up = sym.upper()
        hit = next((c for c in codes if c.upper() in up), None)

        if r.get("expected_code"):
            if hit != r["expected_code"]:
                if r["expected_code"] in codes:
                    l0_miss.append((r["id"], sym, r["expected_code"], hit,
                                    "输入里没写码或写错"))
                else:
                    l0_miss.append((r["id"], sym, r["expected_code"], hit,
                                    "规则库里没有这个码"))
        else:
            key = norm(r.get("expected_cause_keywords", ""))
            hits = rag.search(sym, 5)
            rank = next((i for i, h in enumerate(hits, 1) if key and key in norm(h["text"])), 0)
            if rank == 0:
                in_kb = key and key in kb_text
                ret_miss.append((r["id"], sym, r.get("expected_cause_keywords", ""),
                                 "库里有、表述差异" if in_kb else "库里没有、需补案例",
                                 describe(hits[0])[:40] if hits else "-"))

        cat = cat_map.get(hit, guess_category(sym))
        if cat != r.get("expected_category"):
            cat_miss.append((r["id"], sym, r.get("expected_category"), cat))

    out = ["# 评测诊断报告", "", "## 汇总", "",
           "| 类型 | 数量 | 处理方向 |", "|---|---|---|",
           "| L0 未命中 | %d | 检查码写法或补规则 |" % len(l0_miss),
           "| 检索未命中 | %d | 补案例 / 加同义词 |" % len(ret_miss),
           "| 分类错误 | %d | 补分类词库 |" % len(cat_miss), ""]

    if l0_miss:
        out += ["## 1. L0 未命中", "", "| ID | 现象 | 期望码 | 实际命中 | 判断 |", "|---|---|---|---|---|"]
        out += ["| %s | %s | %s | %s | %s |" % x for x in l0_miss]
        out.append("")

    if ret_miss:
        out += ["## 2. 检索未命中", "", "| ID | 现象 | 期望关键词 | 判断 | 当前 Top-1 |", "|---|---|---|---|---|"]
        out += ["| %s | %s | %s | %s | %s |" % x for x in ret_miss]
        out.append("")
        out += ["### 建议动作", ""]
        need = [x for x in ret_miss if "需补案例" in x[3]]
        diff = [x for x in ret_miss if "表述差异" in x[3]]
        if need:
            out.append("**补案例**（库里确实没有这些现象）：")
            for x in need:
                out.append("- `%s` → 建议新增 case：现象=%s，根因含「%s」" % (x[0], x[1], x[2]))
            out.append("")
        if diff:
            out.append("**加同义词**（库里有，但说法不同）：")
            for x in diff:
                out.append("- `%s`：%s 里没有「%s」的说法" % (x[0], x[1], x[2]))
            out.append("")

    if cat_miss:
        out += ["## 3. 分类错误", "", "| ID | 现象 | 期望 | 实际 |", "|---|---|---|---|"]
        out += ["| %s | %s | %s | %s |" % x for x in cat_miss]
        out.append("")
        by_cat = {}
        for x in cat_miss:
            by_cat.setdefault(x[2], []).append(x[1])
        out.append("### 建议补的关键词")
        for cat, syms in by_cat.items():
            out.append("- **%s**：%s" % (cat, "；".join(syms[:3])))

    open("eval_diagnosis.md", "w", encoding="utf-8").write("\n".join(out) + "\n")

    print("L0 miss:", len(l0_miss), "| ret miss:", len(ret_miss), "| cat miss:", len(cat_miss))
    for x in ret_miss:
        print("  RET", x[0], x[1][:24], "|", x[3])
    for x in cat_miss:
        print("  CAT", x[0], x[1][:24], "| exp", x[2], "-> got", x[3])
    print("报告：eval_diagnosis.md")


if __name__ == "__main__":
    raise SystemExit(main())
